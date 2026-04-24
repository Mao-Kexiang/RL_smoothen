"""高斯分布采样与预训练。

用 p(x) = N(0, std²·I_2) 作为 base model 的预训练目标，
模型学会生成高斯样本后再由 RL 微调。
"""

import torch


def sample_gaussian(n, std=2.0, dtype=torch.float32):
    """从正态分布 N(0, std²·I_2) 中采 n 个 2D 样本。"""
    return torch.randn(n, 2, dtype=dtype) * std


def pretrain_gaussian(model, n_epochs=100, batch_size=512, lr=1e-3,
                      std=2.0, dataset_size=0):
    """预训练：让模型学会按正态分布 N(0, std²·I) 生成样本。

    dataset_size > 0 时预先采好固定数据集，每 epoch 随机取 batch；
    dataset_size = 0 时每 epoch 重新采样（默认）。

    Returns:
        history: dict with 'train_loss', 'val_loss', 'val_epochs'
    """
    dtype = next(model.parameters()).dtype

    if dataset_size > 0:
        print(f"  Generating fixed dataset: {dataset_size} samples "
              f"from N(0, {std}²·I)...")
        dataset = sample_gaussian(dataset_size, std=std, dtype=dtype)
        print(f"  Dataset ready. Shape: {dataset.shape}")
    else:
        dataset = None

    rng_state = torch.random.get_rng_state()
    val_data = sample_gaussian(2048, std=std, dtype=dtype)
    torch.random.set_rng_state(rng_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_epochs = []
    best_val = float('inf')
    best_epoch = 0
    best_params = model.get_flat_params().clone()

    for epoch in range(n_epochs):
        if dataset is not None:
            idx = torch.randint(0, len(dataset), (batch_size,))
            target = dataset[idx]
        else:
            target = sample_gaussian(batch_size, std=std, dtype=dtype)

        nll = -model.log_prob(target).mean()

        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        train_losses.append(nll.item())

        if epoch % 5 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_nll = -model.log_prob(val_data).mean().item()
            model.train()
            val_losses.append(val_nll)
            val_epochs.append(epoch)
            if val_nll < best_val:
                best_val = val_nll
                best_epoch = epoch
                best_params = model.get_flat_params().clone()
            print(f"  Pretrain (Gaussian std={std}) epoch {epoch}/{n_epochs}, "
                  f"NLL={nll.item():.4f}, val_NLL={val_nll:.4f}")

    model.set_flat_params(best_params)
    print(f"  Restored best pretrain model from epoch {best_epoch} "
          f"(val_NLL={best_val:.4f})")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_epochs': val_epochs,
    }
    return history
