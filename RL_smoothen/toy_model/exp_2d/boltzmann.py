"""Boltzmann 分布采样与预训练。

用 p(x) ∝ exp(-β · Rastrigin(x)) 替代均匀分布，
使 base model 集中在 r(x) 极大值附近。
"""

import torch
from env import rastrigin


def sample_boltzmann(n, beta=0.1, dtype=torch.float32):
    """拒绝采样：从 Boltzmann 分布 p(x) ∝ exp(-β·Rastrigin(x)) 中采 n 个 2D 样本。

    因为 Rastrigin(x) ≥ 0，接受概率 exp(-β·R(x)) ≤ 1，
    以 Uniform[-5,5]² 为提议分布即可。
    """
    samples = []
    collected = 0
    while collected < n:
        batch = max(n * 50, 1024)
        x = torch.rand(batch, 2, dtype=dtype) * 10 - 5
        accept_prob = torch.exp(-beta * rastrigin(x))
        mask = torch.rand(batch) < accept_prob
        accepted = x[mask]
        if len(accepted) > 0:
            samples.append(accepted)
            collected += len(accepted)
    return torch.cat(samples)[:n]


def pretrain_boltzmann(model, n_epochs=100, batch_size=512, lr=1e-3, beta=0.1,
                       dataset_size=0, warmup_epochs=100):
    """预训练：让模型学会按 Boltzmann 分布生成样本。

    dataset_size > 0 时预先采好固定数据集，每 epoch 随机取 batch；
    dataset_size = 0 时每 epoch 重新采样（默认）。

    Returns:
        history: dict with 'train_loss', 'val_loss', 'val_epochs'
    """
    dtype = next(model.parameters()).dtype

    if dataset_size > 0:
        print(f"  Generating fixed dataset: {dataset_size} samples "
              f"from Boltzmann(beta={beta})...")
        dataset = sample_boltzmann(dataset_size, beta=beta, dtype=dtype)
        print(f"  Dataset ready. Shape: {dataset.shape}")
    else:
        dataset = None

    rng_state = torch.random.get_rng_state()
    val_data = sample_boltzmann(2048, beta=beta, dtype=dtype)
    torch.random.set_rng_state(rng_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=warmup_epochs)
    else:
        warmup_sched = None

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
            target = sample_boltzmann(batch_size, beta=beta, dtype=dtype)

        nll = -model.log_prob(target).mean()

        optimizer.zero_grad()
        nll.backward()
        optimizer.step()

        train_losses.append(nll.item())

        if warmup_sched is not None and epoch < warmup_epochs:
            warmup_sched.step()

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
            print(f"  Pretrain (Boltzmann β={beta}) epoch {epoch}/{n_epochs}, "
                  f"NLL={nll.item():.4f}, val_NLL={val_nll:.4f}, "
                  f"lr={optimizer.param_groups[0]['lr']:.2e}")

    model.set_flat_params(best_params)
    print(f"  Restored best pretrain model from epoch {best_epoch} "
          f"(val_NLL={best_val:.4f})")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_epochs': val_epochs,
    }
    return history
