"""2D 离散扩散流模型：精确计算 2×2 Jacobian 行列式。

编码方向（data→noise）为 forward：
    z_0 = x
    z_{k+1} = z_k + h · v_θ(z_k, t_k)    k = 0, ..., T-1
    z_T ~ N(0, I)  (训练目标)

log_prob(x): 沿编码方向计算精确 log|det J_k|
sample(n):   z_T ~ N(0,I)，通过不动点迭代反演得到 x
"""

import math
import torch
import torch.nn as nn
import numpy as np


def sinusoidal_time_embedding(t, dim, device=None):
    """正弦时间嵌入。

    Args:
        t: (B,) 时间值 ∈ [0, 1]
        dim: 嵌入维度
    Returns:
        (B, dim) 嵌入向量
    """
    if device is None:
        device = t.device
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=device) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class VelocityNet(nn.Module):
    """速度场 v_θ(x, t) : R² × [0,1] → R²。"""

    def __init__(self, hidden_dim=128, n_hidden=3, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim

        layers = [nn.Linear(2 + time_emb_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t):
        """
        Args:
            x: (B, 2)
            t: (B,) 时间标量
        Returns:
            v: (B, 2) 速度向量
        """
        t_emb = sinusoidal_time_embedding(t, self.time_emb_dim, device=x.device)
        return self.net(torch.cat([x, t_emb], dim=-1))


class DiffusionFlow(nn.Module):
    """2D 离散扩散流，精确 Jacobian 行列式。"""

    def __init__(self, n_steps=10, hidden_dim=128, n_hidden=3,
                 time_emb_dim=32, n_fixed_point=15):
        super().__init__()
        self.n_steps = n_steps
        self.n_fixed_point = n_fixed_point
        self.velocity_net = VelocityNet(hidden_dim, n_hidden, time_emb_dim)

    def encode(self, x, training=False):
        """编码 x → z，累积精确 log|det J_k|。

        Args:
            x: (B, 2) 数据点
            training: True → create_graph=True，用于训练反传
        Returns:
            z: (B, 2) 潜变量
            total_log_det: (B,) 总 log|det|
        """
        B = x.shape[0]
        h = 1.0 / self.n_steps
        device = x.device

        z = x.clone()
        if not z.requires_grad:
            z = z.requires_grad_(True)

        total_log_det = torch.zeros(B, device=device)

        for k in range(self.n_steps):
            t_k = torch.full((B,), k * h, device=device)
            v = self.velocity_net(z, t_k)

            dv0_dz = torch.autograd.grad(
                v[:, 0].sum(), z,
                create_graph=training, retain_graph=True
            )[0]
            dv1_dz = torch.autograd.grad(
                v[:, 1].sum(), z,
                create_graph=training
            )[0]

            # det(I + h·∂v/∂z)
            det = ((1 + h * dv0_dz[:, 0]) * (1 + h * dv1_dz[:, 1])
                   - h ** 2 * dv0_dz[:, 1] * dv1_dz[:, 0])

            total_log_det = total_log_det + torch.log(det.abs().clamp(min=1e-8))
            z = z + h * v

        return z, total_log_det

    def decode(self, z):
        """解码 z → x，不动点迭代反演。

        Args:
            z: (B, 2) 潜变量（如 N(0,I) 采样）
        Returns:
            x: (B, 2) 数据空间样本
        """
        h = 1.0 / self.n_steps
        device = z.device
        B = z.shape[0]

        current = z.clone()
        for k in range(self.n_steps - 1, -1, -1):
            t_k = torch.full((B,), k * h, device=device)
            z_k = current.clone()
            for _ in range(self.n_fixed_point):
                v = self.velocity_net(z_k, t_k)
                z_k = current - h * v
            current = z_k

        return current

    def sample(self, n, device='cpu'):
        """生成 n 个样本及其精确 log p_θ(x)。

        Returns:
            x: (n, 2) 样本
            log_prob: (n,) 精确对数概率
        """
        z = torch.randn(n, 2, device=device)

        with torch.no_grad():
            x = self.decode(z)

        lp = self._log_prob_internal(x)
        return x.detach(), lp.detach()

    def log_prob(self, x):
        """精确计算 log p_θ(x)。"""
        return self._log_prob_internal(x)

    def _log_prob_internal(self, x):
        """内部实现：根据当前梯度上下文决定 create_graph。

        当外部为 no_grad 时，内部仍需 enable_grad 计算 Jacobian，
        但结果 detach 以避免 broken graph 传播到 loss.backward()。
        """
        outer_grad_enabled = torch.is_grad_enabled()

        with torch.enable_grad():
            z, log_det = self.encode(x, training=outer_grad_enabled)
            log_pz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
            result = log_pz + log_det

        if not outer_grad_enabled:
            result = result.detach()
        return result

    def get_flat_params(self):
        return torch.cat([p.data.reshape(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
            offset += numel


def pretrain_flow_matching(model, n_epochs=300, batch_size=512, lr=1e-3,
                           beta=1.0, device='cpu', dataset_size=0):
    """Flow Matching 预训练：直接回归条件速度场，不需要算 Jacobian。

    沿直线插值 z_t = (1-t)·x₀ + t·ε，目标速度 u = ε - x₀。
    Loss = E[||v_θ(z_t, t) - u||²]

    训练完后 v_θ 定义了 data→noise 的 ODE flow，
    log_prob 仍通过精确 Jacobian 计算。
    """
    from boltzmann import sample_boltzmann

    if dataset_size > 0:
        print(f"  Generating fixed dataset: {dataset_size} samples "
              f"from Boltzmann(beta={beta})...")
        dataset = sample_boltzmann(dataset_size, beta=beta).to(device)
        print(f"  Dataset ready. Shape: {dataset.shape}")
    else:
        dataset = None

    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if device != 'cpu' and torch.cuda.is_available() else None
    val_data = sample_boltzmann(2048, beta=beta).to(device)
    torch.random.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    train_losses = []
    val_losses = []
    val_epochs = []
    best_val = float('inf')
    best_epoch = 0
    best_params = model.get_flat_params().clone().cpu()

    for epoch in range(n_epochs):
        if dataset is not None:
            idx = torch.randint(0, len(dataset), (batch_size,), device=device)
            x0 = dataset[idx]
        else:
            x0 = sample_boltzmann(batch_size, beta=beta).to(device)

        eps = torch.randn_like(x0)
        t = torch.rand(batch_size, device=device)

        z_t = (1 - t).unsqueeze(-1) * x0 + t.unsqueeze(-1) * eps
        target_v = eps - x0

        pred_v = model.velocity_net(z_t, t)
        loss = (pred_v - target_v).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        train_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            rng_state_val = torch.random.get_rng_state()
            cuda_rng_val = torch.cuda.get_rng_state() if device != 'cpu' and torch.cuda.is_available() else None
            model.eval()
            with torch.no_grad():
                val_eps = torch.randn_like(val_data)
                val_t = torch.rand(len(val_data), device=device)
                val_zt = (1 - val_t).unsqueeze(-1) * val_data + val_t.unsqueeze(-1) * val_eps
                val_target_v = val_eps - val_data
                val_pred_v = model.velocity_net(val_zt, val_t)
                val_loss = (val_pred_v - val_target_v).pow(2).mean().item()

                x_sample = model.decode(torch.randn(256, 2, device=device))
                from env import reward
                mean_r = reward(x_sample).mean().item()
            model.train()
            torch.random.set_rng_state(rng_state_val)
            if cuda_rng_val is not None:
                torch.cuda.set_rng_state(cuda_rng_val)
            val_losses.append(val_loss)
            val_epochs.append(epoch)
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                best_params = model.get_flat_params().clone().cpu()
            print(f"  FM epoch {epoch}/{n_epochs}, loss={loss.item():.4f}, "
                  f"val_loss={val_loss:.4f}, sample_mean_r={mean_r:.2f}")

    model.set_flat_params(best_params.to(device))
    print(f"  Restored best pretrain model from epoch {best_epoch} "
          f"(val_loss={best_val:.4f})")
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_epochs': val_epochs,
    }
    return history


def pretrain_diffusion(model, n_epochs=300, batch_size=512, lr=1e-3, beta=1.0,
                       device='cpu'):
    """在 Boltzmann 分布上预训练扩散流模型。

    数据集：p(x) ∝ exp(-β · Rastrigin(x))，使用拒绝采样。

    Returns:
        history: dict with 'train_loss', 'val_loss', 'val_epochs'
    """
    from boltzmann import sample_boltzmann

    rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if device != 'cpu' and torch.cuda.is_available() else None
    val_data = sample_boltzmann(2048, beta=beta).to(device)
    torch.random.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_nll = float('inf')
    best_val_nll = float('inf')
    best_epoch = 0
    best_params = model.get_flat_params().clone().cpu()

    train_losses = []
    val_losses = []
    val_epochs = []

    for epoch in range(n_epochs):
        target = sample_boltzmann(batch_size, beta=beta).to(device)
        nll = -model.log_prob(target).mean()

        if not torch.isfinite(nll):
            model.set_flat_params(best_params.clone().to(device))
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.5
            print(f"  [Epoch {epoch}] NaN detected, rollback + halve lr")
            continue

        optimizer.zero_grad()
        nll.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        nll_val = nll.item()
        train_losses.append(nll_val)
        if nll_val < best_nll:
            best_nll = nll_val

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                val_nll = -model.log_prob(val_data).mean().item()
            model.train()
            val_losses.append(val_nll)
            val_epochs.append(epoch)
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                best_epoch = epoch
                best_params = model.get_flat_params().clone().cpu()
            print(f"  Pretrain epoch {epoch}/{n_epochs}, "
                  f"NLL={nll_val:.4f}, val_NLL={val_nll:.4f}")

    model.set_flat_params(best_params.to(device))
    print(f"  Restored best pretrain model from epoch {best_epoch} "
          f"(val_NLL={best_val_nll:.4f})")
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_epochs': val_epochs,
    }
    return history
