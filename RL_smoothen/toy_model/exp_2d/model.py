"""流模型：支持仿射耦合（RealNVP）和样条耦合（Neural Spline Flow）。

Neural Spline Flow 使用有理二次样条替代仿射变换，
每层可以做复杂的非线性单调映射，能高效表达多峰分布。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# ======== Rational Quadratic Spline Transform ========

def rational_quadratic_spline(inputs, widths, heights, derivatives,
                               inverse=False, tail_bound=5.0):
    """有理二次样条变换 (Durkan et al., 2019)。

    Args:
        inputs: (batch,) 待变换值
        widths: (batch, K) 未归一化的 bin 宽度
        heights: (batch, K) 未归一化的 bin 高度
        derivatives: (batch, K+1) 未归一化的节点导数
        inverse: 是否计算逆变换
        tail_bound: B，[-B, B] 外为恒等变换

    Returns:
        outputs: (batch,) 变换后的值
        log_det: (batch,) log |dy/dx|
    """
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # 归一化参数
    widths = F.softmax(widths, dim=-1) * 2 * tail_bound
    heights = F.softmax(heights, dim=-1) * 2 * tail_bound
    derivatives = F.softplus(derivatives) + 1e-3

    # 累积 → 节点位置
    cumwidths = F.pad(torch.cumsum(widths, dim=-1), (1, 0)) - tail_bound
    cumheights = F.pad(torch.cumsum(heights, dim=-1), (1, 0)) - tail_bound

    # 找到每个输入所在的 bin
    if inverse:
        bin_idx = torch.searchsorted(cumheights[:, 1:].contiguous(),
                                     inputs.unsqueeze(-1)).squeeze(-1)
    else:
        bin_idx = torch.searchsorted(cumwidths[:, 1:].contiguous(),
                                     inputs.unsqueeze(-1)).squeeze(-1)
    bin_idx = bin_idx.clamp(0, widths.shape[-1] - 1)

    # 收集当前 bin 的参数
    idx = bin_idx.unsqueeze(-1)
    w_k = widths.gather(-1, idx).squeeze(-1)
    h_k = heights.gather(-1, idx).squeeze(-1)
    x_k = cumwidths.gather(-1, idx).squeeze(-1)
    y_k = cumheights.gather(-1, idx).squeeze(-1)
    d_k = derivatives.gather(-1, idx).squeeze(-1)
    d_k1 = derivatives.gather(-1, idx + 1).squeeze(-1)
    s_k = h_k / w_k

    if inverse:
        # 解二次方程求 ξ
        a = h_k * (s_k - d_k) + (inputs - y_k) * (d_k1 + d_k - 2 * s_k)
        b = h_k * d_k - (inputs - y_k) * (d_k1 + d_k - 2 * s_k)
        c = -s_k * (inputs - y_k)

        discriminant = (b.pow(2) - 4 * a * c).clamp(min=0)
        xi = (2 * c) / (-b - discriminant.sqrt())
        xi = xi.clamp(0, 1)

        outputs = xi * w_k + x_k

        # log det (正向的取负)
        denom = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
        numer = s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi)
                              + d_k * (1 - xi).pow(2))
        log_det = -(numer.log() - 2 * denom.log())
    else:
        xi = ((inputs - x_k) / w_k).clamp(0, 1)

        denom = s_k + (d_k1 + d_k - 2 * s_k) * xi * (1 - xi)
        outputs = y_k + h_k * (s_k * xi.pow(2) + d_k * xi * (1 - xi)) / denom

        numer = s_k.pow(2) * (d_k1 * xi.pow(2) + 2 * s_k * xi * (1 - xi)
                              + d_k * (1 - xi).pow(2))
        log_det = numer.log() - 2 * denom.log()

    outputs = torch.where(inside, outputs, inputs)
    log_det = torch.where(inside, log_det, torch.zeros_like(log_det))

    return outputs, log_det


# ======== Initialization ========

def _init_net(net, init='small'):
    """初始化网络权重。

    'small': 所有权重 ~ N(0, 0.01)，最后一层零初始化 → 初始变换接近恒等
    'kaiming': 隐藏层 Kaiming 初始化，偏置零，最后一层零 → 初始输出 ~ N(0, 1)
    """
    if init == 'kaiming':
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    else:
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0.0, 0.01)
                nn.init.normal_(layer.bias, 0.0, 0.01)
    # 最后一层零初始化，保证初始时整个耦合层是恒等变换
    last = [m for m in net if isinstance(m, nn.Linear)][-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)


# ======== Coupling Layers ========

class AffineCoupling(nn.Module):
    """2D 仿射耦合层：y_var = x_var * exp(s(x_fix)) + t(x_fix)。"""

    @staticmethod
    def _build_net(hidden_dim, init='small'):
        net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        _init_net(net, init)
        return net

    def __init__(self, fix_dim, hidden_dim=64, init='small'):
        super().__init__()
        self.fix_dim = fix_dim
        self.transform_dim = 1 - fix_dim
        self.s_net = self._build_net(hidden_dim, init)
        self.t_net = self._build_net(hidden_dim, init)

    def forward(self, x):
        x_fix = x[:, self.fix_dim:self.fix_dim + 1]
        x_var = x[:, self.transform_dim:self.transform_dim + 1]

        s = self.s_net(x_fix).clamp(-8, 8)
        t = self.t_net(x_fix)

        y_var = x_var * s.exp() + t
        log_det = s.squeeze(-1)

        if self.fix_dim == 0:
            y = torch.cat([x_fix, y_var], dim=1)
        else:
            y = torch.cat([y_var, x_fix], dim=1)
        return y, log_det

    def inverse(self, y):
        y_fix = y[:, self.fix_dim:self.fix_dim + 1]
        y_var = y[:, self.transform_dim:self.transform_dim + 1]

        s = self.s_net(y_fix).clamp(-8, 8)
        t = self.t_net(y_fix)

        x_var = (y_var - t) * (-s).exp()
        log_det = -s.squeeze(-1)

        if self.fix_dim == 0:
            x = torch.cat([y_fix, x_var], dim=1)
        else:
            x = torch.cat([x_var, y_fix], dim=1)
        return x, log_det


class SplineCoupling(nn.Module):
    """2D 样条耦合层：用有理二次样条替代仿射变换。"""

    def __init__(self, fix_dim, hidden_dim=64, n_bins=8, tail_bound=5.0, init='small'):
        super().__init__()
        self.fix_dim = fix_dim
        self.transform_dim = 1 - fix_dim
        self.n_bins = n_bins
        self.tail_bound = tail_bound

        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3 * n_bins + 1),
        )
        _init_net(self.net, init)

    def _get_params(self, x_fix):
        raw = self.net(x_fix)
        W = raw[:, :self.n_bins]
        H = raw[:, self.n_bins:2 * self.n_bins]
        D = raw[:, 2 * self.n_bins:]
        return W, H, D

    def forward(self, x):
        x_fix = x[:, self.fix_dim:self.fix_dim + 1]
        x_var = x[:, self.transform_dim]

        W, H, D = self._get_params(x_fix)
        y_var, log_det = rational_quadratic_spline(
            x_var, W, H, D, inverse=False, tail_bound=self.tail_bound)

        if self.fix_dim == 0:
            y = torch.cat([x_fix, y_var.unsqueeze(-1)], dim=1)
        else:
            y = torch.cat([y_var.unsqueeze(-1), x_fix], dim=1)
        return y, log_det

    def inverse(self, y):
        y_fix = y[:, self.fix_dim:self.fix_dim + 1]
        y_var = y[:, self.transform_dim]

        W, H, D = self._get_params(y_fix)
        x_var, log_det = rational_quadratic_spline(
            y_var, W, H, D, inverse=True, tail_bound=self.tail_bound)

        if self.fix_dim == 0:
            x = torch.cat([y_fix, x_var.unsqueeze(-1)], dim=1)
        else:
            x = torch.cat([x_var.unsqueeze(-1), y_fix], dim=1)
        return x, log_det


# ======== Flow Model ========

class FlowModel(nn.Module):
    """流模型：z ~ N(0,I_2) -> 耦合层 -> x in R^2。

    coupling='spline': Neural Spline Flow（默认，多峰表达力强）
    coupling='affine': RealNVP（仿射耦合，向后兼容）
    """

    def __init__(self, n_layers=8, hidden_dim=64, coupling='spline', n_bins=8,
                 init='small'):
        super().__init__()
        if coupling == 'spline':
            self.layers = nn.ModuleList([
                SplineCoupling(fix_dim=i % 2, hidden_dim=hidden_dim,
                               n_bins=n_bins, init=init)
                for i in range(n_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                AffineCoupling(fix_dim=i % 2, hidden_dim=hidden_dim, init=init)
                for i in range(n_layers)
            ])

    def forward(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device)
        x = z
        for layer in self.layers:
            x, ld = layer(x)
            log_det = log_det + ld
        return x, log_det

    def inverse(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)
        z = x
        for layer in reversed(self.layers):
            z, ld = layer.inverse(z)
            log_det = log_det + ld
        return z, log_det

    def sample(self, n, device='cpu'):
        dtype = next(self.parameters()).dtype
        z = torch.randn(n, 2, device=device, dtype=dtype)
        x, log_det_fwd = self.forward(z)
        log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)
        log_prob = log_pz - log_det_fwd
        return x, log_prob

    def log_prob(self, x):
        z, log_det_inv = self.inverse(x)
        log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)
        return log_pz + log_det_inv

    def get_flat_params(self):
        return torch.cat([p.data.reshape(-1) for p in self.parameters()])

    def set_flat_params(self, flat):
        offset = 0
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
            offset += numel


@torch.no_grad()
def check_normalization(model, bound=8.0, n_grid=500, batch_size=10000):
    """数值积分验证 ∫p(x)dx ≈ 1。

    在 [-bound, bound]² 上铺 n_grid×n_grid 的均匀网格，
    用矩形法（Riemann sum）计算 ∑ exp(log_prob(x_i)) · dx²。

    Args:
        model: FlowModel
        bound: 积分区域 [-bound, bound]²
        n_grid: 每个维度的网格点数
        batch_size: 每批计算的点数（避免 OOM）

    Returns:
        integral: float, 积分值（理想 = 1.0）
    """
    model.eval()
    dtype = next(model.parameters()).dtype
    device = next(model.parameters()).device

    xs = torch.linspace(-bound, bound, n_grid, dtype=dtype, device=device)
    dx = xs[1] - xs[0]
    cell_area = (dx ** 2).item()

    grid_x, grid_y = torch.meshgrid(xs, xs, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    log_probs = []
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        lp = model.log_prob(batch)
        log_probs.append(lp)
    log_probs = torch.cat(log_probs)

    n_total = log_probs.numel()
    valid = torch.isfinite(log_probs)
    n_bad = (~valid).sum().item()
    log_probs = log_probs[valid]

    log_integral = torch.logsumexp(log_probs, dim=0) + np.log(cell_area)
    integral = log_integral.exp().item()

    print(f"  Normalization check: ∫p(x)dx = {integral:.6f}  "
          f"(grid={n_grid}x{n_grid}, bound=[-{bound},{bound}]²)")
    if n_bad > 0:
        print(f"  Warning: {n_bad}/{n_total} grid points had NaN/Inf log_prob "
              f"({n_bad/n_total*100:.2f}%) — excluded from integral")
    model.train()
    return integral


def pretrain(model, n_epochs=200, batch_size=512, lr=1e-3):
    """预训练：让模型学会在 [-5, 5]^2 上均匀生成样本。

    Returns:
        history: dict with 'train_loss', 'val_loss', 'val_epochs'
    """
    dtype = next(model.parameters()).dtype

    rng_state = torch.random.get_rng_state()
    val_data = torch.rand(2048, 2, dtype=dtype) * 10 - 5
    torch.random.set_rng_state(rng_state)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    val_epochs = []
    best_val = float('inf')
    best_epoch = 0
    best_params = model.get_flat_params().clone()

    for epoch in range(n_epochs):
        target = torch.rand(batch_size, 2, dtype=dtype) * 10 - 5
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
            print(f"  Pretrain epoch {epoch}/{n_epochs}, "
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


if __name__ == '__main__':
    import os
    import sys
    import argparse

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from ppo import ppo_train
    from boltzmann import sample_boltzmann, pretrain_boltzmann
    from gaussian import sample_gaussian, pretrain_gaussian
    from direct_optim import cmaes_optimize
    from fim import (compute_pca_directions, compute_trajectory_fim,
                     compute_theta_landscape)
    from visualize import (
        plot_landscape_comparison, plot_landscape_3d,
        plot_trajectory_animation, plot_convergence,
        plot_dataset, plot_distributions, plot_distribution_evolution,
        plot_fim_evolution, plot_step_comparison,
        plot_kl_efficiency, plot_trajectory_ellipses,
        plot_theorem1_check, plot_pretrain_loss,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='boltzmann',
                        choices=['boltzmann', 'gaussian'],
                        help='Pretrain distribution: boltzmann or gaussian')
    parser.add_argument('--gaussian_std', type=float, default=2.0,
                        help='Std for gaussian pretrain distribution')
    parser.add_argument('--scan', action='store_true')
    parser.add_argument('--beta', type=float, default=0.0)
    parser.add_argument('--n_layers', type=int, default=8)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--coupling', type=str, default='affine')
    parser.add_argument('--init', type=str, default='small',
                        choices=['small', 'kaiming'],
                        help='Weight init: small (N(0,0.01)) or kaiming')
    parser.add_argument('--pretrain_epochs', type=int, default=1000)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_batch', type=int, default=512)
    parser.add_argument('--dataset_size', type=int, default=0,
                        help='Fixed dataset size for pretraining (0=resample each epoch)')
    parser.add_argument('--warmup_epochs', type=int, default=100,
                        help='LR warmup epochs for pretraining (0=no warmup)')
    parser.add_argument('--ppo_iters', type=int, default=200)
    parser.add_argument('--ppo_epochs', type=int, default=2)
    parser.add_argument('--ppo_lr', type=float, default=3e-4)
    parser.add_argument('--ppo_kl', type=float, default=0.5)
    parser.add_argument('--ppo_batch', type=int, default=256)
    parser.add_argument('--diverge_thresh', type=float, default=5.0)
    parser.add_argument('--dtype', type=str, default='float32',
                        choices=['float32', 'float64'],
                        help='Precision: float32 (default) or float64')
    args = parser.parse_args()

    output_dir = os.path.join(ROOT, 'output', 'flow')
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    dtype = torch.float64 if args.dtype == 'float64' else torch.float32

    # Step 1: 预训练
    print("=" * 60)
    print(f"Step 1: Building FlowModel + {args.pretrain} Pretraining...")
    print(f"  dtype: {args.dtype}, init: {args.init}")
    print("=" * 60)
    model = FlowModel(n_layers=args.n_layers, hidden_dim=args.hidden_dim,
                      coupling=args.coupling, init=args.init)
    if dtype == torch.float64:
        model = model.double()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: layers={args.n_layers}, hidden={args.hidden_dim}, "
          f"coupling={args.coupling}, params={n_params:,}")
    if args.pretrain == 'gaussian':
        pretrain_history = pretrain_gaussian(model, n_epochs=args.pretrain_epochs,
                          batch_size=args.pretrain_batch,
                          lr=args.pretrain_lr, std=args.gaussian_std,
                          dataset_size=args.dataset_size)
    else:
        pretrain_history = pretrain_boltzmann(model, n_epochs=args.pretrain_epochs,
                           batch_size=args.pretrain_batch,
                           lr=args.pretrain_lr, beta=args.beta,
                           dataset_size=args.dataset_size,
                           warmup_epochs=args.warmup_epochs)
    pretrain_params = model.get_flat_params().clone()

    print("\n  Verifying pretrained model normalization...")
    check_normalization(model, bound=8.0, n_grid=500)

    # Step 2: PPO
    print("\n" + "=" * 60)
    print("Step 2: PPO Training (RL fine-tuning)...")
    print("=" * 60)
    ppo_history = ppo_train(
        model,
        n_iters=args.ppo_iters,
        batch_size=args.ppo_batch,
        ppo_epochs=args.ppo_epochs,
        clip_eps=0.2,
        lr=args.ppo_lr,
        kl_coeff=args.ppo_kl,
        save_every=1,
        diverge_threshold=args.diverge_thresh,
    )

    # Step 3: CMA-ES
    print("\n" + "=" * 60)
    print("Step 3: CMA-ES Direct Optimization...")
    print("=" * 60)
    cmaes_history = cmaes_optimize(n_evals=51200, sigma0=3.0, seed=42)

    # Step 4: PCA 方向 + 沿轨迹 FIM + 可选 2D 网格扫描
    print("\n" + "=" * 60)
    print("Step 4: Computing PCA directions + Trajectory FIM...")
    print("=" * 60)

    d1, d2, pca_center, traj_alpha, traj_beta, explained_var = \
        compute_pca_directions(ppo_history['param_snapshots'])

    print(f"\n  Computing trajectory FIM (201 points)...")
    traj_fim = compute_trajectory_fim(
        model, ppo_history['param_snapshots'], d1, d2,
        n_samples=1000, trim_ratio=0.1)

    theta_landscape = None
    if args.scan:
        print(f"\n  Scanning 2D grid (--scan enabled)...")
        theta_landscape = compute_theta_landscape(
            model, ppo_history['param_snapshots'],
            d1, d2, pca_center, traj_alpha, traj_beta,
            n_grid=21, grid_range=1.2,
            n_eval_samples=500, n_fim_samples=100,
        )

    # Step 5: 可视化
    print("\n" + "=" * 60)
    print("Step 5: Generating visualizations...")
    print("=" * 60)

    if args.pretrain == 'gaussian':
        pt_title = f'Pretrain Loss (Gaussian std={args.gaussian_std})'
    else:
        pt_title = f'Pretrain Loss (Boltzmann $\\beta$={args.beta})'
    plot_pretrain_loss(pretrain_history,
                       save_path=os.path.join(output_dir, 'fig0_pretrain_loss.png'),
                       title=pt_title)

    show_fim = theta_landscape is not None
    if show_fim:
        plot_landscape_comparison(theta_landscape,
                                  save_path=os.path.join(output_dir, 'fig1_landscape.png'),
                                  show_fim=True)

    plot_trajectory_animation(cmaes_history, ppo_history,
                              save_path=os.path.join(output_dir, 'fig2_trajectory.gif'),
                              ppo_batch_size=args.ppo_batch)

    plot_convergence(cmaes_history, ppo_history,
                     batch_size=args.ppo_batch,
                     save_path=os.path.join(output_dir, 'fig3_convergence.png'))

    if show_fim:
        plot_landscape_3d(theta_landscape,
                          save_path=os.path.join(output_dir, 'fig4_landscape_3d.png'),
                          show_fim=True)

    if args.pretrain == 'gaussian':
        dataset_pts = sample_gaussian(2048, std=args.gaussian_std)
        dataset_title = f'Pretraining Dataset (Gaussian std={args.gaussian_std})'
    else:
        dataset_pts = sample_boltzmann(2048, beta=args.beta)
        dataset_title = f'Pretraining Dataset (Boltzmann β={args.beta})'
    plot_dataset(dataset_pts,
                 save_path=os.path.join(output_dir, 'fig5_dataset.png'),
                 title=dataset_title)

    ppo_params = model.get_flat_params().clone()
    plot_distributions(model, pretrain_params, ppo_params,
                       save_path=os.path.join(output_dir, 'fig6_distributions.png'))

    plot_distribution_evolution(model, ppo_history['param_snapshots'],
                                save_path=os.path.join(output_dir, 'fig7_distribution_evolution.gif'))

    # 沿轨迹 FIM 图
    plot_fim_evolution(traj_fim, ppo_history,
                       save_path=os.path.join(output_dir, 'fig8_fim_evolution.png'))

    plot_step_comparison(traj_fim,
                         save_path=os.path.join(output_dir, 'fig9_step_comparison.png'))

    plot_kl_efficiency(traj_fim, ppo_history,
                       save_path=os.path.join(output_dir, 'fig10_kl_efficiency.png'))

    if show_fim:
        plot_trajectory_ellipses(theta_landscape, traj_fim,
                                  save_path=os.path.join(output_dir, 'fig11_trajectory_ellipses.png'))

    plot_theorem1_check(traj_fim, ppo_history,
                        save_path=os.path.join(output_dir, 'fig12_theorem1.png'))

    print("\n" + "=" * 60)
    final_mean = ppo_history['mean_reward'][-1]
    ppo_best = max(ppo_history['best_reward'])
    cma_best = cmaes_history['final_reward']

    # 第一次达到 best reward 的时机
    ppo_first_idx = next(i for i, r in enumerate(ppo_history['best_reward'])
                         if r >= ppo_best)
    ppo_first_evals = ppo_first_idx * args.ppo_batch

    cma_best_rewards = cmaes_history['best_rewards']
    cma_first_idx = next(i for i, r in enumerate(cma_best_rewards)
                         if r >= cma_best)
    cma_first_evals = cmaes_history['n_evals_list'][cma_first_idx]

    print(f"Results (2D Rastrigin, FlowModel {args.coupling}):")
    print(f"  PPO final mean reward:   {final_mean:.4f}")
    print(f"  PPO best reward:         {ppo_best:.4f}  (first at iter {ppo_first_idx}, evals={ppo_first_evals})")
    print(f"  CMA-ES best reward:      {cma_best:.4f}  (first at evals={cma_first_evals})")
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 60)
