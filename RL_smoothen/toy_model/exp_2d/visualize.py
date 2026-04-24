"""可视化模块：三组对比图 + 动画 + 3D 曲面图 + 沿轨迹 FIM 分析图"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from env import rastrigin, reward
import torch


plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150


def plot_rastrigin_landscape(ax, xlim=(-5, 5), ylim=(-5, 5), n_grid=300):
    """在 ax 上画 Rastrigin 函数的热力图。"""
    x1 = np.linspace(*xlim, n_grid)
    x2 = np.linspace(*ylim, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid = torch.tensor(np.stack([X1, X2], axis=-1), dtype=torch.float32)
    R = -rastrigin(grid).numpy()

    im = ax.contourf(X1, X2, R, levels=50, cmap='viridis')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_aspect('equal')
    return im


# ======== 图1：三面板对比 ========

def plot_landscape_comparison(theta_landscape, save_path='fig1_landscape.png', show_fim=False):
    """二/三面板：r(x) 原始空间 | J(θ) 欧氏 θ 空间 [| J(θ) FIM 非均匀网格]。"""
    alphas = theta_landscape['alphas']
    betas = theta_landscape['betas']
    J_grid = theta_landscape['J_grid']

    # 统一 colorbar 范围
    vmin = J_grid.min()
    vmax = J_grid.max()
    levels = np.linspace(vmin, vmax, 50)

    n_panels = 3 if show_fim else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.5))

    # --- 面板1：r(x) 原始空间 ---
    im0 = plot_rastrigin_landscape(axes[0])
    axes[0].set_title('Objective $r(x)$ in original space\n(Extremely rugged)', fontsize=13)
    axes[0].plot(0, 0, 'r*', markersize=15, zorder=5, label='Global optimum')
    axes[0].legend(loc='upper right', fontsize=9)
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im0, cax=cax0, label='$r(x)$')

    # --- 面板2：J(θ) 欧氏 PCA 切面 ---
    A_e, B_e = np.meshgrid(alphas, betas)
    im1 = axes[1].contourf(A_e, B_e, J_grid, levels=levels, cmap='viridis',
                            extend='both')
    traj_a = theta_landscape['traj_alpha']
    traj_b = theta_landscape['traj_beta']
    axes[1].plot(traj_a, traj_b, 'r-', linewidth=1.2, alpha=0.6, zorder=4)
    axes[1].plot(traj_a[0], traj_b[0], 'o', color='blue', markersize=9,
                 zorder=6, label='Start')
    axes[1].plot(traj_a[-1], traj_b[-1], '^', color='red', markersize=11,
                 zorder=6, label='End')
    axes[1].set_xlabel('$d_1$ (start→end)')
    axes[1].set_ylabel('$d_2$ (max ⊥ variance)')
    axes[1].set_title('$J(\\theta)$ in Euclidean $\\theta$-space\n(2D slice)', fontsize=13)
    axes[1].legend(loc='lower right', fontsize=9)
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax1, label='$J(\\theta)$')

    if show_fim:
        # --- 面板3：J(θ) 在 FIM / KL 坐标下（均匀 KL 网格） ---
        kl_alphas_u = theta_landscape['kl_alphas_uniform']
        kl_betas_u = theta_landscape['kl_betas_uniform']
        J_kl = theta_landscape['J_kl_grid']

        A_kl, B_kl = np.meshgrid(kl_alphas_u, kl_betas_u)
        im2 = axes[2].contourf(A_kl, B_kl, J_kl, levels=levels, cmap='viridis',
                                extend='both')
        axes[2].set_xlim(kl_alphas_u[0], kl_alphas_u[-1])
        axes[2].set_ylim(kl_betas_u[0], kl_betas_u[-1])

        kl_alpha_2d = np.clip(theta_landscape['kl_alpha_2d'],
                              kl_alphas_u[0], kl_alphas_u[-1])
        kl_beta_2d = np.clip(theta_landscape['kl_beta_2d'],
                             kl_betas_u[0], kl_betas_u[-1])
        n_grid = len(theta_landscape['alphas'])
        grid_step = max(1, n_grid // 10)
        for j in range(0, n_grid, grid_step):
            axes[2].plot(kl_alpha_2d[j, :], kl_beta_2d[j, :],
                         'w-', linewidth=0.3, alpha=0.4)
        for i in range(0, n_grid, grid_step):
            axes[2].plot(kl_alpha_2d[:, i], kl_beta_2d[:, i],
                         'w-', linewidth=0.3, alpha=0.4)

        traj_af = theta_landscape['traj_alpha_kl']
        traj_bf = theta_landscape['traj_beta_kl']
        axes[2].plot(traj_af, traj_bf, 'r-', linewidth=1.2, alpha=0.6, zorder=4)
        axes[2].plot(traj_af[0], traj_bf[0], 'o', color='blue', markersize=9,
                     zorder=6, label='Start')
        axes[2].plot(traj_af[-1], traj_bf[-1], '^', color='red', markersize=11,
                     zorder=6, label='End')
        axes[2].set_xlabel('$\\sqrt{\\mathrm{KL}}$ along PC1')
        axes[2].set_ylabel('$\\sqrt{\\mathrm{KL}}$ along PC2')
        axes[2].set_title('$J(\\theta)$ in Fisher Information metric\n(non-uniform grid from full 2D FIM)', fontsize=13)
        axes[2].legend(loc='lower right', fontsize=9)
        divider2 = make_axes_locatable(axes[2])
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax2, label='$J(\\theta)$')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图4：三面板 3D 曲面图 ========

def plot_landscape_3d(theta_landscape, save_path='fig4_landscape_3d.png', show_fim=False):
    """二/三面板 3D 曲面图：r(x) | J(θ) 欧氏 [| J(θ) FIM]，与 fig1 数据一致。"""
    from scipy.interpolate import RegularGridInterpolator

    alphas = theta_landscape['alphas']
    betas = theta_landscape['betas']
    J_grid = theta_landscape['J_grid']

    vmin_j = J_grid.min()
    vmax_j = J_grid.max()
    norm_j = Normalize(vmin=vmin_j, vmax=vmax_j)
    z_lift = 0

    n_panels = 3 if show_fim else 2
    fig = plt.figure(figsize=(11 * n_panels, 6.5))

    # --- Panel 1: r(x) 3D surface ---
    ax0 = fig.add_subplot(1, n_panels, 1, projection='3d')
    n_surf = 200
    x1 = np.linspace(-5, 5, n_surf)
    x2 = np.linspace(-5, 5, n_surf)
    X1, X2 = np.meshgrid(x1, x2)
    grid = torch.tensor(np.stack([X1, X2], axis=-1), dtype=torch.float32)
    R = -rastrigin(grid).numpy()

    norm_r = Normalize(vmin=R.min(), vmax=R.max())
    colors_r = cm.viridis(norm_r(R))
    ax0.plot_surface(X1, X2, R, facecolors=colors_r, rstride=2, cstride=2,
                     linewidth=0, antialiased=True, shade=False)
    ax0.set_xlabel('$x_1$')
    ax0.set_ylabel('$x_2$')
    ax0.set_zlabel('$r(x)$')
    ax0.set_title('Objective $r(x)$ in original space', fontsize=12)
    ax0.view_init(elev=35, azim=225)

    # --- Panel 2: J(θ) Euclidean 3D surface ---
    ax1 = fig.add_subplot(1, n_panels, 2, projection='3d')
    A_e, B_e = np.meshgrid(alphas, betas)
    colors_j = cm.viridis(norm_j(J_grid))
    colors_j[..., 3] = 0.6  # 半透明，让轨迹可见
    ax1.plot_surface(A_e, B_e, J_grid, facecolors=colors_j, rstride=1, cstride=1,
                     linewidth=0, antialiased=True, shade=False)

    # PPO trajectory on surface (interpolated)
    traj_a = np.asarray(theta_landscape['traj_alpha'])
    traj_b = np.asarray(theta_landscape['traj_beta'])
    interp_j = RegularGridInterpolator(
        (betas, alphas), J_grid, method='linear', bounds_error=False,
        fill_value=None)
    traj_z = interp_j(np.column_stack([traj_b, traj_a])) + z_lift
    ax1.plot3D(traj_a, traj_b, traj_z, 'r-', linewidth=2.5, alpha=1.0)
    ax1.scatter([traj_a[0]], [traj_b[0]], [traj_z[0]],
                color='blue', s=60, zorder=5, depthshade=False)
    ax1.scatter([traj_a[-1]], [traj_b[-1]], [traj_z[-1]],
                color='red', s=70, marker='^', zorder=5, depthshade=False)

    ax1.set_xlabel('$d_1$ (start→end)')
    ax1.set_ylabel('$d_2$ (max ⊥ variance)')
    ax1.set_zlabel('$J(\\theta)$')
    ax1.set_title('$J(\\theta)$ in Euclidean $\\theta$-space', fontsize=12)
    ax1.view_init(elev=35, azim=225)

    if show_fim:
        # --- Panel 3: J(θ) FIM / KL 3D surface ---
        ax2 = fig.add_subplot(1, n_panels, 3, projection='3d')
        kl_alphas_u = theta_landscape['kl_alphas_uniform']
        kl_betas_u = theta_landscape['kl_betas_uniform']
        J_kl = theta_landscape['J_kl_grid']

        A_kl, B_kl = np.meshgrid(kl_alphas_u, kl_betas_u)
        colors_kl = cm.viridis(norm_j(np.clip(J_kl, vmin_j, vmax_j)))
        colors_kl[..., 3] = 0.6
        ax2.plot_surface(A_kl, B_kl, J_kl, facecolors=colors_kl, rstride=1, cstride=1,
                         linewidth=0, antialiased=True, shade=False)

        traj_af = np.asarray(theta_landscape['traj_alpha_kl'])
        traj_bf = np.asarray(theta_landscape['traj_beta_kl'])
        interp_kl = RegularGridInterpolator(
            (kl_betas_u, kl_alphas_u), J_kl, method='linear', bounds_error=False,
            fill_value=None)
        traj_z_kl = interp_kl(np.column_stack([traj_bf, traj_af])) + z_lift
        ax2.plot3D(traj_af, traj_bf, traj_z_kl, 'r-', linewidth=2.5, alpha=1.0)
        ax2.scatter([traj_af[0]], [traj_bf[0]], [traj_z_kl[0]],
                    color='blue', s=60, zorder=5, depthshade=False)
        ax2.scatter([traj_af[-1]], [traj_bf[-1]], [traj_z_kl[-1]],
                    color='red', s=70, marker='^', zorder=5, depthshade=False)

        ax2.set_xlabel('$\\sqrt{\\mathrm{KL}}$ along PC1')
        ax2.set_ylabel('$\\sqrt{\\mathrm{KL}}$ along PC2')
        ax2.set_zlabel('$J(\\theta)$')
        ax2.set_title('$J(\\theta)$ in Fisher Information metric', fontsize=12)
        ax2.view_init(elev=35, azim=225)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
# ======== 图2：优化轨迹动画 ========

def plot_trajectory_animation(
    cmaes_history, ppo_history,
    save_path='fig2_trajectory.gif',
    n_frames=80,
    ppo_batch_size=512,
):
    """左边 CMA-ES 轨迹在 r(x) 上，右边 RL 分布演化。
    两边按 function evaluations 同步推进。
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax in axes:
        plot_rastrigin_landscape(ax)

    axes[0].set_title('CMA-ES: Direct Search Trajectory')
    axes[1].set_title('RL (PPO): Distribution Evolution')

    cma_traj = np.array(cmaes_history['trajectory'])
    cma_samples = cmaes_history['all_samples']
    cma_evals = np.array(cmaes_history['n_evals_list'])

    ppo_samples = ppo_history['sample_snapshots']
    ppo_snap_evals = np.array([it * ppo_batch_size + 1 for it, _ in ppo_samples])

    max_evals = max(cma_evals[-1], ppo_snap_evals[-1]) if len(ppo_snap_evals) > 0 else cma_evals[-1]
    eval_timeline = np.logspace(0, np.log10(max_evals), n_frames)

    cma_scatter = axes[0].scatter([], [], c='red', s=10, alpha=0.3, zorder=3)
    cma_best, = axes[0].plot([], [], 'r*', markersize=12, zorder=4)
    cma_trail, = axes[0].plot([], [], 'r-', linewidth=1, alpha=0.5, zorder=2)

    ppo_scatter = axes[1].scatter([], [], c='red', s=10, alpha=0.3, zorder=3)

    frame_text_0 = axes[0].text(0.02, 0.98, '', transform=axes[0].transAxes,
                                 va='top', fontsize=10, color='white',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    frame_text_1 = axes[1].text(0.02, 0.98, '', transform=axes[1].transAxes,
                                 va='top', fontsize=10, color='white',
                                 bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def animate(frame):
        current_eval = eval_timeline[frame]

        cma_idx = np.searchsorted(cma_evals, current_eval, side='right') - 1
        cma_idx = max(0, min(cma_idx, len(cma_samples) - 1))
        samples = cma_samples[cma_idx]
        cma_scatter.set_offsets(samples[:, :2])
        best = cma_traj[cma_idx]
        cma_best.set_data([best[0]], [best[1]])
        trail = cma_traj[:cma_idx + 1]
        cma_trail.set_data(trail[:, 0], trail[:, 1])
        frame_text_0.set_text(f'Evals: {int(cma_evals[cma_idx])}\n'
                               f'Best r = {cmaes_history["best_rewards"][cma_idx]:.2f}')

        ppo_idx = np.searchsorted(ppo_snap_evals, current_eval, side='right') - 1
        ppo_idx = max(0, min(ppo_idx, len(ppo_samples) - 1))
        it, samp = ppo_samples[ppo_idx]
        ppo_scatter.set_offsets(samp[:, :2])
        r_idx = min(it, len(ppo_history["mean_reward"]) - 1)
        frame_text_1.set_text(f'Evals: {int(ppo_snap_evals[ppo_idx])}\n'
                               f'Mean r = {ppo_history["mean_reward"][r_idx]:.2f}')

        return cma_scatter, cma_best, cma_trail, ppo_scatter, frame_text_0, frame_text_1

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200, blit=True)
    anim.save(save_path, writer='pillow', fps=5)
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图3：收敛曲线 ========

def plot_convergence(cmaes_history, ppo_history, batch_size=256,
                     save_path='fig3_convergence.png'):
    """收敛曲线对比：实线 = mean（分布质量），虚线 = cumulative best。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- CMA-ES ---
    cma_evals = [1] + cmaes_history['n_evals_list']
    cma_mean = [cmaes_history['mean_rewards'][0]] + cmaes_history['mean_rewards']
    cma_best = [cmaes_history['best_rewards'][0]] + cmaes_history['best_rewards']
    ax.plot(cma_evals, cma_mean, 'b-', linewidth=2, label='CMA-ES mean $r$')
    ax.plot(cma_evals, cma_best, 'b--', linewidth=1.5, alpha=0.5,
            label='CMA-ES best $r$ (cumulative)')

    # --- PPO ---
    ppo_evals = [i * batch_size + 1 for i in range(len(ppo_history['mean_reward']))]
    ax.plot(ppo_evals, ppo_history['mean_reward'], 'r-', linewidth=2,
            label='PPO mean $r$ ($J(\\theta)$)')
    ppo_best = np.maximum.accumulate(ppo_history['best_reward'])
    ax.plot(ppo_evals, ppo_best, 'r--', linewidth=1.5, alpha=0.5,
            label='PPO best $r$ (cumulative)')

    ax.set_xlabel('Function Evaluations')
    ax.set_ylabel('Reward $r(x)$')
    ax.set_title('Convergence Comparison')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图5：预训练数据集可视化 ========

def plot_dataset(dataset_points, save_path='fig5_dataset.png', title='Pretraining Dataset'):
    """将预训练数据集散点叠加在 r(x) 热力图上。"""
    pts = dataset_points.numpy() if hasattr(dataset_points, 'numpy') else np.array(dataset_points)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = plot_rastrigin_landscape(ax)
    ax.scatter(pts[:, 0], pts[:, 1], c='white', s=6, alpha=0.4, zorder=3,
               label=f'n={len(pts)}')
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='$r(x)$')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图6：学到的概率分布对比 ========

def plot_distributions(model, pretrain_params, ppo_params,
                       save_path='fig6_distributions.png'):
    """三面板：奖励函数 | 预训练分布 log π_pretrain | PPO后分布 log π_ppo。"""
    n_grid = 200
    x1 = np.linspace(-5, 5, n_grid)
    x2 = np.linspace(-5, 5, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid = torch.tensor(
        np.stack([X1, X2], axis=-1).reshape(-1, 2), dtype=torch.float32)

    def compute_log_prob(params):
        model.set_flat_params(params)
        model.eval()
        with torch.no_grad():
            log_p = model.log_prob(grid).clamp(-20, 5)
        return log_p.reshape(n_grid, n_grid).numpy()

    logp_pre = compute_log_prob(pretrain_params)
    logp_ppo = compute_log_prob(ppo_params)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: reward landscape
    im0 = plot_rastrigin_landscape(axes[0])
    axes[0].set_title('Reward $r(x)$\n(target landscape)', fontsize=13)
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im0, cax=cax0, label='$r(x)$')

    # 固定 colorbar 范围
    vmin = -10
    vmax = 3

    # Panel 2: pretrained log distribution
    im1 = axes[1].contourf(X1, X2, logp_pre, levels=50, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[1].set_title('Pretrained $\\log\\,\\pi_\\theta(x)$\n(before RL)', fontsize=13)
    axes[1].set_xlabel('$x_1$'); axes[1].set_ylabel('$x_2$')
    axes[1].set_aspect('equal')
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im1, cax=cax1, label='$\\log\\,\\pi(x)$')

    # Panel 3: PPO log distribution
    im2 = axes[2].contourf(X1, X2, logp_ppo, levels=50, cmap='turbo', vmin=vmin, vmax=vmax)
    axes[2].set_title('After PPO $\\log\\,\\pi_\\theta(x)$\n(after RL)', fontsize=13)
    axes[2].set_xlabel('$x_1$'); axes[2].set_ylabel('$x_2$')
    axes[2].set_aspect('equal')
    divider2 = make_axes_locatable(axes[2])
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图7：log π(x) 随 PPO 训练的演化动画 ========

def plot_distribution_evolution(model, param_snapshots,
                                save_path='fig7_distribution_evolution.gif',
                                n_grid=80, fps=10):
    """动画：log π_θ(x) 随 PPO 训练步数的演化。

    每帧对应一个 param_snapshot，按训练顺序播放。
    """
    x1 = np.linspace(-5, 5, n_grid)
    x2 = np.linspace(-5, 5, n_grid)
    X1, X2 = np.meshgrid(x1, x2)
    grid = torch.tensor(
        np.stack([X1, X2], axis=-1).reshape(-1, 2), dtype=torch.float32)

    print(f"    Computing log π for {len(param_snapshots)} snapshots...")
    frames = []
    for it, params in param_snapshots:
        model.set_flat_params(params)
        model.eval()
        with torch.no_grad():
            log_p = model.log_prob(grid).clamp(-20, 5)
        frames.append((it, log_p.reshape(n_grid, n_grid).numpy()))

    vmin = -10
    vmax = 3

    fig, ax = plt.subplots(figsize=(6, 5.5))
    plt.tight_layout(pad=2)

    im = ax.imshow(frames[0][1], extent=[-5, 5, -5, 5], origin='lower',
                   cmap='turbo', vmin=vmin, vmax=vmax, aspect='equal')
    title = ax.set_title('PPO iter 0  |  log π(x)', fontsize=12)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    plt.colorbar(im, ax=ax, label='$\\log\\,\\pi(x)$')

    def update(i):
        it, logp = frames[i]
        im.set_data(logp)
        title.set_text(f'PPO iter {it}  |  log π(x)')
        return [im, title]

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True)
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图8：FIM 沿轨迹的演化 ========

def plot_fim_evolution(traj_fim, ppo_history, save_path='fig8_fim_evolution.png'):
    """F11(t), F22(t) + reward 曲线。"""
    iters = np.arange(len(traj_fim['F11']))

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.semilogy(iters, traj_fim['F11'], 'b-', linewidth=1.5, label='$F_{11}$ (along $d_1$)')
    ax1.semilogy(iters, traj_fim['F22'], 'r-', linewidth=1.5, label='$F_{22}$ (along $d_2$)')
    ax1.set_xlabel('PPO Iteration')
    ax1.set_ylabel('FIM projection (log scale)', color='black')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    mean_r = ppo_history['mean_reward']
    ax2.plot(np.arange(len(mean_r)), mean_r, 'g--', linewidth=1.5, alpha=0.7, label='Mean reward')
    ax2.set_ylabel('Mean reward', color='green')
    ax2.legend(loc='upper right', fontsize=10)

    ax1.set_title('FIM evolution along PPO trajectory', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== Pretrain Loss 曲线 ========

def plot_pretrain_loss(pretrain_history, save_path='fig_pretrain_loss.png',
                       title='Pretrain Loss'):
    """画 train_loss vs val_loss 对比曲线。

    pretrain_history: dict with 'train_loss', 'val_loss', 'val_epochs'
    """
    train_loss = np.array(pretrain_history['train_loss'])
    val_loss = np.array(pretrain_history['val_loss'])
    val_epochs = np.array(pretrain_history['val_epochs'])

    epochs = np.arange(len(train_loss))

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(epochs, train_loss, 'b-', linewidth=0.6, alpha=0.4, label='Train loss (raw)')

    if len(train_loss) > 20:
        window = min(20, len(train_loss) // 5)
        smoothed = np.convolve(train_loss, np.ones(window) / window, mode='valid')
        ax.plot(epochs[window - 1:], smoothed, 'b-', linewidth=2,
                label=f'Train loss (avg w={window})')

    ax.plot(val_epochs, val_loss, 'ro-', linewidth=2, markersize=3,
            label='Val loss')

    best_idx = np.argmin(val_loss)
    ax.axvline(x=val_epochs[best_idx], color='green', linestyle='--', alpha=0.5)
    ax.annotate(f'Best val: {val_loss[best_idx]:.4f}\n(epoch {val_epochs[best_idx]})',
                xy=(val_epochs[best_idx], val_loss[best_idx]),
                xytext=(20, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=9, color='green')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    y_lower = min(train_loss.min(), val_loss.min()) - 0.05
    ax.set_ylim(y_lower, 1.2 * train_loss[0])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图9：欧氏步长 vs KL 步长 ========

def plot_step_comparison(traj_fim, save_path='fig9_step_comparison.png'):
    """s_Euclid, s_KL, ρ 随 iteration 变化。"""
    T = len(traj_fim['s_kl'])
    iters = np.arange(T)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(iters, traj_fim['s_euclid'], 'b-', linewidth=1.2, label='$s_{\\mathrm{Euclid}}$')
    ax1.plot(iters, traj_fim['s_kl'], 'r-', linewidth=1.2, label='$s_{\\mathrm{KL}}$')
    ax1.set_xlabel('PPO Iteration')
    ax1.set_ylabel('Step size')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    rho = traj_fim['rho']
    ax2.semilogy(iters, np.maximum(rho, 1e-6), 'g-', linewidth=1, alpha=0.6,
                 label='$\\rho = s_{KL}/s_{Euclid}$')
    ax2.set_ylabel('$\\rho$ (log scale)', color='green')
    ax2.legend(loc='upper right', fontsize=10)

    ax1.set_title('Euclidean vs KL step size along trajectory', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图10：累积 KL 距离 vs Reward ========

def plot_kl_efficiency(traj_fim, ppo_history, save_path='fig10_kl_efficiency.png'):
    """x 轴 = 累积 KL 弧长，y 轴 = J(θ)。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    cum_kl = traj_fim['cumulative_kl']
    mean_r = ppo_history['mean_reward']
    n = min(len(cum_kl), len(mean_r))

    ax.plot(cum_kl[:n], mean_r[:n], 'r-', linewidth=2, label='PPO')
    ax.plot(cum_kl[0], mean_r[0], 'o', color='blue', markersize=10, zorder=5, label='Start')
    ax.plot(cum_kl[n-1], mean_r[n-1], '^', color='red', markersize=12, zorder=5, label='End')

    ax.set_xlabel('Cumulative KL arc length $S_{\\mathrm{KL}}$')
    ax.set_ylabel('Mean reward $J(\\theta)$')
    ax.set_title('Learning efficiency: reward vs KL distance traveled', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图11：轨迹 + FIM 椭圆 ========

def plot_trajectory_ellipses(theta_landscape, traj_fim,
                             save_path='fig11_trajectory_ellipses.png',
                             every_n=10):
    """在 PCA 平面上画轨迹 + 每隔 N 步的 FIM 椭圆。"""
    from matplotlib.patches import Ellipse

    alphas = theta_landscape['alphas']
    betas = theta_landscape['betas']
    J_grid = theta_landscape['J_grid']
    traj_a = theta_landscape['traj_alpha']
    traj_b = theta_landscape['traj_beta']

    fig, ax = plt.subplots(figsize=(8, 7))

    # J(θ) 热力图
    A_e, B_e = np.meshgrid(alphas, betas)
    levels = np.linspace(J_grid.min(), J_grid.max(), 50)
    im = ax.contourf(A_e, B_e, J_grid, levels=levels, cmap='viridis', extend='both')

    # 轨迹
    ax.plot(traj_a, traj_b, 'r-', linewidth=1.2, alpha=0.6, zorder=4)
    ax.plot(traj_a[0], traj_b[0], 'o', color='blue', markersize=9, zorder=6, label='Start')
    ax.plot(traj_a[-1], traj_b[-1], '^', color='red', markersize=11, zorder=6, label='End')

    # FIM 椭圆
    F11 = traj_fim['F11']
    F22 = traj_fim['F22']
    n_pts = min(len(traj_a), len(F11))

    # 椭圆大小归一化：用中位数 FIM 作为参考
    f_ref = max(np.median(np.concatenate([F11, F22])), 1e-6)
    base_size = (alphas[-1] - alphas[0]) * 0.03

    for t in range(0, n_pts, every_n):
        w = base_size * np.sqrt(max(F11[t], 1e-6) / f_ref)
        h = base_size * np.sqrt(max(F22[t], 1e-6) / f_ref)
        w = np.clip(w, base_size * 0.2, base_size * 5)
        h = np.clip(h, base_size * 0.2, base_size * 5)

        progress = t / max(n_pts - 1, 1)
        color = plt.cm.coolwarm(progress)

        ellipse = Ellipse(
            (traj_a[t], traj_b[t]), width=w, height=h,
            fill=False, edgecolor=color, linewidth=1.5, alpha=0.8, zorder=5)
        ax.add_patch(ellipse)

    ax.set_xlabel('$d_1$ (start→end)')
    ax.set_ylabel('$d_2$ (max ⊥ variance)')
    ax.set_title('$J(\\theta)$ landscape + local FIM ellipses\n(size ∝ √F, color = training progress)',
                 fontsize=12)
    ax.legend(loc='lower right', fontsize=9)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='$J(\\theta)$')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# ======== 图12：Theorem 1 验证 ========

def plot_theorem1_check(traj_fim, ppo_history, save_path='fig12_theorem1.png'):
    """验证 Theorem 1: C_v ≤ Var(r)，使用配对样本估计。

    C_v = E[r·(g·v)]² / E[(g·v)²]  是 ||∇̃J||²_F 沿轨迹方向的下界。
    上面板：√C_v vs √Var(r) vs R_max^eff
    下面板：C_v / Var(r) = Corr(r, g·v)²，应 ≤ 1
    """
    C_v = traj_fim['C_v']
    paired_var = traj_fim['paired_reward_var']
    rmax_eff = np.array(ppo_history['reward_rmax_eff'])

    T = len(C_v)
    n = min(T, len(paired_var) - 1, len(rmax_eff) - 1)

    sqrt_Cv = np.sqrt(np.maximum(C_v[:n], 1e-10))
    sqrt_var = np.sqrt(np.maximum(paired_var[:n], 1e-10))
    rmax = rmax_eff[:n]

    iters = np.arange(n)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.08})

    # --- 上面板 ---
    ax1.semilogy(iters, rmax, '-', color='orange', linewidth=1.5, alpha=0.7,
                 label='$R_{\\max}^{\\mathrm{eff}}(\\pi_{\\theta_t})$')
    ax1.semilogy(iters, sqrt_var, 'r-', linewidth=2,
                 label='$\\sqrt{\\mathrm{Var}_{\\pi_{\\theta}}(r)}$  (Thm 1 bound)')
    ax1.semilogy(iters, sqrt_Cv, 'b-', linewidth=2,
                 label='$\\sqrt{C_v}$  (paired, lower bound on $\\|\\tilde{\\nabla}J\\|_F$)')

    ax1.set_ylabel('Value (log scale)')
    ax1.set_title('Theorem 1:  $C_v = \\frac{\\mathrm{Cov}(r,\\, g{\\cdot}v)^2}{\\mathrm{Var}(g{\\cdot}v)} '
                   '\\leq \\|\\tilde{\\nabla}J\\|_F^2 \\leq \\mathrm{Var}(r) \\leq (R_{\\max}^{\\mathrm{eff}})^2$',
                   fontsize=11)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- 下面板：tightness ---
    ratio = C_v[:n] / np.maximum(paired_var[:n], 1e-10)
    ax2.plot(iters, ratio, 'b-', linewidth=1.5)
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5,
                label='Bound ($\\leq 1$)')
    ax2.set_xlabel('PPO Iteration')
    ax2.set_ylabel('$C_v \\;/\\; \\mathrm{Var}(r) = \\mathrm{Corr}(r,\\, g{\\cdot}v)^2$')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, min(max(ratio.max() * 1.2, 0.5), 1.5))

    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
