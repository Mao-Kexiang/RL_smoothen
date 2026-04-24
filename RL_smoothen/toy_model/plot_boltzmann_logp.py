"""画 Boltzmann 分布 log p_data(x) = -β·Rastrigin(x) - log Z 的热力图。

Z 通过 [-5,5]² 上的密网格数值积分计算（2D 问题，精度足够）。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from env import rastrigin

plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 150

beta = 1.0
n_grid = 1000
xlim = (-5, 5)

x1 = np.linspace(*xlim, n_grid)
x2 = np.linspace(*xlim, n_grid)
dx = (xlim[1] - xlim[0]) / (n_grid - 1)

X1, X2 = np.meshgrid(x1, x2)
grid = torch.tensor(np.stack([X1, X2], axis=-1), dtype=torch.float32)

R = rastrigin(grid).numpy()
log_unnorm = -beta * R

# log Z = log ∫ exp(-β R(x)) dx ≈ log( Σ exp(-β R(x_i)) · dx² )
# 用 log-sum-exp 数值稳定
log_unnorm_flat = log_unnorm.ravel()
max_val = log_unnorm_flat.max()
log_Z = max_val + np.log(np.sum(np.exp(log_unnorm_flat - max_val))) + 2 * np.log(dx)

log_p = log_unnorm - log_Z
log_p_clamped = np.clip(log_p, -20, 5)

print(f"beta = {beta}")
print(f"Grid: {n_grid}x{n_grid}, dx = {dx:.6f}")
print(f"log Z = {log_Z:.6f},  Z = {np.exp(log_Z):.6f}")
print(f"log p range: [{log_p.min():.4f}, {log_p.max():.4f}]")
print(f"log p clamped range: [{log_p_clamped.min():.4f}, {log_p_clamped.max():.4f}]")
print(f"Integral check: ∫ p(x) dx ≈ {np.sum(np.exp(log_p)) * dx**2:.6f} (should ≈ 1)")

# --- 画图（与 fig6 一致的 scale：clamp(-20,5), cmap='turbo'） ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel 1: Rastrigin landscape (reward = -Rastrigin) — 同 fig6 panel 1
from exp_2d.visualize import plot_rastrigin_landscape
im0 = plot_rastrigin_landscape(axes[0])
axes[0].set_title('Reward $r(x)$\n(target landscape)', fontsize=13)
divider0 = make_axes_locatable(axes[0])
cax0 = divider0.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im0, cax=cax0, label='$r(x)$')

# Panel 2: log p_data(x) — 同 fig6 的 clamp(-20,5) + turbo
ax = axes[1]
vmin, vmax = -20, 5
im1 = ax.contourf(X1, X2, log_p_clamped, levels=50, cmap='turbo', vmin=vmin, vmax=vmax)
ax.set_title(f'$\\log\\, p_{{\\mathrm{{data}}}}(x)$\n(Boltzmann $\\beta={beta}$)', fontsize=13)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im1, cax=cax, label='$\\log\\, p(x)$')

# Panel 3: p_data(x) heatmap (linear scale)
ax = axes[2]
p_data = np.exp(log_p)
im2 = ax.contourf(X1, X2, p_data, levels=50, cmap='hot_r')
ax.set_title(f'$p_{{\\mathrm{{data}}}}(x)$\n(Boltzmann $\\beta={beta}$)', fontsize=13)
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_aspect('equal')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.1)
plt.colorbar(im2, cax=cax, label='$p(x)$')

plt.suptitle(f'Boltzmann distribution:  $p_{{\\mathrm{{data}}}}(x) = '
             f'\\exp(-\\beta \\cdot \\mathrm{{Rastrigin}}(x)) \\;/\\; Z$'
             f'    ($\\beta={beta},\\; \\log Z={log_Z:.4f}$)',
             fontsize=14, y=1.02)

plt.tight_layout()
out_dir = 'output/flow'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'fig_boltzmann_logp_beta{beta}.png')
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"\nSaved: {out_path}")
