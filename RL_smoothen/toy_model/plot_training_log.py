"""解析 rl_flow.out 日志，画 Pretrain NLL + PPO mean_r 训练曲线。"""

import re
import numpy as np
import matplotlib.pyplot as plt

log_path = '../script/rl_flow.out'

with open(log_path, 'r') as f:
    lines = f.readlines()

# --- 解析 Pretrain ---
pretrain_epochs = []
pretrain_nll = []
for line in lines:
    m = re.search(r'Pretrain.*epoch (\d+)/\d+, NLL=([\d.+-]+)', line)
    if m:
        pretrain_epochs.append(int(m.group(1)))
        pretrain_nll.append(float(m.group(2)))

# --- 解析 PPO ---
ppo_iters = []
ppo_mean_r = []
ppo_kl = []
for line in lines:
    m = re.search(r'PPO iter (\d+)/\d+: mean_r=([\d.eE+-]+), best_r=[\d.eE+-]+, kl=([\d.eE+-]+)', line)
    if m:
        ppo_iters.append(int(m.group(1)))
        ppo_mean_r.append(float(m.group(2)))
        ppo_kl.append(float(m.group(3)))

# --- 画图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：Pretrain NLL vs Epoch
ax = axes[0]
ax.plot(pretrain_epochs, pretrain_nll, 'b-', linewidth=0.8, alpha=0.6, label='NLL (raw)')
# 滑动平均
if len(pretrain_nll) > 10:
    window = 20
    nll_arr = np.array(pretrain_nll)
    smoothed = np.convolve(nll_arr, np.ones(window)/window, mode='valid')
    ep_smoothed = pretrain_epochs[window-1:]
    ax.plot(ep_smoothed, smoothed, 'r-', linewidth=2, label=f'NLL (moving avg, w={window})')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Negative Log-Likelihood (NLL)', fontsize=12)
ax.set_title('Pretrain: Boltzmann $\\beta=1.0$, Affine Flow', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 右图：PPO mean_r vs Iteration
ax = axes[1]
color_r = 'tab:red'
ax.plot(ppo_iters, ppo_mean_r, '-', color=color_r, linewidth=1.5, label='Mean reward $J(\\theta)$')
ax.set_xlabel('PPO Iteration', fontsize=12)
ax.set_ylabel('Mean Reward', fontsize=12, color=color_r)
ax.tick_params(axis='y', labelcolor=color_r)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.grid(True, alpha=0.3)

# 右侧 y 轴：KL divergence
ax2 = ax.twinx()
color_kl = 'tab:blue'
ax2.plot(ppo_iters, ppo_kl, '-', color=color_kl, linewidth=1, alpha=0.6, label='$D_{\\mathrm{KL}}(\\pi_\\theta \\| \\pi_{\\mathrm{base}})$')
ax2.set_ylabel('KL Divergence', fontsize=12, color=color_kl)
ax2.tick_params(axis='y', labelcolor=color_kl)

ax.set_title('PPO Fine-tuning: batch=1024, $\\lambda_{\\mathrm{KL}}=0$', fontsize=13)

# 合并图例
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

plt.tight_layout()
out_path = 'output/flow/fig_training_curves.png'
plt.savefig(out_path, bbox_inches='tight', dpi=150)
plt.close()
print(f"Saved: {out_path}")

# --- 打印摘要 ---
print(f"\nPretrain: {len(pretrain_epochs)} points, epoch {pretrain_epochs[0]}–{pretrain_epochs[-1]}")
print(f"  NLL: {pretrain_nll[0]:.4f} → {pretrain_nll[-1]:.4f} (min={min(pretrain_nll):.4f})")
print(f"\nPPO: {len(ppo_iters)} points, iter {ppo_iters[0]}–{ppo_iters[-1]}")
print(f"  mean_r: {ppo_mean_r[0]:.4f} → {ppo_mean_r[-1]:.4f}")
print(f"  KL: {ppo_kl[0]:.4f} → {ppo_kl[-1]:.4f}")
