"""分析 PCA 方向的稀疏性：d1, d2 的系数分布 + 按层分组。"""

import torch
import numpy as np
from sklearn.decomposition import PCA
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, 'exp_2d'))

from model import FlowModel
from boltzmann import pretrain_boltzmann
from ppo import ppo_train

torch.manual_seed(42)
np.random.seed(42)

# --- 复现训练，拿到轨迹 ---
print("=== Building model + training ===")
model = FlowModel(n_layers=8, hidden_dim=64, coupling='spline')
pretrain_boltzmann(model, n_epochs=300, batch_size=512, lr=1e-3, beta=0.0, dataset_size=100000)

ppo_history = ppo_train(model, n_iters=200, batch_size=256, ppo_epochs=2,
                        lr=3e-4, kl_coeff=0.0, save_every=1)

# --- PCA ---
print("\n=== PCA Analysis ===")
traj_params = torch.stack([fp for _, fp in ppo_history['param_snapshots']])
T, D = traj_params.shape
print(f"Trajectory: {T} snapshots, {D} parameters")

pca = PCA(n_components=2)
pca.fit(traj_params.numpy())
d1 = pca.components_[0]  # (D,)
d2 = pca.components_[1]  # (D,)

print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, "
      f"PC2={pca.explained_variance_ratio_[1]:.4f}")

# --- d1, d2 系数分布 ---
abs_d1 = np.abs(d1)
abs_d2 = np.abs(d2)

print(f"\n=== d1 coefficient distribution ===")
print(f"  max |d1[i]|  = {abs_d1.max():.6f}  (index {abs_d1.argmax()})")
print(f"  top-1 explains {abs_d1.max()**2 * 100:.4f}% of d1 variance")
top10 = np.sort(abs_d1)[-10:][::-1]
top10_var = np.sum(top10**2) * 100
print(f"  top-10 explains {top10_var:.2f}%")
top100 = np.sort(abs_d1)[-100:][::-1]
top100_var = np.sum(top100**2) * 100
print(f"  top-100 explains {top100_var:.2f}%")
top1000 = np.sort(abs_d1)[-1000:][::-1]
top1000_var = np.sum(top1000**2) * 100
print(f"  top-1000 explains {top1000_var:.2f}%")

print(f"\n=== d2 coefficient distribution ===")
print(f"  max |d2[i]|  = {abs_d2.max():.6f}  (index {abs_d2.argmax()})")
print(f"  top-1 explains {abs_d2.max()**2 * 100:.4f}% of d2 variance")
top10_d2 = np.sort(abs_d2)[-10:][::-1]
print(f"  top-10 explains {np.sum(top10_d2**2) * 100:.2f}%")
top100_d2 = np.sort(abs_d2)[-100:][::-1]
print(f"  top-100 explains {np.sum(top100_d2**2) * 100:.2f}%")

# --- 按层分组 ---
print(f"\n=== Per-layer contribution to d1, d2 ===")
print(f"{'Layer':<45s} {'#params':>8s} {'||d1||_layer':>12s} {'%d1':>8s} {'||d2||_layer':>12s} {'%d2':>8s}")
print("-" * 95)

offset = 0
layer_info = []
for name, p in model.named_parameters():
    n = p.numel()
    chunk_d1 = d1[offset:offset+n]
    chunk_d2 = d2[offset:offset+n]
    norm_d1 = np.linalg.norm(chunk_d1)
    norm_d2 = np.linalg.norm(chunk_d2)
    frac_d1 = norm_d1**2 * 100  # d1 is unit vector, so sum of squares = 1
    frac_d2 = norm_d2**2 * 100
    layer_info.append((name, n, norm_d1, frac_d1, norm_d2, frac_d2))
    print(f"{name:<45s} {n:>8d} {norm_d1:>12.6f} {frac_d1:>7.2f}% {norm_d2:>12.6f} {frac_d2:>7.2f}%")
    offset += n

# --- 按耦合层汇总 ---
print(f"\n=== Per coupling layer (aggregated) ===")
for layer_idx in range(8):
    prefix = f"layers.{layer_idx}."
    total_d1_sq = sum(fi[3] for fi in layer_info if fi[0].startswith(prefix))
    total_d2_sq = sum(fi[5] for fi in layer_info if fi[0].startswith(prefix))
    n_params = sum(fi[1] for fi in layer_info if fi[0].startswith(prefix))
    print(f"  Coupling layer {layer_idx}: {n_params:>6d} params, "
          f"d1={total_d1_sq:>6.2f}%, d2={total_d2_sq:>6.2f}%")

# --- 衰减曲线数据 ---
print(f"\n=== Sorted |d1| decay (first 30) ===")
sorted_d1 = np.sort(abs_d1)[::-1]
cumvar = np.cumsum(sorted_d1**2) * 100
for i in [0, 1, 2, 4, 9, 19, 29, 49, 99, 199, 499, 999]:
    if i < len(sorted_d1):
        print(f"  rank {i+1:>5d}: |d1|={sorted_d1[i]:.6f}, cumulative={cumvar[i]:.2f}%")
