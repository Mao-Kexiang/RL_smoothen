"""2D 离散扩散流实验：Boltzmann 预训练 + PPO + CMA-ES + 可视化。

用法:
  python exp_diffusion/run_diffusion.py                   # 默认 beta=1.0
  python exp_diffusion/run_diffusion.py --beta 0.5        # 指定 beta
  python exp_diffusion/run_diffusion.py --scan             # 启用 FIM θ 空间扫描（慢）
  python exp_diffusion/run_diffusion.py --n_steps 20       # 20 步 ODE
"""

import os
import sys
import argparse
import torch
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'exp_2d'))

from model_diffusion import DiffusionFlow, pretrain_flow_matching, pretrain_diffusion
from ppo_diffusion import ppo_train_diffusion
from direct_optim import cmaes_optimize
from boltzmann import sample_boltzmann
from fim import compute_theta_landscape
from visualize import (
    plot_landscape_comparison,
    plot_landscape_3d,
    plot_trajectory_animation,
    plot_convergence,
    plot_dataset,
    plot_distributions,
    plot_distribution_evolution,
    plot_pretrain_loss,
)


def main():
    parser = argparse.ArgumentParser(description='2D Diffusion Flow Experiment')
    parser.add_argument('--scan', action='store_true',
                        help='Enable FIM theta-space scan (slow)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Boltzmann inverse temperature for pretraining')
    parser.add_argument('--n_steps', type=int, default=10,
                        help='Number of ODE discretization steps')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_hidden', type=int, default=3)
    parser.add_argument('--n_fixed_point', type=int, default=15)
    parser.add_argument('--pretrain_epochs', type=int, default=300)
    parser.add_argument('--pretrain_lr', type=float, default=1e-3)
    parser.add_argument('--pretrain_batch', type=int, default=512)
    parser.add_argument('--pretrain_method', type=str, default='fm',
                        choices=['fm', 'nll'],
                        help='Pretrain method: fm=flow matching, nll=MLE')
    parser.add_argument('--dataset_size', type=int, default=0,
                        help='Fixed dataset size for FM pretrain (0=resample each epoch)')
    parser.add_argument('--ppo_iters', type=int, default=200)
    parser.add_argument('--ppo_epochs', type=int, default=2)
    parser.add_argument('--ppo_lr', type=float, default=3e-4)
    parser.add_argument('--ppo_kl', type=float, default=0.5)
    parser.add_argument('--ppo_batch', type=int, default=256)
    parser.add_argument('--diverge_thresh', type=float, default=5.0)
    args = parser.parse_args()

    output_dir = os.path.join(ROOT, 'output', 'diffusion')
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # ====== Step 1: 构建并预训练扩散流模型 ======
    print("=" * 60)
    print("Step 1: Building DiffusionFlow + Boltzmann Pretraining...")
    print("=" * 60)
    model = DiffusionFlow(
        n_steps=args.n_steps,
        hidden_dim=args.hidden_dim,
        n_hidden=args.n_hidden,
        n_fixed_point=args.n_fixed_point,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: T={args.n_steps}, hidden={args.hidden_dim}, "
          f"n_hidden={args.n_hidden}, params={n_params:,}")
    if args.pretrain_method == 'fm':
        pretrain_history = pretrain_flow_matching(model, n_epochs=args.pretrain_epochs,
                               batch_size=args.pretrain_batch,
                               lr=args.pretrain_lr, beta=args.beta,
                               device=device, dataset_size=args.dataset_size)
    else:
        pretrain_history = pretrain_diffusion(model, n_epochs=args.pretrain_epochs,
                           batch_size=args.pretrain_batch,
                           lr=args.pretrain_lr, beta=args.beta, device=device)
    pretrain_params = model.get_flat_params().clone().cpu()

    # ====== Step 2: PPO 训练 ======
    print("\n" + "=" * 60)
    print("Step 2: PPO Training (RL fine-tuning)...")
    print("=" * 60)
    ppo_history = ppo_train_diffusion(
        model,
        n_iters=args.ppo_iters,
        batch_size=args.ppo_batch,
        ppo_epochs=args.ppo_epochs,
        clip_eps=0.2,
        lr=args.ppo_lr,
        kl_coeff=args.ppo_kl,
        save_every=1,
        diverge_threshold=args.diverge_thresh,
        device=device,
    )

    # ====== Step 3: CMA-ES 直接优化 ======
    print("\n" + "=" * 60)
    print("Step 3: CMA-ES Direct Optimization...")
    print("=" * 60)
    cmaes_history = cmaes_optimize(n_evals=51200, sigma0=3.0, seed=42)

    # ====== Step 4: 迁移回 CPU 用于 FIM 扫描和可视化 ======
    model.cpu()
    theta_landscape = None
    if args.scan:
        print("\n" + "=" * 60)
        print("Step 4: Scanning J(theta) landscape (FIM)...")
        print("=" * 60)
        theta_landscape = compute_theta_landscape(
            model,
            ppo_history['param_snapshots'],
            n_grid=21,
            grid_range=3.0,
            n_eval_samples=500,
            n_fim_samples=20,
        )
    else:
        print("\n  [Step 4 skipped] Use --scan to enable FIM landscape scanning.")

    # ====== Step 5: 可视化 ======
    print("\n" + "=" * 60)
    print("Step 5: Generating visualizations...")
    print("=" * 60)

    print("\n  [Fig 0] Pretrain loss curves...")
    plot_pretrain_loss(
        pretrain_history,
        save_path=os.path.join(output_dir, 'fig0_pretrain_loss.png'),
        title=f'Pretrain Loss (Boltzmann $\\beta$={args.beta}, {args.pretrain_method})',
    )

    if theta_landscape is not None:
        print("\n  [Fig 1] Landscape comparison...")
        plot_landscape_comparison(
            theta_landscape,
            save_path=os.path.join(output_dir, 'fig1_landscape.png'),
        )

    print("\n  [Fig 2] Trajectory animation...")
    plot_trajectory_animation(
        cmaes_history, ppo_history,
        save_path=os.path.join(output_dir, 'fig2_trajectory.gif'),
        ppo_batch_size=args.ppo_batch,
    )

    print("\n  [Fig 3] Convergence curves...")
    plot_convergence(
        cmaes_history, ppo_history,
        batch_size=args.ppo_batch,
        save_path=os.path.join(output_dir, 'fig3_convergence.png'),
    )

    if theta_landscape is not None:
        print("\n  [Fig 4] 3D landscape surfaces...")
        plot_landscape_3d(
            theta_landscape,
            save_path=os.path.join(output_dir, 'fig4_landscape_3d.png'),
        )

    print("\n  [Fig 5] Pretraining dataset...")
    dataset_pts = sample_boltzmann(2048, beta=args.beta)
    plot_dataset(
        dataset_pts,
        save_path=os.path.join(output_dir, 'fig5_dataset.png'),
        title=f'Pretraining Dataset (Boltzmann $\\beta$={args.beta})',
    )

    print("\n  [Fig 6] Learned distributions (pretrain vs PPO)...")
    ppo_params = model.get_flat_params().clone()
    plot_distributions(
        model, pretrain_params, ppo_params,
        save_path=os.path.join(output_dir, 'fig6_distributions.png'),
    )

    print("\n  [Fig 7] Distribution evolution animation...")
    plot_distribution_evolution(
        model, ppo_history['param_snapshots'],
        save_path=os.path.join(output_dir, 'fig7_distribution_evolution.gif'),
    )

    # ====== 汇总 ======
    print("\n" + "=" * 60)
    final_mean = ppo_history['mean_reward'][-1]
    ppo_best = max(ppo_history['best_reward'])
    cma_best = cmaes_history['final_reward']

    ppo_first_idx = next(i for i, r in enumerate(ppo_history['best_reward'])
                         if r >= ppo_best)
    ppo_first_evals = ppo_first_idx * args.ppo_batch

    cma_best_rewards = cmaes_history['best_rewards']
    cma_first_idx = next(i for i, r in enumerate(cma_best_rewards)
                         if r >= cma_best)
    cma_first_evals = cmaes_history['n_evals_list'][cma_first_idx]

    print(f"Results (2D Rastrigin, DiffusionFlow T={args.n_steps}):")
    print(f"  PPO final mean reward:   {final_mean:.4f}")
    print(f"  PPO best reward:         {ppo_best:.4f}  (first at iter {ppo_first_idx}, evals={ppo_first_evals})")
    print(f"  CMA-ES best reward:      {cma_best:.4f}  (first at evals={cma_first_evals})")
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
