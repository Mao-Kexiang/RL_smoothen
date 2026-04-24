# RL_smoothen

**Does policy optimization (RL) smooth rugged optimization landscapes? And Why?**

This project provides theoretical analysis and toy-model experiments investigating how lifting a low-dimensional non-convex optimization problem into a higher-dimensional policy parameter space — combined with pretraining and KL constraints — creates a smoother landscape that PPO can navigate effectively. The core framework is built on KL divergence and Fisher Information Geometry.

# Report.md
A theory report which claims the main ideas of this project. Some important theorems are proved in it.

The most important theorem proved in 4.2 section of the file is that:

For any parametric family $\pi_\theta$ with $\mathrm{Var}_ {\pi_\theta}(r) < \infty$:

```math
\|\tilde{\nabla}_\theta J(\theta)\|_F^2 \leq \mathrm{Var}_{\pi_\theta}(r) \leq \max \text{supp}_{\pi_\theta(x)} r^2(x) \leq R_\max ^2
```

**which implies that a good base model $\pi_\theta(x)$ avoid $|r(x)|\to\infty$ is quite important for the validty of RL.**

![fig4](images/fig4_landscape_3d.png)

## Project Structure

```
RL_smoothen/
├── report.md                       # Main theoretical report
├── report_kl_extension.md          # Extended theory on KL regularization
├── report_kl_extension_zh.md       # Chinese version of the KL extension report
├── script/
│   ├── rl_flow.sh                  # SLURM job script for flow model training
│   └── rl_diffusion.sh             # SLURM job script for diffusion model training
└── toy_model/
    ├── env.py                      # 2D Rastrigin reward environment
    ├── fim.py                      # Fisher Information Matrix computation & landscape analysis
    ├── analyze_pca.py              # PCA analysis of parameter trajectories
    ├── plot_boltzmann_logp.py      # Boltzmann log-probability visualization
    ├── plot_training_log.py        # Training curve plotting from logs
    ├── exp_2d/                     # RealNVP normalizing flow experiments
    │   ├── model.py                # Flow model (Affine / Spline coupling layers)
    │   ├── ppo.py                  # PPO training loop with divergence recovery
    │   ├── boltzmann.py            # Boltzmann distribution pretraining (MLE)
    │   ├── gaussian.py             # Gaussian distribution pretraining
    │   ├── direct_optim.py         # CMA-ES baseline (direct x-space optimization)
    │   ├── visualize.py            # Multi-panel landscape & trajectory visualization
    │   └── test_model.py           # Unit tests for model architecture
    ├── exp_diffusion/              # Discrete diffusion flow experiments
    │   ├── run_diffusion.py        # Full diffusion pipeline runner
    │   ├── model_diffusion.py      # Discrete diffusion flow architecture
    │   └── ppo_diffusion.py        # PPO adapted for diffusion models
    ├── output/                     # Generated experiment results (figures, logs)
    └── *.md                        # Technical notes and derivation documents
```

## File Descriptions

### Theoretical Reports

| File | Description |
|---|---|
| `report.md` | Main report. Covers FIM as Riemannian metric, TRPO/PPO trust regions, **Theorem 1** (natural gradient norm bounded by reward variance), and the three-stage pipeline: Pretraining → Effective bounds → RL success. |
| `report_kl_extension.md` | Extends the theory to KL-regularized objectives. Introduces **Theorem 1'** (KL penalty further reduces gradient variance) and **Theorem 2** (strong concavity guarantees exponential convergence). |
| `report_kl_extension_zh.md` | Chinese translation of the KL extension report. |

### Environment & Core Modules

| File | Description |
|---|---|
| `toy_model/env.py` | Defines the 2D Rastrigin function as the optimization target. Domain: [-5, 5]², ~80 local minima, global optimum at origin. Provides `reward()` for PyTorch and `reward_np()` for NumPy. |
| `toy_model/fim.py` | Computes the Fisher Information Matrix on a 2D PCA plane through parameter space. Evaluates J(θ) landscapes, FIM tensors, KL arc lengths, and the C_v statistic for Theorem 1 verification. |

### Flow Model Experiments (`toy_model/exp_2d/`)

| File | Description |
|---|---|
| `model.py` | RealNVP normalizing flow with affine or rational-quadratic-spline coupling layers. 8 layers, 64 hidden units, ~35K parameters. Supports `sample()`, `log_prob()`, and `inverse()`. |
| `boltzmann.py` | Pretrains the flow to match a Boltzmann distribution p(x) ∝ exp(-β·Rastrigin(x)) via MLE. Controls the quality of initialization through inverse temperature β. |
| `gaussian.py` | Simpler pretraining alternative — fits the flow to a Gaussian N(0, σ²I). |
| `ppo.py` | PPO fine-tuning loop. Includes NaN recovery, divergence rollback, log-ratio clamping, and gradient clipping for stability. Tracks rewards, KL divergence, and parameter snapshots. |
| `direct_optim.py` | CMA-ES baseline that optimizes directly in x-space (no neural network). Used for comparison against the RL approach. |
| `visualize.py` | Generates 12+ figures: reward landscapes in original/θ/FIM coordinates, PPO trajectories, convergence curves, 3D surfaces, distribution evolution, and Theorem 1 verification plots. |
| `test_model.py` | Unit tests verifying invertibility, log-probability consistency, and gradient flow of the flow model. |

### Diffusion Model Experiments (`toy_model/exp_diffusion/`)

| File | Description |
|---|---|
| `model_diffusion.py` | Discrete diffusion flow via learned velocity fields. ODE: z_{k+1} = z_k + h·v_θ(z_k, t_k). More numerically stable than affine coupling for extreme parameter exploration. |
| `ppo_diffusion.py` | PPO training adapted for diffusion flow models. |
| `run_diffusion.py` | End-to-end pipeline: build model → pretrain → PPO → CMA-ES baseline → visualization. Accepts command-line args for β, ODE steps, pretraining method, and PPO hyperparameters. |

### Analysis & Visualization Tools

| File | Description |
|---|---|
| `analyze_pca.py` | Analyzes sparsity and per-layer contributions of PCA directions extracted from parameter trajectories. |
| `plot_training_log.py` | Parses training logs to plot pretrain NLL and PPO reward curves with moving averages. |
| `plot_boltzmann_logp.py` | Visualizes the log-probability landscape learned during Boltzmann pretraining. |

### SLURM Scripts (`script/`)

| File | Description |
|---|---|
| `rl_flow.sh` | Submits flow model experiment to SLURM. Requests 1× A100 40GB, 8 CPUs, 32GB RAM, 1-day walltime. |
| `rl_diffusion.sh` | Submits diffusion experiment to SLURM. Requests 1× H200, 8 CPUs, 64GB RAM, 1-day walltime. |

### Technical Notes (`toy_model/*.md`)

| File | Description |
|---|---|
| `technical_report.md` | Implementation details of RealNVP architecture, score function computation, and pretrain+PPO pipeline (in Chinese). |
| `FIM_plotting_report.md` | Documents the FIM visualization implementation and coordinate transformation methodology. |
| `RealNVP_limitations.md` | Analyzes numerical stability issues in affine coupling flows (scale clamping, FIM divergence). |
| `theorem1_verification.md` | Mathematical verification of Theorem 1 and the C_v / Var(r) relationship. |
| `exp_2d/RealNVP_training_explained.md` | Detailed walkthrough of RealNVP training implementation. |
| `exp_2d/test_model_report.md` | Test results for model architecture verification. |

### Output (`toy_model/output/`)

Experiment results organized by configuration:

| Directory | Configuration |
|---|---|
| `flow/` | Default RealNVP (8 layers, β=1.0) |
| `flow_beta_0.0/` | Ablation: no Boltzmann pretraining (uniform init) |
| `flow_beta_1.0/` | β=1.0 Boltzmann pretraining |
| `flow_beta_1.0_layer_8/` | 8 coupling layers |
| `flow_beta_1.0_layer_32/` | 32 coupling layers |
| `flow_lr_1e-3/` | Higher learning rate |
| `flow_sigma_0.5/` | Different noise level |
| `flow_beta_1.0_PPO_batch_256/` | Smaller PPO batch size |
| `diffusion/` | Discrete diffusion model |

Each directory contains PNG visualizations (landscape, trajectory, convergence, etc.), metric logs, and optional GIF animations.

## Dependencies

- PyTorch, NumPy, Matplotlib, SciPy
- `cma` (CMA-ES optimization)
- scikit-learn (PCA)
- CUDA (optional, for GPU acceleration)

---

# RL_smoothen（中文说明）

**为什么策略优化（RL）能平滑崎岖的优化地形？**

本项目从理论和实验两个角度研究：将低维非凸优化问题提升到高维策略参数空间后，结合预训练和 KL 约束，如何使优化地形变得平滑，从而让 PPO 能够有效导航。核心框架建立在 KL 散度和 Fisher 信息几何之上。

## 项目结构

```
RL_smoothen/
├── report.md                       # 主要理论报告
├── report_kl_extension.md          # KL 正则化扩展理论
├── report_kl_extension_zh.md       # KL 扩展报告（中文版）
├── script/
│   ├── rl_flow.sh                  # Flow 模型 SLURM 作业脚本
│   └── rl_diffusion.sh             # Diffusion 模型 SLURM 作业脚本
└── toy_model/
    ├── env.py                      # 2D Rastrigin 奖励环境
    ├── fim.py                      # Fisher 信息矩阵计算与地形分析
    ├── analyze_pca.py              # 参数轨迹 PCA 分析
    ├── plot_boltzmann_logp.py      # Boltzmann 对数概率可视化
    ├── plot_training_log.py        # 训练曲线绘图
    ├── exp_2d/                     # RealNVP 归一化流实验
    │   ├── model.py                # 流模型（仿射/样条耦合层）
    │   ├── ppo.py                  # PPO 训练循环（含发散恢复机制）
    │   ├── boltzmann.py            # Boltzmann 分布预训练（MLE）
    │   ├── gaussian.py             # 高斯分布预训练
    │   ├── direct_optim.py         # CMA-ES 基线（直接 x 空间优化）
    │   ├── visualize.py            # 多面板地形与轨迹可视化
    │   └── test_model.py           # 模型架构单元测试
    ├── exp_diffusion/              # 离散扩散流实验
    │   ├── run_diffusion.py        # 完整扩散流流水线
    │   ├── model_diffusion.py      # 离散扩散流架构
    │   └── ppo_diffusion.py        # 适配扩散模型的 PPO
    ├── output/                     # 实验结果（图表、日志）
    └── *.md                        # 技术笔记与推导文档
```

## 文件说明

### 理论报告

| 文件 | 说明 |
|---|---|
| `report.md` | 主报告。涵盖 FIM 作为黎曼度量、TRPO/PPO 信赖域、**定理 1**（自然梯度范数受奖励方差约束）、以及三阶段流水线：预训练 → 有效界 → RL 成功。 |
| `report_kl_extension.md` | 扩展至 KL 正则化目标。提出**定理 1'**（KL 惩罚进一步降低梯度方差）和**定理 2**（强凹性保证指数收敛）。 |
| `report_kl_extension_zh.md` | KL 扩展报告的中文版。 |

### 环境与核心模块

| 文件 | 说明 |
|---|---|
| `toy_model/env.py` | 定义 2D Rastrigin 函数作为优化目标。定义域：[-5, 5]²，约 80 个局部极小值，全局最优在原点。提供 PyTorch 和 NumPy 两种接口。 |
| `toy_model/fim.py` | 在参数空间的 2D PCA 平面上计算 Fisher 信息矩阵。评估 J(θ) 地形、FIM 张量、KL 弧长和用于验证定理 1 的 C_v 统计量。 |

### 流模型实验（`toy_model/exp_2d/`）

| 文件 | 说明 |
|---|---|
| `model.py` | RealNVP 归一化流，支持仿射耦合层和有理二次样条耦合层。8 层，64 隐藏单元，约 35K 参数。支持 `sample()`、`log_prob()` 和 `inverse()`。 |
| `boltzmann.py` | 通过 MLE 预训练流模型以拟合 Boltzmann 分布 p(x) ∝ exp(-β·Rastrigin(x))。通过逆温度 β 控制初始化质量。 |
| `gaussian.py` | 更简单的预训练替代方案——将流模型拟合到高斯分布 N(0, σ²I)。 |
| `ppo.py` | PPO 微调循环。包含 NaN 恢复、发散回滚、log-ratio 截断和梯度裁剪等稳定性机制。跟踪奖励、KL 散度和参数快照。 |
| `direct_optim.py` | CMA-ES 基线，直接在 x 空间优化（不使用神经网络），用于与 RL 方法对比。 |
| `visualize.py` | 生成 12+ 张图表：原始空间/θ空间/FIM 坐标下的奖励地形、PPO 轨迹、收敛曲线、3D 曲面、分布演化、定理 1 验证图等。 |
| `test_model.py` | 验证流模型可逆性、对数概率一致性和梯度流的单元测试。 |

### 扩散模型实验（`toy_model/exp_diffusion/`）

| 文件 | 说明 |
|---|---|
| `model_diffusion.py` | 基于学习速度场的离散扩散流。ODE：z_{k+1} = z_k + h·v_θ(z_k, t_k)。在极端参数探索时比仿射耦合层更数值稳定。 |
| `ppo_diffusion.py` | 适配扩散流模型的 PPO 训练。 |
| `run_diffusion.py` | 端到端流水线：构建模型 → 预训练 → PPO → CMA-ES 基线 → 可视化。支持通过命令行参数配置 β、ODE 步数、预训练方法和 PPO 超参数。 |

### 分析与可视化工具

| 文件 | 说明 |
|---|---|
| `analyze_pca.py` | 分析参数轨迹 PCA 方向的稀疏性和逐层贡献。 |
| `plot_training_log.py` | 解析训练日志，绘制预训练 NLL 和 PPO 奖励曲线（含移动平均）。 |
| `plot_boltzmann_logp.py` | 可视化 Boltzmann 预训练过程中学到的对数概率地形。 |

### SLURM 脚本（`script/`）

| 文件 | 说明 |
|---|---|
| `rl_flow.sh` | 提交流模型实验到 SLURM。请求 1× A100 40GB、8 CPU、32GB 内存、1 天时限。 |
| `rl_diffusion.sh` | 提交扩散模型实验到 SLURM。请求 1× H200、8 CPU、64GB 内存、1 天时限。 |

### 技术笔记（`toy_model/*.md`）

| 文件 | 说明 |
|---|---|
| `technical_report.md` | RealNVP 架构实现细节、score function 计算方法、预训练+PPO 流水线（中文）。 |
| `FIM_plotting_report.md` | 记录 FIM 可视化实现和坐标变换方法。 |
| `RealNVP_limitations.md` | 分析仿射耦合流的数值稳定性问题（scale 截断、FIM 发散）。 |
| `theorem1_verification.md` | 定理 1 的数学验证及 C_v / Var(r) 关系分析。 |
| `exp_2d/RealNVP_training_explained.md` | RealNVP 训练实现的详细讲解。 |
| `exp_2d/test_model_report.md` | 模型架构验证的测试结果。 |

### 输出结果（`toy_model/output/`）

实验结果按配置组织：

| 目录 | 配置 |
|---|---|
| `flow/` | 默认 RealNVP（8 层，β=1.0） |
| `flow_beta_0.0/` | 消融实验：无 Boltzmann 预训练（均匀初始化） |
| `flow_beta_1.0/` | β=1.0 Boltzmann 预训练 |
| `flow_beta_1.0_layer_8/` | 8 耦合层 |
| `flow_beta_1.0_layer_32/` | 32 耦合层 |
| `flow_lr_1e-3/` | 较高学习率 |
| `flow_sigma_0.5/` | 不同噪声水平 |
| `flow_beta_1.0_PPO_batch_256/` | 较小 PPO 批量 |
| `diffusion/` | 离散扩散模型 |

每个目录包含 PNG 可视化图表（地形、轨迹、收敛曲线等）、指标日志和可选的 GIF 动画。

## 依赖

- PyTorch、NumPy、Matplotlib、SciPy
- `cma`（CMA-ES 优化）
- scikit-learn（PCA）
- CUDA（可选，用于 GPU 加速）
