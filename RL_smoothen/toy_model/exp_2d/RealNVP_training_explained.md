# RealNVP 训练流程代码解析

本文档解析 `exp_2d/model.py` 及其关联模块中 **RealNVP（仿射耦合）** 的完整训练流程。Spline 相关代码不在讨论范围内。

---

## 目录

1. [整体思路](#1-整体思路)
2. [奖励函数 — `env.py`](#2-奖励函数--envpy)
3. [仿射耦合层 — `AffineCoupling`](#3-仿射耦合层--affinecoupling)
4. [流模型 — `FlowModel`](#4-流模型--flowmodel)
5. [预训练阶段](#5-预训练阶段)
6. [PPO 强化学习微调阶段](#6-ppo-强化学习微调阶段)
7. [主函数训练流水线](#7-主函数训练流水线)
8. [求导计算详解](#8-求导计算详解)

---

## 1. 整体思路

本项目的核心思路是：

> **用 Normalizing Flow（RealNVP）作为策略网络，先通过极大似然预训练学会一个初始分布，再用 PPO 强化学习微调，使其采样集中到 2D Rastrigin 函数的全局最优点（原点）附近。**

训练分两步：
1. **预训练（Supervised）**：用 Boltzmann 分布或高斯分布的样本做极大似然训练，让模型学会一个合理的初始分布。
2. **PPO 微调（RL）**：以负 Rastrigin 值为奖励信号，用 PPO 算法将分布推向高奖励区域。

---

## 2. 奖励函数 — `env.py`

```python
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    A = 10.0
    d = x.shape[-1]
    return A * d + (x ** 2 - A * torch.cos(2 * np.pi * x)).sum(dim=-1)

def reward(x: torch.Tensor) -> torch.Tensor:
    return -rastrigin(x)
```

| 要素 | 说明 |
|------|------|
| **Rastrigin 函数** | 经典多局部极值测试函数，全局最小值为 0，在原点 `(0,0)` 取到。表面布满由 `cos` 项产生的"波纹"，形成大量局部极小值。 |
| **奖励函数** | 取 Rastrigin 的负值，即 `reward = -rastrigin(x)`。最大奖励为 0（在原点取到），其余位置奖励均为负。 |
| **为什么选它** | 大量局部极值 + 一个全局极值，是测试优化算法能否逃离局部最优的经典 benchmark。 |

---

## 3. 仿射耦合层 — `AffineCoupling`

这是 RealNVP 的核心构件。对于 2D 输入 `x = (x₀, x₁)`，每一层**固定一个维度、变换另一个维度**。

### 3.1 构造函数 `__init__`

```python
class AffineCoupling(nn.Module):
    def __init__(self, fix_dim, hidden_dim=64):
        super().__init__()
        self.fix_dim = fix_dim          # 0 或 1，表示哪个维度被固定
        self.transform_dim = 1 - fix_dim  # 被变换的维度
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),     # 输入：固定维度的值（标量）
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),     # 输出：s 和 t 两个标量
        )
        nn.init.zeros_(self.net[-1].weight)  # 最后一层初始化为 0
        nn.init.zeros_(self.net[-1].bias)    # 保证初始时变换为恒等映射
```

| 要素 | 说明 |
|------|------|
| `fix_dim` | 决定哪个维度不变。`fix_dim=0` 表示 x₀ 不变、变换 x₁；`fix_dim=1` 则反之。 |
| `net` | 一个 3 层 MLP（1 → hidden → hidden → 2），输入固定维度的值，输出仿射参数 `(s, t)`。 |
| **零初始化** | 最后一层权重和偏置初始化为 0，使得 `s=0, t=0`，即 `y = x·exp(0) + 0 = x`。初始时整个流模型是恒等映射，这是训练稳定性的关键技巧。 |

### 3.2 前向传播 `forward`

```python
def forward(self, x):
    x_fix = x[:, self.fix_dim:self.fix_dim + 1]      # 固定的维度
    x_var = x[:, self.transform_dim:self.transform_dim + 1]  # 待变换的维度

    st = self.net(x_fix)         # 用固定维度预测仿射参数
    s, t = st[:, 0:1], st[:, 1:2]
    s = s.clamp(-2, 2)           # 限制 s 范围，防止 exp(s) 爆炸

    y_var = x_var * s.exp() + t  # 仿射变换：缩放 + 平移
    log_det = s.squeeze(-1)      # log|det(J)| = s（因为 dy/dx = exp(s)）

    if self.fix_dim == 0:
        y = torch.cat([x_fix, y_var], dim=1)   # 拼回 2D
    else:
        y = torch.cat([y_var, x_fix], dim=1)
    return y, log_det
```

**数学公式**：给定输入 `(x_fix, x_var)`，前向变换为：

$$y_{\text{var}} = x_{\text{var}} \cdot e^{s(x_{\text{fix}})} + t(x_{\text{fix}})$$

| 要素 | 说明 |
|------|------|
| **仿射变换** | `y = x * exp(s) + t`，其中 `s, t` 是固定维度经 MLP 算出的。这就是 "Real-valued Non-Volume Preserving" 的含义——`exp(s)` 改变了体积。 |
| **`s.clamp(-2, 2)`** | 将缩放因子限制在 `[exp(-2), exp(2)] ≈ [0.135, 7.389]`，防止数值不稳定。 |
| **log_det** | 雅可比行列式的对数。由于仿射变换的雅可比是三角矩阵，行列式就是对角元素之积。对于单个维度的缩放，`log|det| = s`。 |
| **拼接输出** | 变换后的维度和固定维度按原来的顺序拼回 2D 向量。 |

### 3.3 逆向传播 `inverse`

```python
def inverse(self, y):
    y_fix = y[:, self.fix_dim:self.fix_dim + 1]
    y_var = y[:, self.transform_dim:self.transform_dim + 1]

    st = self.net(y_fix)
    s, t = st[:, 0:1], st[:, 1:2]
    s = s.clamp(-2, 2)

    x_var = (y_var - t) * (-s).exp()   # 逆变换：先减平移，再除以缩放
    log_det = -s.squeeze(-1)           # 逆变换的 log_det 取负

    if self.fix_dim == 0:
        x = torch.cat([y_fix, x_var], dim=1)
    else:
        x = torch.cat([x_var, y_fix], dim=1)
    return x, log_det
```

**数学公式**：

$$x_{\text{var}} = (y_{\text{var}} - t(y_{\text{fix}})) \cdot e^{-s(y_{\text{fix}})}$$

| 要素 | 说明 |
|------|------|
| **逆变换可解析计算** | 这是耦合层设计的精髓：因为固定维度不变（`y_fix = x_fix`），所以可以用 `y_fix` 重新计算 `s, t`，然后解析地求出 `x_var`。 |
| **log_det 取负** | 逆变换的雅可比行列式是前向的倒数，取对数后变号。 |

---

## 4. 流模型 — `FlowModel`

`FlowModel` 将多个耦合层堆叠成一个完整的 Normalizing Flow。

### 4.1 构造函数

```python
class FlowModel(nn.Module):
    def __init__(self, n_layers=8, hidden_dim=64, coupling='spline', n_bins=8):
        super().__init__()
        # 当 coupling='affine' 时（RealNVP）：
        self.layers = nn.ModuleList([
            AffineCoupling(fix_dim=i % 2, hidden_dim=hidden_dim)
            for i in range(n_layers)
        ])
```

| 要素 | 说明 |
|------|------|
| **`fix_dim=i % 2`** | 奇偶交替固定维度。第 0 层固定 x₀ 变换 x₁，第 1 层固定 x₁ 变换 x₀，第 2 层又固定 x₀ …… 这样每个维度都会被充分变换。 |
| **默认 8 层** | 8 层仿射耦合，每个维度各被变换 4 次，表达能力足以处理 2D 问题。 |

### 4.2 前向传播 `forward`：z → x

```python
def forward(self, z):
    log_det = torch.zeros(z.shape[0], device=z.device)
    x = z
    for layer in self.layers:
        x, ld = layer(x)
        log_det = log_det + ld   # 累加每层的 log_det
    return x, log_det
```

将隐空间 `z` 依次通过所有耦合层，变换到数据空间 `x`。总 log-det 是各层 log-det 之和（链式法则）。

### 4.3 逆向传播 `inverse`：x → z

```python
def inverse(self, x):
    log_det = torch.zeros(x.shape[0], device=x.device)
    z = x
    for layer in reversed(self.layers):  # 逆序遍历
        z, ld = layer.inverse(z)
        log_det = log_det + ld
    return z, log_det
```

从数据空间 `x` 反推隐变量 `z`。注意层的遍历顺序是**逆序**的。

### 4.4 采样 `sample`

```python
def sample(self, n, device='cpu'):
    z = torch.randn(n, 2, device=device)          # 从标准正态采 z
    x, log_det_fwd = self.forward(z)               # z → x
    log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)  # log N(z; 0, I)
    log_prob = log_pz - log_det_fwd                 # 变量替换公式
    return x, log_prob
```

| 要素 | 说明 |
|------|------|
| **基础分布** | 2D 标准正态 `N(0, I₂)`。 |
| **变量替换公式** | `log p(x) = log p(z) - log|det(∂x/∂z)|`。前向传播将 z 映射到 x，同时算出雅可比行列式。 |
| **返回值** | 同时返回样本 `x` 和对应的对数概率 `log p(x)`，后者在 PPO 中作为 `old_log_prob` 使用。 |

### 4.5 对数概率 `log_prob`

```python
def log_prob(self, x):
    z, log_det_inv = self.inverse(x)                # x → z
    log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)
    return log_pz + log_det_inv                     # 注意这里是 +
```

给定数据点 `x`，计算其在模型下的对数概率。方向是 x → z（逆向），所以公式中 log_det 的符号与 `sample` 相反：

$$\log p(x) = \log p_z(f^{-1}(x)) + \log\left|\det\frac{\partial f^{-1}}{\partial x}\right|$$

这个函数在**预训练**（极大似然）和 **PPO**（计算 ratio）中都被频繁调用。

### 4.6 参数的扁平化操作

```python
def get_flat_params(self):
    return torch.cat([p.data.reshape(-1) for p in self.parameters()])

def set_flat_params(self, flat):
    offset = 0
    for p in self.parameters():
        numel = p.numel()
        p.data.copy_(flat[offset:offset + numel].reshape(p.shape))
        offset += numel
```

| 要素 | 说明 |
|------|------|
| `get_flat_params` | 将所有参数展平为一个 1D 向量。用于保存参数快照（PPO 轨迹记录、发散恢复）。 |
| `set_flat_params` | 从 1D 向量恢复参数。用于发散恢复时回滚到历史最优参数。 |

---

## 5. 预训练阶段

预训练的目标是让流模型学会一个合理的初始分布，为后续 PPO 提供好的起点。有两种预训练方式可选：

### 5.1 Boltzmann 预训练 (`boltzmann.py`)

```python
def sample_boltzmann(n, beta=0.1):
    # 拒绝采样：p(x) ∝ exp(-β · Rastrigin(x))
    samples = []
    collected = 0
    while collected < n:
        batch = max(n * 50, 1024)
        x = torch.rand(batch, 2) * 10 - 5           # 提议分布：Uniform[-5,5]²
        accept_prob = torch.exp(-beta * rastrigin(x)) # 接受概率
        mask = torch.rand(batch) < accept_prob
        accepted = x[mask]
        if len(accepted) > 0:
            samples.append(accepted)
            collected += len(accepted)
    return torch.cat(samples)[:n]
```

| 要素 | 说明 |
|------|------|
| **Boltzmann 分布** | `p(x) ∝ exp(-β · Rastrigin(x))`，β 越大分布越集中在 Rastrigin 极小值附近。β=0 退化为均匀分布。 |
| **拒绝采样** | 因为 `Rastrigin(x) ≥ 0`，所以 `exp(-β·R(x)) ≤ 1`，可以直接用均匀分布作为提议分布。 |
| **意义** | 预训练后模型就已经倾向于在奖励较高的区域采样，降低了 PPO 的探索难度。 |

```python
def pretrain_boltzmann(model, n_epochs=100, batch_size=512, lr=1e-3, beta=0.1,
                       dataset_size=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        target = sample_boltzmann(batch_size, beta=beta)  # 采样目标分布
        nll = -model.log_prob(target).mean()              # 负对数似然
        optimizer.zero_grad()
        nll.backward()
        optimizer.step()
```

| 要素 | 说明 |
|------|------|
| **训练目标** | 最小化负对数似然 `NLL = -E[log p_model(x)]`，等价于最小化 KL(p_data ‖ p_model)。 |
| **`model.log_prob(target)`** | 调用上面的逆向传播，计算目标样本在模型下的对数概率。 |
| **`dataset_size`** | 为 0 时每个 epoch 重新采样（在线学习）；大于 0 时预采固定数据集（离线学习）。 |

### 5.2 高斯预训练 (`gaussian.py`)

```python
def pretrain_gaussian(model, n_epochs=100, batch_size=512, lr=1e-3,
                      std=2.0, dataset_size=0):
    # 与 Boltzmann 预训练结构完全相同，只是目标分布换成 N(0, std²·I)
    target = torch.randn(batch_size, 2) * std
    nll = -model.log_prob(target).mean()
```

更简单的预训练：让模型学会生成高斯分布样本。相比 Boltzmann 预训练，这个初始分布不利用任何奖励信息。

---

## 6. PPO 强化学习微调阶段 (`ppo.py`)

预训练完成后，用 PPO 算法继续优化模型，使其采样分布逐渐集中到高奖励区域。

### 6.1 初始化

```python
def ppo_train(model, n_iters=200, batch_size=256, ppo_epochs=2,
              clip_eps=0.2, lr=3e-4, kl_coeff=0.5, ...):
    base_model = copy.deepcopy(model)   # 深拷贝预训练模型作为 KL 惩罚的参考
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False         # 冻结，不参与训练

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
```

| 要素 | 说明 |
|------|------|
| `base_model` | 预训练后的模型快照，用于计算 KL 散度惩罚项 `KL(π_current ‖ π_pretrain)`，防止策略偏离初始分布太远。 |
| **冻结参数** | base_model 只用于前向推理（计算 log_prob），不需要梯度。 |

### 6.2 采样与奖励计算

```python
for it in range(n_iters):
    model.eval()
    with torch.no_grad():
        x, old_log_prob = model.sample(batch_size)   # 从当前策略采样
        r = reward(x)                                 # 计算奖励 = -Rastrigin(x)

        # 过滤无效样本（NaN / Inf）
        valid = (torch.isfinite(x).all(dim=-1)
                 & torch.isfinite(old_log_prob) & torch.isfinite(r))
        # ... 过滤后 ...
        r_normalized = (r - r.mean()) / (r.std() + 1e-8)  # 标准化奖励
```

| 要素 | 说明 |
|------|------|
| **`model.sample`** | 从标准正态采 z，经前向传播得到 x 和 log_prob。 |
| **有效性过滤** | 如果某些样本产生了 NaN/Inf（数值不稳定），直接丢弃。若有效样本不足 32 个，回滚到最佳参数并降低学习率。 |
| **奖励标准化** | 减均值除标准差，让 PPO 的优势估计更稳定。 |

### 6.3 PPO 策略更新

```python
    model.train()
    for _ in range(ppo_epochs):           # 对同一批数据做多轮更新
        new_log_prob = model.log_prob(x)  # 用更新后的模型重新算 log_prob

        log_ratio = (new_log_prob - old_log_prob).clamp(-5, 5)
        ratio = log_ratio.exp()           # π_new(x) / π_old(x)

        # PPO Clipped Objective
        surr1 = ratio * r_normalized
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * r_normalized
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL 惩罚（相对于预训练模型）
        with torch.no_grad():
            base_log_prob = base_model.log_prob(x)
        kl = (new_log_prob - base_log_prob).clamp(-20, 20).mean()

        loss = policy_loss + kl_coeff * kl  # 总损失

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
        optimizer.step()
```

| 要素 | 说明 |
|------|------|
| **重要比率 ratio** | `π_new(x) / π_old(x) = exp(log π_new - log π_old)`，衡量策略更新幅度。 |
| **PPO Clip** | `min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)`，当 ratio 偏离 1 太多时截断，防止策略更新步长过大。默认 `ε=0.2`。 |
| **KL 惩罚** | `kl_coeff * KL(π_current ‖ π_pretrain)`，拉住策略不要偏离预训练分布太远。这是本项目特有的设计——标准 PPO 通常不加这个约束。 |
| **梯度裁剪** | `clip_grad_norm` 限制梯度范数 ≤ 1.0，防止梯度爆炸。 |
| **多轮更新** | `ppo_epochs=2`，对同一批采样数据做 2 轮更新，提高样本利用效率。 |

### 6.4 发散恢复机制

```python
        # NaN 恢复
        if not torch.isfinite(loss):
            model.set_flat_params(best_params.clone())
            nan_count += 1
            break

    # 发散恢复
    if mean_r < best_mean_r - diverge_threshold:
        model.set_flat_params(best_params.clone())
        diverge_count += 1

    # 记录最优参数
    if mean_r > best_mean_r:
        best_mean_r = mean_r
        best_params = model.get_flat_params().clone().cpu()
```

| 要素 | 说明 |
|------|------|
| **NaN 恢复** | 如果损失变成 NaN，回滚到历史最优参数，同时将学习率减半（下限为初始 lr 的 5%）。 |
| **发散恢复** | 如果当前平均奖励比历史最优低超过 `diverge_threshold`（默认 5.0），判定为发散，回滚参数。 |
| **最优参数追踪** | 始终记录历史最佳参数，训练结束后恢复到最佳状态。 |

### 6.5 训练结束

```python
    if best_params is not None:
        model.set_flat_params(best_params)
        print(f"  Restored best model (mean_r={best_mean_r:.4f})")
    return history
```

训练完成后恢复到历史最优参数，返回包含完整训练记录的 `history` 字典。

---

## 7. 主函数训练流水线

`model.py` 的 `__main__` 部分定义了完整的 5 步流水线：

### Step 1: 构建模型 + 预训练

```python
model = FlowModel(n_layers=args.n_layers, hidden_dim=args.hidden_dim,
                  coupling=args.coupling)      # coupling='affine' → RealNVP

if args.pretrain == 'gaussian':
    pretrain_gaussian(model, ...)
else:
    pretrain_boltzmann(model, ...)              # 默认 Boltzmann 预训练

pretrain_params = model.get_flat_params().clone()  # 保存预训练参数快照
```

### Step 2: PPO 微调

```python
ppo_history = ppo_train(
    model,
    n_iters=args.ppo_iters,        # 默认 200 轮
    batch_size=args.ppo_batch,      # 默认 256
    ppo_epochs=args.ppo_epochs,     # 默认 2
    clip_eps=0.2,
    lr=args.ppo_lr,                 # 默认 3e-4
    kl_coeff=args.ppo_kl,           # 默认 0.5
    diverge_threshold=args.diverge_thresh,  # 默认 5.0
)
```

### Step 3: CMA-ES 对照实验

```python
cmaes_history = cmaes_optimize(n_evals=51200, sigma0=3.0, seed=42)
```

作为对比基线，用无梯度的 CMA-ES 直接在参数空间优化（不使用流模型）。

### Step 4: PCA + FIM 分析

```python
d1, d2, pca_center, traj_alpha, traj_beta, explained_var = \
    compute_pca_directions(ppo_history['param_snapshots'])

traj_fim = compute_trajectory_fim(model, ppo_history['param_snapshots'], ...)
```

对 PPO 训练轨迹做 PCA 降维，并沿轨迹计算 Fisher 信息矩阵（FIM），用于分析参数空间的几何结构。

### Step 5: 可视化

生成一系列图表，包括：
- 损失景观对比图
- PPO vs CMA-ES 轨迹动画
- 收敛曲线
- 预训练数据集分布
- 预训练 vs PPO 后的采样分布对比
- 分布演化动画
- FIM 特征值演化
- KL 效率分析

---

## 附：数据流全景图

```
                          预训练阶段                        PPO 阶段
                    ┌─────────────────┐            ┌─────────────────────┐
                    │  Boltzmann 采样  │            │  model.sample(z→x)  │
                    │  或 Gaussian 采样 │            │         ↓           │
                    │       ↓         │            │   reward(x) = -R(x) │
                    │  target samples │            │         ↓           │
                    │       ↓         │            │  PPO clipped loss   │
                    │  model.log_prob │            │  + KL(π‖π_pretrain) │
                    │       ↓         │            │         ↓           │
                    │  NLL = -E[logp] │            │  optimizer.step()   │
                    │       ↓         │            │         ↓           │
                    │  optimizer.step │            │  发散检测 & 回滚     │
                    └─────────────────┘            └─────────────────────┘

    z ~ N(0,I) ──→ [Layer0: fix x₀, transform x₁]
                ──→ [Layer1: fix x₁, transform x₀]
                ──→ [Layer2: fix x₀, transform x₁]
                ──→ ...（共 8 层）
                ──→ x ∈ R²
```

---

## 8. 求导计算详解

本节详细分析 PyTorch autograd 如何在 RealNVP 的两个训练阶段中完成梯度计算，并逐行对应到源代码。

### 8.1 需要求导的目标函数

两个阶段的损失函数不同，但梯度都需要穿过 `model.log_prob(x)`：

| 阶段 | 损失函数 | 代码位置 | 梯度路径 |
|------|----------|----------|----------|
| **预训练** | `NLL = -model.log_prob(target).mean()` | `boltzmann.py:55` / `gaussian.py:38` | ∂NLL/∂θ 直接穿过 `log_prob` |
| **PPO** | `loss = policy_loss + kl_coeff * kl` | `ppo.py:112` | ∂loss/∂θ 穿过 `new_log_prob = model.log_prob(x)` |

因此 **`log_prob` 的求导是整个系统的核心**。下面逐层拆解。

### 8.2 `log_prob` 的计算图

> 对应 `model.py:250-253`

```python
def log_prob(self, x):
    z, log_det_inv = self.inverse(x)                      # ① model.py:251
    log_pz = -0.5 * (z.pow(2) + np.log(2*np.pi)).sum(-1)  # ② model.py:252
    return log_pz + log_det_inv                            # ③ model.py:253
```

最终输出是两项之和，梯度分别从两条路径回传：

```
                          log_prob(x)
                         ╱           ╲
                    log_pz        log_det_inv
                      |               |
                  ②  z.pow(2)     Σ(-sₖ)        ← 各层的 log_det 累加
                      |               |
                  ①  z = f⁻¹(x)   sₖ = net_k(·)  ← 各层 MLP 的输出
                      |               |
                   [逆向耦合层链]   [各层 MLP 参数 θ]
```

**两条梯度路径**：
- **路径 A**（通过 `log_pz`）：`∂log_pz/∂θ = ∂log_pz/∂z · ∂z/∂θ`，即基础分布对数概率对 z 求导，再链式传播到参数。
- **路径 B**（通过 `log_det_inv`）：`∂log_det_inv/∂θ = Σ ∂(-sₖ)/∂θ`，即各层 log-det 直接对 MLP 参数求导。

#### 代码实现：路径 A — `log_pz` 的求导

```python
# model.py:252
log_pz = -0.5 * (z.pow(2) + np.log(2 * np.pi)).sum(-1)
```

PyTorch autograd 在此记录的操作链：

| 步骤 | PyTorch 操作 | autograd 记录的 backward 函数 | 回传时的梯度 |
|------|-------------|------------------------------|-------------|
| 1 | `z.pow(2)` | `PowBackward0` | `∂/∂z = 2z` |
| 2 | `(·).sum(-1)` | `SumBackward1` | 将标量梯度广播回 `(batch, 2)` |
| 3 | `* (-0.5)` | `MulBackward0` | `∂/∂(·) = -0.5` |

合并后：`∂log_pz/∂z = -z`（形状 `(batch, 2)`），这就是标准正态分布的 score function。

这个梯度 `-z` 会继续回传到 `self.inverse(x)` 中产生 `z` 的所有操作。

#### 代码实现：路径 B — `log_det_inv` 的求导

```python
# model.py:237-239 (inverse 内部)
def inverse(self, x):
    log_det = torch.zeros(x.shape[0], device=x.device)
    z = x
    for layer in reversed(self.layers):
        z, ld = layer.inverse(z)
        log_det = log_det + ld    # 累加各层 ld = -s_k
    return z, log_det
```

`log_det` 是一个标量求和链：`log_det = ld₇ + ld₆ + ... + ld₀`。PyTorch 对加法的反向传播是恒等的：`∂log_det/∂ldₖ = 1`。因此每层的 `ldₖ = -sₖ` 都直接收到来自最终 loss 的梯度，不经过其他层的衰减。

### 8.3 单层 AffineCoupling 的求导

> 对应 `model.py:131-146` (`inverse` 方法)

`log_prob` 调用 `inverse`，以下逐行分析 `inverse` 中每个操作的梯度。

#### (a) 维度分割与 MLP 前向传播

```python
# model.py:132-133  维度分割
y_fix = y[:, self.fix_dim:self.fix_dim + 1]        # 切片，梯度直通
y_var = y[:, self.transform_dim:self.transform_dim + 1]  # 切片，梯度直通

# model.py:135  MLP 前向
st = self.net(y_fix)
```

`self.net` 的定义在 `model.py:104-109`：

```python
self.net = nn.Sequential(
    nn.Linear(1, hidden_dim),      # W₁: (64,1), b₁: (64,)     ← θ 的一部分
    nn.ReLU(),                     # 逐元素，梯度 = 1{h>0}
    nn.Linear(hidden_dim, hidden_dim),  # W₂: (64,64), b₂: (64,)
    nn.ReLU(),
    nn.Linear(hidden_dim, 2),      # W₃: (2,64), b₃: (2,)
)
```

**MLP 前向传播中 autograd 记录的操作**：

```
y_fix                          # shape: (batch, 1)
  │
  ▼  h₁ = W₁ @ y_fix + b₁    # AddmmBackward0    ← 记录 W₁, b₁, y_fix
  │
  ▼  a₁ = ReLU(h₁)            # ReluBackward0     ← 记录 h₁ 的正/负 mask
  │
  ▼  h₂ = W₂ @ a₁ + b₂       # AddmmBackward0    ← 记录 W₂, b₂, a₁
  │
  ▼  a₂ = ReLU(h₂)            # ReluBackward0
  │
  ▼  st = W₃ @ a₂ + b₃       # AddmmBackward0    ← 记录 W₃, b₃, a₂
```

反向传播时，对 W₃ 的梯度为例：
```
∂L/∂W₃ = (∂L/∂st)ᵀ @ a₂      # 上游梯度 × 本层输入的转置
```

其中 `∂L/∂st` 来自 `s` 和 `t` 的下游使用（见 (b)(c)(d)）。

**关于 `y_fix` 的梯度**：`y_fix` 作为 MLP 输入也会收到梯度 `∂L/∂y_fix = W₁ᵀ @ ∂L/∂h₁`。当 `y_fix` 来自上一层的输出时，这个梯度会继续回传到更早的层。当 `y_fix` 是最外层输入 `target`（预训练）或 `x`（PPO，无梯度）时，这条路径终止。

#### (b) 参数拆分与 clamp

```python
# model.py:136  拆分 s 和 t
s, t = st[:, 0:1], st[:, 1:2]    # 切片操作，梯度直通（SliceBackward0）

# model.py:137  数值截断
s = s.clamp(-2, 2)               # ClampBackward1
```

`clamp` 在 autograd 中记录了 `ClampBackward1`，反向传播逻辑：

```python
# PyTorch 内部等效实现：
grad_input = grad_output.clone()
grad_input[s_raw < -2] = 0.0     # 低于下界：梯度归零
grad_input[s_raw > 2] = 0.0      # 高于上界：梯度归零
# -2 ≤ s_raw ≤ 2 的部分：梯度原样通过
```

$$\frac{\partial s_{\text{clamped}}}{\partial s_{\text{raw}}} = \begin{cases} 1 & \text{if } -2 < s_{\text{raw}} < 2 \\ 0 & \text{otherwise} \end{cases}$$

当 `s_raw` 超出 `[-2, 2]` 时，**梯度被完全截断为 0**。效果：
- 对应样本的 MLP 参数不会收到来自这个样本的梯度信号
- 隐式正则化：阻止缩放因子过大或过小
- 控制 `exp(s)` 在 `[e⁻², e²] ≈ [0.135, 7.389]` 范围内

#### (c) 逆仿射变换

```python
# model.py:139
x_var = (y_var - t) * (-s).exp()
```

这一行包含 4 个 autograd 操作：

| 步骤 | 操作 | autograd 节点 | 反向梯度 |
|------|------|--------------|----------|
| 1 | `y_var - t` → `diff` | `SubBackward0` | `∂L/∂t = -∂L/∂diff` |
| 2 | `-s` → `neg_s` | `NegBackward0` | `∂L/∂s += -∂L/∂neg_s` |
| 3 | `neg_s.exp()` → `scale` | `ExpBackward0` | `∂L/∂neg_s = ∂L/∂scale · scale` |
| 4 | `diff * scale` → `x_var` | `MulBackward0` | `∂L/∂diff = ∂L/∂x_var · scale`<br>`∂L/∂scale = ∂L/∂x_var · diff` |

**合并后对 s 和 t 的梯度**（设上游梯度为 `g = ∂L/∂x_var`）：

$$\frac{\partial L}{\partial t} = -g \cdot e^{-s}$$

$$\frac{\partial L}{\partial s} = -g \cdot (y_{\text{var}} - t) \cdot e^{-s} \cdot (-1) = g \cdot (y_{\text{var}} - t) \cdot e^{-s}$$

注意 `exp(-s)` 是一个始终为正的缩放因子——当 `s` 变大时 `exp(-s)` 接近 0，梯度衰减；当 `s` 变小时 `exp(-s)` 放大。`s.clamp(-2,2)` 限制了这个放大/衰减范围。

然后 `∂L/∂s` 和 `∂L/∂t` 拼回 `∂L/∂st = [∂L/∂s, ∂L/∂t]`（形状 `(batch, 2)`），通过 MLP 的 `AddmmBackward0` 链继续回传到 `W₁, W₂, W₃, b₁, b₂, b₃`。

#### (d) log_det

```python
# model.py:140
log_det = -s.squeeze(-1)       # NegBackward0 + SqueezeBackward1
```

`∂L/∂s` 从此处额外获得一个分量 `-∂L/∂log_det`。由于 `log_det` 在 `FlowModel.inverse`（`model.py:239`）中被直接加到总 `log_det` 上，`∂L/∂log_det = ∂L/∂(总log_det) = 1`（加法的梯度），所以：

$$\frac{\partial L}{\partial s}\bigg|_{\text{来自 log\_det}} = -1 \cdot \frac{\partial L}{\partial \text{total\_log\_det}}$$

**s 的总梯度 = 来自 (c) x_var 的梯度 + 来自 (d) log_det 的梯度**。PyTorch autograd 自动累加这两个来源，因为 `s` 这个张量被 (c) 和 (d) 共同使用。

#### (e) 输出拼接

```python
# model.py:142-145
if self.fix_dim == 0:
    x = torch.cat([y_fix, x_var], dim=1)     # CatBackward0
else:
    x = torch.cat([x_var, y_fix], dim=1)
```

`torch.cat` 的反向传播是 `torch.split`——把上游梯度按维度拆开，分别传给 `y_fix` 和 `x_var`。

### 8.4 多层链式求导：梯度如何穿越 8 层耦合

> 对应 `model.py:235-241`

```python
def inverse(self, x):
    log_det = torch.zeros(x.shape[0], device=x.device)  # model.py:236
    z = x
    for layer in reversed(self.layers):  # model.py:238: Layer7 → Layer6 → ... → Layer0
        z, ld = layer.inverse(z)         # model.py:239
        log_det = log_det + ld           # model.py:240
    return z, log_det                    # model.py:241
```

以 8 层为例（`n_layers=8`，`model.py:214`），具体展开这个循环：

```python
# 实际执行顺序（逆序）：
z = x                               # 初始值

z, ld7 = layers[7].inverse(z)       # fix_dim=1: 固定 dim1，变换 dim0
log_det = 0 + ld7                   # ld7 = -s₇

z, ld6 = layers[6].inverse(z)       # fix_dim=0: 固定 dim0（刚被 Layer7 变换过），变换 dim1
log_det = log_det + ld6             # ld6 = -s₆

z, ld5 = layers[5].inverse(z)       # fix_dim=1
log_det = log_det + ld5

# ... 以此类推 ...

z, ld0 = layers[0].inverse(z)       # fix_dim=0
log_det = log_det + ld0
```

**梯度回传时也是逐层逆序**，但方向相反——autograd 按计算图的拓扑逆序回传：

```
∂L/∂z (来自 log_pz = -0.5*z²)
  │
  ▼  Layer0.inverse 的反向传播
  │    ∂L/∂θ₀ ← 本层参数获得梯度
  │    ∂L/∂z₁ ← 传给 Layer0 的输入（= Layer1 的输出）
  │
  ▼  Layer1.inverse 的反向传播
  │    ∂L/∂θ₁ ← 本层参数获得梯度
  │    ∂L/∂z₂ ← 继续回传
  │
  ... (逐层回传)
  │
  ▼  Layer7.inverse 的反向传播
       ∂L/∂θ₇ ← 本层参数获得梯度
       ∂L/∂x  ← 传给最初输入（但 x 无 requires_grad，此处终止）
```

**奇偶交替的维度分离如何简化梯度传播**：

以 Layer7（fix_dim=1）→ Layer6（fix_dim=0）为例：

```
Layer7.inverse 的输入:  y = (y₀, y₁)
  fix_dim=1 → 固定 y₁，变换 y₀
  s₇, t₇ = net₇(y₁)                  # MLP 以 y₁ 为输入
  x₀ = (y₀ - t₇) * exp(-s₇)          # 只改变 dim0
  输出: z₇ = (x₀, y₁)                 # dim1 不变

Layer6.inverse 的输入:  z₇ = (x₀, y₁)
  fix_dim=0 → 固定 x₀（Layer7 刚变换过的值），变换 y₁
  s₆, t₆ = net₆(x₀)                  # MLP 以 x₀ 为输入 ← x₀ 依赖 θ₇!
  x₁ = (y₁ - t₆) * exp(-s₆)          # 只改变 dim1
  输出: z₆ = (x₀, x₁)
```

**关键观察**：Layer6 的 MLP 输入 `x₀` 是 Layer7 的输出，它依赖 θ₇。因此：
- `∂L/∂θ₇` **不仅**来自 Layer7 自身的 `s₇, t₇, log_det₇`
- **还会**经过 Layer6 的 `net₆(x₀)` → `s₆, t₆` → `x₁, log_det₆` → ... 一路传下去

PyTorch autograd 自动处理这个跨层依赖——它只关心计算图的拓扑结构，不区分"层"的边界。

**每层的梯度来自两个来源**（对第 k 层参数 θₖ）：

| 来源 | 路径 | 是否穿越其他层 |
|------|------|---------------|
| **来源 1：通过 z → log_pz** | `∂log_pz/∂z · ∂z/∂θₖ` | 是，需穿越 Layer k-1 到 Layer 0 |
| **来源 2：通过 log_det** | `∂(-sₖ)/∂θₖ`（直接依赖）+ `sⱼ` 通过 `hₖ` 间接依赖 θₖ 的部分 | 间接依赖部分需穿越其他层 |

### 8.5 预训练阶段的完整求导链

> 对应 `boltzmann.py:46-61`（Boltzmann）或 `gaussian.py:29-48`（Gaussian）

以 Boltzmann 预训练为例，逐行标注梯度状态：

```python
# boltzmann.py:46
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
# Adam 管理所有 θ = {W₁,b₁,W₂,b₂,W₃,b₃} × 8 层

for epoch in range(n_epochs):
    # boltzmann.py:49-53: 采样（或从固定数据集取）
    target = sample_boltzmann(batch_size, beta=beta)
    # target 是普通 tensor，requires_grad=False
    # → 它不会参与梯度计算，只是 log_prob 的输入数据

    # boltzmann.py:55: 前向传播 —— 构建计算图
    nll = -model.log_prob(target).mean()
    #         │              │         └── MeanBackward0: ∂/∂(·) = 1/N
    #         │              └── NegBackward0: ∂/∂(·) = -1
    #         └── 进入 model.inverse(target) → 8 层 AffineCoupling.inverse
    #             每层记录: AddmmBackward0 × 3 (MLP) + ClampBackward1 (s)
    #                      + SubBackward0 + ExpBackward0 + MulBackward0 (逆仿射)
    #                      + NegBackward0 (log_det)

    # boltzmann.py:57: 清零旧梯度
    optimizer.zero_grad()
    # 将所有 param.grad 置为 None 或 0

    # boltzmann.py:58: 反向传播 —— 遍历计算图，计算所有 ∂nll/∂θ
    nll.backward()
    # autograd 从 nll 出发，沿记录的 backward 函数链逆序回传
    # 每个 param 的 .grad 属性被累加
    # 完成后计算图被释放（默认 retain_graph=False）

    # boltzmann.py:59: 参数更新
    optimizer.step()
    # Adam 用 .grad 更新参数: θ_new = θ_old - lr * m̂/(√v̂ + ε)
    # 其中 m̂, v̂ 是梯度的一阶/二阶矩估计
```

**完整计算图可视化**（对于 batch 中的单个样本）：

```
target[i] = (x₀, x₁)  ─── requires_grad=False，梯度到此终止
    │
    ▼
Layer7.inverse:  fix_dim=1
    │  y_fix = x₁ (无梯度叶节点)
    │  st = net₇(x₁)       → 记录 AddmmBackward0 × 3, ReluBackward0 × 2
    │  s₇ = st[:,0].clamp   → 记录 SliceBackward0, ClampBackward1
    │  t₇ = st[:,1]         → 记录 SliceBackward0
    │  x_var₇ = (x₀-t₇)*exp(-s₇) → 记录 SubBackward0, NegBackward0,
    │                                      ExpBackward0, MulBackward0
    │  ld₇ = -s₇            → 记录 NegBackward0, SqueezeBackward1
    ▼
h₆ = (x_var₇, x₁)          → 记录 CatBackward0
    │
    ▼
Layer6.inverse:  fix_dim=0
    │  y_fix = x_var₇ (有梯度! 来自 Layer7 的输出)
    │  st = net₆(x_var₇)   → 梯度将回传到 x_var₇ → θ₇
    │  s₆, t₆, x_var₆, ld₆ = ...
    ▼
... (Layer5 ~ Layer0 同理)
    │
    ▼
z = (z₀, z₁)                ← 最终隐变量
    │
    ├──→ log_pz = -0.5 * Σ(zᵢ² + log2π)      [路径 A]
    │        反向梯度: ∂log_pz/∂z = -z
    │
    └──→ log_det = ld₇ + ld₆ + ... + ld₀      [路径 B]
              反向梯度: ∂log_det/∂ldₖ = 1 (对每个 k)
              │
              ▼
         log_prob = log_pz + log_det     → AddBackward0
              │
              ▼
         nll = -log_prob.mean()          → NegBackward0, MeanBackward0
              │
              ▼
         nll.backward()
```

**反向传播的实际执行顺序**（autograd 拓扑排序后）：

```
1. MeanBackward0:  grad = 1.0   →   grad = 1/N        (对每个样本)
2. NegBackward0:   grad = 1/N   →   grad = -1/N       (取负)
3. AddBackward0:   grad = -1/N  →   路径 A: -1/N 传给 log_pz
                                    路径 B: -1/N 传给 log_det
4. 路径 A: -z/N 传入 Layer0.inverse 的反向
5. 路径 B: -1/N 分别传入各层的 ld₀...ld₇
6. Layer0 反向 → Layer1 反向 → ... → Layer7 反向
   每层内部: MulBackward0 → ExpBackward0 → NegBackward0 → ClampBackward1
             → SliceBackward0 → AddmmBackward0 (×3, 穿过 MLP)
             → 权重 W₁,W₂,W₃ 和偏置 b₁,b₂,b₃ 的 .grad 被累加
```

### 8.6 PPO 阶段的完整求导链

> 对应 `ppo.py:63-123`

PPO 阶段的梯度路径更长，但关键仍然是 `model.log_prob(x)`。逐行标注：

```python
# ==== 采样阶段（ppo.py:64-86）—— 全程无梯度 ====
model.eval()                                         # ppo.py:64: 关闭 dropout/batchnorm（本模型无影响）
with torch.no_grad():                                # ppo.py:65: 禁用 autograd 记录
    x, old_log_prob = model.sample(batch_size)       # ppo.py:66
    # model.sample 内部 (model.py:243-248):
    #   z = torch.randn(n, 2)          # 基础采样
    #   x, log_det_fwd = self.forward(z)  # 前向变换
    #   log_pz = -0.5*(z²+log2π).sum()
    #   log_prob = log_pz - log_det_fwd
    # 因为在 no_grad 内，所有操作都不记录计算图
    # → x 和 old_log_prob 都没有 grad_fn，等同于纯数据

    r = reward(x)                                    # ppo.py:67: 纯数据
    # ...过滤、标准化...
    r_normalized = (r - r.mean()) / (r.std() + 1e-8) # ppo.py:86: 纯数据

# ==== 策略更新（ppo.py:88-123）—— 开始构建计算图 ====
model.train()                                        # ppo.py:88
for _ in range(ppo_epochs):                          # ppo.py:90: 默认 2 轮

    # ---- 唯一的梯度入口 ----
    new_log_prob = model.log_prob(x)                 # ppo.py:91
    # x 没有 requires_grad（来自 no_grad 块）
    # 但 model 的参数有 requires_grad=True
    # → autograd 记录从参数到 new_log_prob 的完整计算图
    # → 内部调用 model.inverse(x)，同预训练的计算图结构

    # ---- ratio 计算 ----
    log_ratio = (new_log_prob - old_log_prob).clamp(-5, 5)  # ppo.py:97
    # SubBackward0: old_log_prob 是常数 → ∂/∂new_log_prob = 1
    # ClampBackward1: |log_ratio| > 5 时梯度为 0

    ratio = log_ratio.exp()                          # ppo.py:98
    # ExpBackward0: ∂ratio/∂log_ratio = ratio

    # ---- PPO clipped objective ----
    surr1 = ratio * r_normalized                     # ppo.py:100
    # MulBackward0: ∂surr1/∂ratio = r_normalized (常数)

    surr2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * r_normalized  # ppo.py:101
    # ClampBackward1 + MulBackward0
    # ratio ∈ [0.8, 1.2] 时: ∂surr2/∂ratio = r_normalized
    # ratio ∉ [0.8, 1.2] 时: ∂surr2/∂ratio = 0 (clamp 截断)

    policy_loss = -torch.min(surr1, surr2).mean()    # ppo.py:102
    # MinBackward1: 选择较小者的梯度通过，较大者梯度为 0
    # NegBackward0 + MeanBackward0

    # ---- KL 散度惩罚 ----
    with torch.no_grad():                            # ppo.py:104
        base_log_prob = base_model.log_prob(x)       # ppo.py:105: 冻结模型，纯数据
    kl = (new_log_prob - base_log_prob).clamp(-20, 20).mean()  # ppo.py:110
    # SubBackward0: base_log_prob 是常数 → ∂/∂new_log_prob = 1
    # ClampBackward1 + MeanBackward0

    # ---- 总损失 ----
    loss = policy_loss + kl_coeff * kl               # ppo.py:112
    # AddBackward0: 两条支路的梯度相加

    # ---- 反向传播 ----
    optimizer.zero_grad()                            # ppo.py:120
    loss.backward()                                  # ppo.py:121

    # ---- 梯度裁剪 ----
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ppo.py:122
    # 计算所有参数梯度的全局 L2 范数:
    #   total_norm = sqrt(Σ ‖param.grad‖²)
    # 如果 total_norm > 1.0:
    #   param.grad *= (1.0 / total_norm)   对每个 param
    # 效果: 保持梯度方向不变，但缩放范数到 1.0

    # ---- 参数更新 ----
    optimizer.step()                                 # ppo.py:123
```

**PPO 中 `∂loss/∂new_log_prob` 的具体展开**：

对于 batch 中第 i 个样本，设 `nlp = new_log_prob[i]`：

```
                              loss
                            ╱      ╲
                   policy_loss    kl_coeff * kl
                        |              |
                        ▼              ▼
对 nlp 的梯度:    ∂policy_loss    kl_coeff * ∂kl
                  ─────────────   ───────────────
                   ∂nlp[i]           ∂nlp[i]
```

**policy_loss 支路**：

```python
# 展开 min(surr1, surr2) 对 nlp 的梯度:
ratio_i = exp(nlp - old_log_prob_i)    # old_log_prob_i 是常数
d_ratio = ratio_i                      # ∂ratio/∂nlp = exp(log_ratio) = ratio

if surr1_i <= surr2_i:
    # min 选择 surr1 → surr2 的梯度被丢弃
    d_surr = r_norm_i * d_ratio        # ∂surr1/∂nlp = r_norm * ratio
else:
    # min 选择 surr2
    if 1-eps <= ratio_i <= 1+eps:
        d_surr = r_norm_i * d_ratio    # ratio 未被 clamp
    else:
        d_surr = 0                     # ratio 被 clamp，梯度归零

d_policy_loss = -d_surr / N            # 前面有负号和 mean
```

**kl 支路**：

```python
# kl = mean(nlp - base_log_prob)，base_log_prob 是常数
d_kl = 1.0 / N                        # ∂kl/∂nlp[i] = 1/N
d_kl_term = kl_coeff * d_kl           # = kl_coeff / N
```

**合并**：

$$\frac{\partial \text{loss}}{\partial \text{nlp}[i]} = \underbrace{-\frac{r_i^{\text{norm}} \cdot \text{ratio}_i}{N} \cdot \mathbb{1}[\text{未被 clip}]}_{\text{policy\_loss 贡献}} + \underbrace{\frac{\text{kl\_coeff}}{N}}_{\text{kl 贡献}}$$

然后这个 `∂loss/∂nlp[i]` 作为上游梯度，沿 `model.log_prob` → `model.inverse` → 8 层 `AffineCoupling.inverse` 的计算图回传，最终到达每一层的 MLP 参数。

**与预训练的关键差异**：

| 方面 | 预训练 | PPO |
|------|--------|-----|
| 输入 x | 外部数据（无梯度） | `model.sample` 但被 `no_grad` 包裹（无梯度） |
| 上游梯度 `∂L/∂log_prob[i]` | 均匀的 `-1/N` | 被 `ratio`、`r_normalized`、`clip`、`min` 调制 |
| 额外正则 | 无 | `kl_coeff/N` 的常数梯度附加项 |
| 梯度后处理 | 无 | `clip_grad_norm_(·, 1.0)` 全局缩放 |

### 8.7 `torch.no_grad()` 的梯度阻断：代码实现

代码中有 3 处使用 `torch.no_grad()` 来阻断梯度路径：

**（1）PPO 采样阶段**（`ppo.py:65-66`）：

```python
with torch.no_grad():
    x, old_log_prob = model.sample(batch_size)
```

`torch.no_grad()` 的 PyTorch 内部实现：设置全局标志 `torch.is_grad_enabled() = False`，使得所有 tensor 操作**不记录 grad_fn**。结果：`x` 和 `old_log_prob` 的 `grad_fn = None`，等同于纯数据。

**为什么不对采样过程求导？** 这是 REINFORCE / score function 估计器的选择：

$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \pi_\theta}\left[R(x) \cdot \nabla_\theta \log \pi_\theta(x)\right]$$

梯度只通过 $\log \pi_\theta(x)$ 传播（在 `ppo.py:91` 的 `model.log_prob(x)` 中实现），采样点 x 被视为固定常数。

**（2）base_model 推理**（`ppo.py:104-109`）：

```python
with torch.no_grad():
    base_log_prob = base_model.log_prob(x)
```

`base_model` 在 `ppo.py:27-29` 被冻结：

```python
base_model = copy.deepcopy(model)    # 深拷贝
base_model.eval()
for p in base_model.parameters():
    p.requires_grad = False           # 参数本身也关闭梯度
```

双重保险：`requires_grad=False` + `torch.no_grad()`。`base_log_prob` 是固定参考值，KL 惩罚中只需对 `new_log_prob` 求导。

**（3）评估采样**（`ppo.py:150-156`）：

```python
model.eval()
with torch.no_grad():
    x_eval, _ = model.sample(2048)    # 仅用于统计，不需要梯度
    r_eval = reward(x_eval)
```

纯评估用途，计算 reward 方差等统计量，无需构建计算图。

### 8.8 数值稳定性措施对梯度的影响

代码中有多处数值保护，每一处都影响梯度行为：

| 操作 | 代码位置 | 前向效果 | 反向梯度效果 |
|------|----------|---------|-------------|
| `s.clamp(-2, 2)` | `model.py:120,137` | 限制缩放因子范围 | 超出 `[-2,2]` 时梯度为 0 |
| `log_ratio.clamp(-5, 5)` | `ppo.py:97` | ratio 限制在 `[e⁻⁵, e⁵]` | 极端 ratio 的样本不贡献梯度 |
| `torch.clamp(ratio, 0.8, 1.2)` | `ppo.py:101` | PPO clip | ratio 超出 clip 范围且为较小分支时，梯度为 0 |
| `kl.clamp(-20, 20)` | `ppo.py:110` | 限制 KL 值 | 极端 KL 的梯度被截断 |
| `clip_grad_norm_(·, 1.0)` | `ppo.py:122` | 不影响前向 | 全局梯度范数 > 1 时等比缩小所有 `.grad` |
| `torch.min(surr1, surr2)` | `ppo.py:102` | 取较小值 | 只有较小分支的梯度回传 |
| NaN 替换 | `ppo.py:94-95` | `new_log_prob` 中 NaN 替换为 `old_log_prob` | 被替换的样本梯度为 0（`torch.where` 的 mask 分支） |
| NaN loss 回滚 | `ppo.py:114-118` | 回滚参数 | 完全放弃本轮梯度 |

**NaN 替换的具体实现**（`ppo.py:93-95`）：

```python
bad = ~torch.isfinite(new_log_prob)         # 找出 NaN/Inf
if bad.any():
    new_log_prob = torch.where(bad, old_log_prob, new_log_prob)
```

`torch.where(cond, a, b)` 的反向梯度：
- `cond=True`（bad 样本）：梯度传给 `a`（`old_log_prob`，无 grad_fn），**梯度消失**
- `cond=False`（正常样本）：梯度传给 `b`（`new_log_prob`），**梯度正常传播**

效果：NaN 样本被静默跳过，不污染梯度。

### 8.9 为什么 RealNVP 的求导如此高效

总结 RealNVP 架构为梯度计算带来的三个关键优势：

**1. 三角雅可比 → log_det 无矩阵运算**

一般的 Normalizing Flow 需要计算 $\log|\det J_f|$，其中 $J_f$ 是一个 $d \times d$ 矩阵，行列式计算复杂度为 $O(d^3)$。但 RealNVP 的耦合结构使雅可比矩阵为三角形：

$$J = \begin{pmatrix} 1 & 0 \\ \frac{\partial y_{\text{var}}}{\partial x_{\text{fix}}} & e^s \end{pmatrix}$$

行列式 = 对角元素之积 = $1 \times e^s = e^s$，所以 $\log|\det J| = s$。

**代码对应**（`model.py:123` 和 `model.py:140`）：

```python
# forward: log_det 就是 s
log_det = s.squeeze(-1)          # model.py:123

# inverse: log_det 就是 -s
log_det = -s.squeeze(-1)         # model.py:140
```

不需要构造雅可比矩阵、不需要调用 `torch.det()` 或 `torch.slogdet()`。**求导时只需要 $\partial s / \partial \theta$**，就是一个标准的 MLP 反向传播。

**2. 逆变换可解析 → 不需要隐式微分**

某些流模型（如 Neural ODE）的逆变换需要迭代求解，反向传播需要 adjoint method。RealNVP 的逆变换是闭式的：

```python
# model.py:139 —— 一行代码，直接解析求逆
x_var = (y_var - t) * (-s).exp()
```

对比 Neural ODE 需要的 `torchdiffeq.odeint_adjoint()`，RealNVP 就是普通的 tensor 运算，autograd 原生支持。

**3. 维度分离 → 梯度路径不交叉**

```python
# model.py:114-116 —— 维度分割
x_fix = x[:, self.fix_dim:self.fix_dim + 1]        # 直通，梯度 = 1
x_var = x[:, self.transform_dim:self.transform_dim + 1]  # 被变换

# model.py:118 —— MLP 只读取固定维度
st = self.net(x_fix)   # x_var 完全不参与 s, t 的计算
```

固定维度 `x_fix` 作为 MLP 输入产生 `s, t`，变换维度 `x_var` 只被 `s, t` 作用。两条路径在本层内不交叉，使计算图简洁、autograd 高效。

---

## 附：关键超参数默认值

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `n_layers` | 8 | 耦合层数量 |
| `hidden_dim` | 64 | MLP 隐藏层宽度 |
| `coupling` | `'affine'` (命令行默认) | 耦合类型，affine = RealNVP |
| `pretrain_epochs` | 300 | 预训练轮数 |
| `pretrain_lr` | 1e-3 | 预训练学习率 |
| `ppo_iters` | 200 | PPO 迭代次数 |
| `ppo_lr` | 3e-4 | PPO 学习率 |
| `ppo_kl` | 0.5 | KL 惩罚系数 |
| `clip_eps` | 0.2 | PPO clip 范围 |
| `diverge_thresh` | 5.0 | 发散检测阈值 |
| `beta` | 0.0 | Boltzmann 温度（0 = 均匀分布） |
