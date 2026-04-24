# 技术报告：Normalizing Flow + PPO 的训练与精确 Score 计算

## 1. 网络架构：RealNVP 仿射耦合流

### 1.1 整体结构

模型 `FlowModel` 是一个 normalizing flow，将简单的基分布 $z \sim \mathcal{N}(0, I_2)$ 通过一系列可逆变换映射为复杂分布 $x = f_\theta(z)$。

```
z ∈ ℝ² ~ N(0, I)  →  Layer₁  →  Layer₂  →  ...  →  Layer₈  →  x ∈ ℝ²
```

默认配置：8 层仿射耦合层（`AffineCoupling`），每层内含一个 3 层 MLP（1→64→64→2），总参数量约 35K。

### 1.2 仿射耦合层（AffineCoupling）

每层将 2D 输入 $(x_{\text{fix}}, x_{\text{var}})$ 变换为：

$$y_{\text{var}} = x_{\text{var}} \cdot \exp(s(x_{\text{fix}})) + t(x_{\text{fix}})$$

其中 $(s, t) = \text{MLP}(x_{\text{fix}})$，$x_{\text{fix}}$ 保持不变。

- **交替分割**：奇数层固定 $x_0$、变换 $x_1$；偶数层固定 $x_1$、变换 $x_0$。这保证了经过多层后两个维度都被充分变换。
- **数值稳定**：$s$ 被 clamp 到 $[-2, 2]$，防止 $\exp(s)$ 过大或过小。
- **初始化**：最后一层线性层的权重和偏置初始化为零，使得初始模型接近恒等映射（$s \approx 0, t \approx 0$）。

**逆变换**（解析可逆）：

$$x_{\text{var}} = (y_{\text{var}} - t(y_{\text{fix}})) \cdot \exp(-s(y_{\text{fix}}))$$

**对数行列式**：

$$\log\left|\det\frac{\partial y}{\partial x}\right| = s(x_{\text{fix}})$$

每层只有标量 $s$，因此整个流的 log-determinant 是各层 $s$ 的求和，计算开销极低。

### 1.3 前向与逆向

**前向** $z \to x$（采样用）：

$$x = f_\theta(z), \quad \log\det\frac{\partial f_\theta}{\partial z} = \sum_{\ell=1}^{L} s_\ell$$

**逆向** $x \to z$（密度求值用）：

$$z = f_\theta^{-1}(x), \quad \log\det\frac{\partial f_\theta^{-1}}{\partial x} = -\sum_{\ell=1}^{L} s_\ell$$

两个方向都是精确的，不需要任何近似。

---

## 2. 训练流程

### 2.1 第一阶段：Boltzmann 预训练

目标：让模型学会覆盖 $[-5, 5]^2$ 区域，为后续 RL 提供良好的初始化。

**数据生成**（`boltzmann.py`）：

从 Boltzmann 分布 $p(x) \propto \exp(-\beta \cdot \text{Rastrigin}(x))$ 中采样。当 $\beta = 0$ 时退化为 $[-5,5]^2$ 上的均匀分布；当 $\beta > 0$ 时样本偏向低 Rastrigin（高 reward）区域。采样通过拒绝采样实现：

1. 在 $[-5, 5]^2$ 上均匀提议
2. 以概率 $\exp(-\beta \cdot (\text{Rastrigin}(x) - \min))$ 接受

**训练方式**：最大似然估计（MLE），最小化负对数似然：

$$\mathcal{L}_{\text{pretrain}} = -\frac{1}{N}\sum_{i=1}^{N} \log \pi_\theta(x_i)$$

使用 Adam 优化器，学习率 $10^{-3}$，训练 300 epochs。

**`dataset_size` 参数**：
- `dataset_size=0`（默认）：每个 epoch 重新采样一批数据
- `dataset_size>0`：预采一个固定数据集，每个 epoch 从中随机抽取 mini-batch

### 2.2 第二阶段：PPO 强化学习微调

目标：最大化期望 reward $J(\theta) = \mathbb{E}_{x \sim \pi_\theta}[r(x)]$，其中 $r(x) = -\text{Rastrigin}(x)$。

**采样与评估**：

每次迭代：
1. 从当前策略 $\pi_\theta$ 采样 $N = 256$ 个样本
2. 计算 reward $r_i = r(x_i)$ 和旧 log-probability $\log\pi_{\theta_{\text{old}}}(x_i)$
3. 归一化 reward：$\hat{r}_i = (r_i - \bar{r}) / (\sigma_r + 10^{-8})$

**PPO 目标函数**：

$$\mathcal{L}_{\text{PPO}} = -\frac{1}{N}\sum_{i}\min\left(\rho_i \hat{r}_i, \;\text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{r}_i\right) + \lambda_{\text{KL}} \cdot \text{KL}(\pi_\theta \| \pi_{\text{base}})$$

其中：
- $\rho_i = \pi_\theta(x_i) / \pi_{\theta_{\text{old}}}(x_i) = \exp(\log\pi_\theta(x_i) - \log\pi_{\theta_{\text{old}}}(x_i))$
- $\epsilon = 0.2$（clipping 参数）
- $\lambda_{\text{KL}}$：KL 约束系数（实验中设为 0.0）
- $\pi_{\text{base}}$：预训练结束时冻结的模型副本

**KL 散度估计**：

$$\text{KL}(\pi_\theta \| \pi_{\text{base}}) \approx \frac{1}{N}\sum_i \left[\log\pi_\theta(x_i) - \log\pi_{\text{base}}(x_i)\right]$$

注意这里 $x_i$ 来自当前策略 $\pi_\theta$ 的采样，所以这是前向 KL 的蒙特卡洛估计。

**稳定性机制**：

1. **NaN 恢复**：当有效样本 < 32 或 loss 为 NaN 时，回滚到历史最优参数，学习率减半（下限为初始 lr 的 5%）
2. **发散恢复**：当 $\bar{r}_t < \bar{r}_{\text{best}} - \Delta_{\text{thresh}}$（默认 $\Delta_{\text{thresh}} = 5.0$）时，回滚到最优参数
3. **log-ratio clamp**：$\log\rho$ 截断在 $[-5, 5]$，防止极端 importance weight
4. **梯度裁剪**：全局梯度范数截断为 1.0

**PPO epochs**：每批数据上做 2 个 epoch 的优化步（不重新采样，复用同一批数据）。若某步的 KL > 50，提前停止该迭代的优化。

---

## 3. 精确计算 $\nabla_\theta \log\pi_\theta(x)$（Score Function）

这是本框架的核心优势：normalizing flow 允许**精确**计算 score function，不需要任何近似。

### 3.1 密度公式

由变量替换公式，flow 模型的对数密度为：

$$\log\pi_\theta(x) = \log p_z\!\left(f_\theta^{-1}(x)\right) + \log\left|\det\frac{\partial f_\theta^{-1}}{\partial x}\right|$$

其中：
- $f_\theta^{-1}$：flow 的逆变换（从数据空间到潜变量空间）
- $p_z = \mathcal{N}(0, I_2)$：基分布密度
- 行列式项来自变量替换的 Jacobian

展开后：

$$\log\pi_\theta(x) = -\frac{1}{2}\|z\|^2 - \frac{d}{2}\log(2\pi) + \sum_{\ell=1}^{L} \log\left|\det\frac{\partial h_\ell^{-1}}{\partial h_{\ell-1}^{-1}}\right|$$

其中 $z = h_L^{-1} \circ \cdots \circ h_1^{-1}(x)$，每个 $h_\ell^{-1}$ 是一个耦合层的逆变换。

### 3.2 为什么是精确的

对于仿射耦合层，每个因子都有解析表达：

1. **逆变换 $z = f_\theta^{-1}(x)$**：每层解析可逆
   $$x_{\text{var}}^{(\ell-1)} = \left(x_{\text{var}}^{(\ell)} - t_\ell(x_{\text{fix}}^{(\ell)})\right) \cdot \exp\left(-s_\ell(x_{\text{fix}}^{(\ell)})\right)$$

2. **对数行列式**：每层贡献 $\pm s_\ell(x_{\text{fix}})$，是网络输出的直接函数

3. **基分布密度 $\log p_z(z)$**：高斯密度的解析式

因此 $\log\pi_\theta(x)$ 是 $\theta$（网络权重）的**可微函数**，整个计算图由标准的神经网络操作（线性层、ReLU、exp、求和）组成。

### 3.3 通过自动微分计算 Score

Score 的计算分两步：

**步骤 1**：前向计算 $\log\pi_\theta(x)$

```python
# x 是给定的数据点（detached，不需要对 x 求梯度）
z, log_det_inv = model.inverse(x)        # 逆变换 + log|det|
log_pz = -0.5 * (z² + log(2π)).sum(-1)   # 基分布密度
log_prob = log_pz + log_det_inv           # 变量替换公式
```

**步骤 2**：反向传播得 $\nabla_\theta \log\pi_\theta(x)$

```python
model.zero_grad()
log_prob.backward()    # PyTorch 自动微分
score = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
```

`backward()` 沿计算图反向传播：
$$\frac{\partial \log\pi_\theta}{\partial \theta_k} = \frac{\partial \log p_z(z)}{\partial z} \cdot \frac{\partial z}{\partial \theta_k} + \frac{\partial}{\partial \theta_k}\sum_\ell \log|\det J_\ell^{-1}|$$

这里每一项都是精确的——没有蒙特卡洛近似，没有方差缩减技巧，只有标准的链式法则。

### 3.4 与其他方法的对比

| 方法 | 是否精确 | 适用模型 |
|------|----------|----------|
| **Normalizing flow + autodiff**（本文） | 精确 | Flow（需要解析逆和行列式） |
| REINFORCE / score function estimator | 近似（高方差） | 任意可采样模型 |
| Reparameterization trick | 精确（对 $\theta$） | 需要可微采样路径 |
| Stein score estimator | 近似 | 无需密度，只需采样 |

Flow 模型的关键优势在于：**逆变换和 Jacobian 行列式都有解析形式**，使得 $\log\pi_\theta(x)$ 是 $\theta$ 的显式可微函数。这意味着 policy gradient $\nabla_\theta J = \mathbb{E}_{x \sim \pi_\theta}[r(x) \cdot \nabla_\theta\log\pi_\theta(x)]$ 中的 score $\nabla_\theta\log\pi_\theta(x)$ 完全精确，唯一的近似来源是期望的蒙特卡洛采样。

### 3.5 实际代码中的使用

在 `fim.py` 的 `_compute_score_projections` 中，score 被投影到指定方向上：

```python
for _ in range(n_samples):
    x, _ = model.sample(1)          # 采一个样本
    x = x.detach()                   # 切断采样路径的梯度
    r_val = reward(x).item()         # 配对的 reward
    log_prob = model.log_prob(x)     # 精确密度
    model.zero_grad()
    log_prob.backward()              # 精确 score
    grad = concat([p.grad for p in model.parameters()])
    for d in directions:
        projections.append(grad @ d) # score 在方向 d 上的投影 = g · d
```

这些投影用于计算：
- **FIM 对角投影**：$F_{kk} = \mathbb{E}[(g \cdot d_k)^2]$
- **KL 弧长**：$s_{\text{KL}} = \sqrt{\frac{1}{2}\mathbb{E}[(g \cdot \Delta\theta)^2]}$
- **$C_v$ 统计量**：$C_v = \frac{(\mathbb{E}[r \cdot (g \cdot v)])^2}{\mathbb{E}[(g \cdot v)^2]}$

其中 $g = \nabla_\theta\log\pi_\theta(x)$ 是精确的 score vector。

---

## 4. 环境：2D Rastrigin

$$\text{Rastrigin}(x) = A \cdot d + \sum_{i=1}^{d}\left(x_i^2 - A\cos(2\pi x_i)\right), \quad A = 10, \; d = 2$$

$$r(x) = -\text{Rastrigin}(x)$$

- 全局最优：$x^* = (0, 0)$，$r^* = 0$
- 多个局部最优（约每隔 1 个单位一个）
- 在 $[-5, 5]^2$ 上 reward 范围约 $[-80, 0]$

---

## 5. 实验配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 耦合类型 | affine (RealNVP) | 仿射变换，训练更稳定 |
| 层数 | 8 | 交替固定 $x_0$ 和 $x_1$ |
| 隐藏维度 | 64 | 每层 MLP: 1→64→64→2 |
| 总参数 | ~35K | |
| 预训练 epochs | 300 | MLE on Boltzmann samples |
| 预训练 lr | $10^{-3}$ | Adam |
| PPO 迭代 | 200 | |
| PPO batch | 256 | 每次迭代采样数 |
| PPO epochs | 2 | 每批数据优化步数 |
| PPO lr | $3 \times 10^{-4}$ | Adam |
| Clip $\epsilon$ | 0.2 | PPO clipping |
| KL 系数 | 0.0 | 不加 KL 约束 |
