# FIM Landscape 可视化实现报告

## 目录
1. [背景与动机](#1-背景与动机)
2. [从欧氏坐标到局域 FIM 坐标的变换](#2-从欧氏坐标到局域-fim-坐标的变换)
3. [新坐标下 Landscape 值的确定](#3-新坐标下-landscape-值的确定)
4. [完整流水线总结](#4-完整流水线总结)

---

## 1. 背景与动机

### 1.1 问题设定

我们有一个参数化的生成模型 $\pi_\theta(x)$（Normalizing Flow），通过 PPO 训练使其在 2D Rastrigin 目标函数上最大化期望回报：

$$J(\theta) = \mathbb{E}_{x \sim \pi_\theta}[r(x)]$$

参数 $\theta \in \mathbb{R}^D$（$D \approx 47000$），我们无法直接可视化高维参数空间。需要将其投影到 2D 平面上进行分析。

### 1.2 两套坐标系

在我们的可视化中涉及**两套不同的坐标系**：

| 坐标系 | 含义 | 轴标签 |
|--------|------|--------|
| **欧氏 PCA 坐标** $(\alpha, \beta)$ | 参数空间中沿 PCA 方向的欧氏距离 | $d_1$ (start→end), $d_2$ (max ⊥ variance) |
| **FIM/KL 坐标** $(\alpha_\mathrm{KL}, \beta_\mathrm{KL})$ | 信息几何意义下的 KL 散度距离 | $\sqrt{\mathrm{KL}}$ along PC1/PC2 |

**核心区别**：欧氏坐标下等距的网格线，映射到 KL 坐标后变成非均匀的——FIM 大的区域被"拉伸"，FIM 小的区域被"压缩"。

### 1.3 为什么需要 FIM 坐标？

在欧氏 $\theta$ 空间中，参数走一步 $\Delta\theta$ 所对应的**分布变化量**因位置不同而差异巨大。FIM 定义了参数空间上的黎曼度量：

$$d_\mathrm{KL}(\pi_\theta \| \pi_{\theta+\delta}) \approx \frac{1}{2} \delta^\top F(\theta) \, \delta$$

在 FIM 坐标下，**等距意味着等量的分布变化**，从而更真实地反映优化的难度和效率。

---

## 2. 从欧氏坐标到局域 FIM 坐标的变换

这是整个实现中最核心的部分，分为三个步骤：
1. 构建 PCA 基底（高维→2D 投影）
2. 在 2D 平面上计算 FIM 场
3. 通过路径积分将欧氏坐标变换为 KL 坐标

### 2.1 第一步：构建 PCA 基底（高维→2D 投影）

**目的**：从 $D$ 维参数空间中提取一个 2D 平面，使得 PPO 轨迹的投影信息量最大。

**对应代码**：`fim.py:17-78`，函数 `compute_pca_directions()`

#### 算法

给定 PPO 训练过程中保存的参数快照 $\{\theta_0, \theta_1, \ldots, \theta_T\}$：

**第 1 步：确定主方向 $d_1$**
$$d_1 = \frac{\theta_T - \theta_0}{\|\theta_T - \theta_0\|}$$

即从起点到终点的归一化方向。选择这个方向（而非标准 PCA 第一主成分）是为了**保证起止点严格落在可视化平面上**。

```python
# fim.py:33-35
delta = traj_params[-1] - traj_params[0]
d1 = delta / delta.norm()
```

**第 2 步：确定原点**
$$\text{center} = \theta_0$$

以训练起点作为原点，这样起点的 2D 坐标恰好是 $(0, 0)$。

```python
# fim.py:38
center = traj_params[0].clone()
```

**第 3 步：计算轨迹在 $d_1$ 上的投影**
$$\alpha_t = (\theta_t - \theta_0) \cdot d_1$$

```python
# fim.py:39-40
centered = traj_params - center.unsqueeze(0)
proj_alpha = (centered @ d1).numpy()
```

**第 4 步：计算正交残差并提取 $d_2$**

将轨迹减去 $d_1$ 分量，得到正交残差矩阵，然后对残差做 PCA 取第一主成分：

$$\text{residual}_t = (\theta_t - \theta_0) - \alpha_t \cdot d_1$$
$$d_2 = \text{PCA}_1(\{\text{residual}_t\}_{t=0}^T)$$

```python
# fim.py:43-55
residuals = centered - torch.from_numpy(proj_alpha).unsqueeze(1) * d1.unsqueeze(0)
pca = PCA(n_components=1)
pca.fit(residuals_np)
d2 = torch.tensor(pca.components_[0], dtype=torch.float32)
```

**第 5 步：Gram-Schmidt 正交化确保数值精度**

```python
# fim.py:64-66
d2 = d2 - (d2 @ d1) * d1
d2 = d2 / d2.norm()
```

**第 6 步：计算所有轨迹点的 2D 坐标**

$$\alpha_t = (\theta_t - \theta_0) \cdot d_1, \quad \beta_t = (\theta_t - \theta_0) \cdot d_2$$

```python
# fim.py:69-70
traj_alpha = proj_alpha
traj_beta = (centered @ d2).numpy()
```

#### 几何含义

经过这个投影后，任意参数空间点 $\theta$ 可以用其 2D 坐标 $(\alpha, \beta)$ 近似表示：

$$\theta \approx \theta_0 + \alpha \cdot d_1 + \beta \cdot d_2$$

这是一个 **仿射嵌入**：$(\alpha, \beta)$ 空间是 $\mathbb{R}^D$ 中过 $\theta_0$、由 $d_1, d_2$ 张成的 2D 平面。

### 2.2 第二步：在 2D 平面上计算 FIM 场

**目的**：在每个欧氏网格点 $(\alpha_i, \beta_j)$ 处计算 Fisher 信息矩阵沿 $d_1, d_2$ 方向的投影。

**对应代码**：`fim.py:162-167`（单点）和 `fim.py:259-277`（全网格）

#### FIM 投影的数学定义

Fisher 信息矩阵在参数 $\theta$ 处的完整定义为：

$$F(\theta) = \mathbb{E}_{x \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(x) \, \nabla_\theta \log \pi_\theta(x)^\top\right]$$

这是一个 $D \times D$ 矩阵，无法存储。我们只计算其在 $d_k$ 方向上的投影（即对角分量）：

$$F_{kk}(\theta) = d_k^\top F(\theta) \, d_k = \mathbb{E}_{x \sim \pi_\theta}\left[(\nabla_\theta \log \pi_\theta(x) \cdot d_k)^2\right]$$

即 **score 在方向 $d_k$ 上投影的平方的期望**。

#### 单点计算流程

对每个网格点 $\theta = \theta_0 + \alpha \cdot d_1 + \beta \cdot d_2$：

1. **设置参数**：`model.set_flat_params(θ)`
2. **采样**：$x^{(i)} \sim \pi_\theta$，$i = 1, \ldots, N$
3. **计算 score**：对每个样本 $x^{(i)}$，
   - 前向传播得到 $\log \pi_\theta(x^{(i)})$
   - 反向传播得到 $\nabla_\theta \log \pi_\theta(x^{(i)})$（这是 $D$ 维梯度向量）
4. **投影到方向**：$g_k^{(i)} = \nabla_\theta \log \pi_\theta(x^{(i)}) \cdot d_k$
5. **估计 FIM 投影**：$\hat{F}_{kk} = \text{trimmed\_mean}(\{(g_k^{(i)})^2\})$

```python
# fim.py:110-132（score 计算与投影）
for _ in range(n_samples):
    x, _ = model.sample(1)
    x = x.detach()
    log_prob = model.log_prob(x)
    model.zero_grad()
    log_prob.backward()
    grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    for k, d in enumerate(directions):
        projections[k].append((grad @ d).item())
```

#### 鲁棒估计：Trimmed Mean + MAD

直接计算 $\mathbb{E}[g^2]$ 容易被异常值污染。代码使用两步鲁棒估计：

```python
# fim.py:137-159
def _trimmed_mean_square(vals, trim_ratio=0.1):
    vals_sq = np.array([v**2 for v in vals])
    # 第一步：MAD 异常值过滤（5σ 规则）
    median = np.median(vals_sq)
    mad = np.median(np.abs(vals_sq - median))
    mask = np.abs(vals_sq - median) <= 5 * mad
    vals_sq = vals_sq[mask]
    # 第二步：Trimmed mean（去掉上下 10%）
    n_trim = int(n * trim_ratio)
    vals_sorted = np.sort(vals_sq)
    return vals_sorted[n_trim:-n_trim].mean()
```

#### 全网格 FIM 场

对 2D 欧氏网格的每个点重复上述计算：

$$F_{11}[j, i] = F_{11}(\theta_0 + \alpha_i \cdot d_1 + \beta_j \cdot d_2)$$
$$F_{22}[j, i] = F_{22}(\theta_0 + \alpha_i \cdot d_1 + \beta_j \cdot d_2)$$

得到两个 $(n_\beta \times n_\alpha)$ 的矩阵，描述了整个 2D 平面上 FIM 的空间分布。

```python
# fim.py:259-277
for i, alpha in enumerate(alphas):
    for j, beta in enumerate(betas):
        theta = center + alpha * d1 + beta * d2
        F11[j, i], F22[j, i] = compute_fim_diagonal(
            model, theta, d1, d2, n_samples, trim_ratio)
```

### 2.3 第三步：从欧氏坐标到 KL 坐标的变换（核心）

**目的**：利用 FIM 场将欧氏坐标 $(\alpha, \beta)$ 变换为 KL 坐标 $(\alpha_\mathrm{KL}, \beta_\mathrm{KL})$。

**对应代码**：`fim.py:280-314`，函数 `build_kl_coords_2d()`

#### 数学原理

FIM 定义了参数空间上的黎曼度量。在 $d_k$ 方向上，无穷小的 KL 距离为：

$$ds_{\mathrm{KL},k} = \sqrt{\frac{1}{2} F_{kk}(\theta)} \cdot |d\alpha_k|$$

其中 $\frac{1}{2}$ 因子来源于 KL 散度与 Fisher 信息的关系：$D_\mathrm{KL} \approx \frac{1}{2} \delta^\top F \delta$。

从某个参考点（网格中心）出发，沿坐标轴积分得到 KL 坐标：

$$\alpha_\mathrm{KL}(\alpha) = \int_0^\alpha \sqrt{\frac{1}{2} F_{11}(\alpha', \beta_0)} \, d\alpha'$$

$$\beta_\mathrm{KL}(\beta) = \int_0^\beta \sqrt{\frac{1}{2} F_{22}(\alpha_0, \beta')} \, d\beta'$$

#### 离散化实现

使用**梯形法则**从网格中心向两侧逐步积分：

**沿 $\alpha$ 方向（固定 $\beta_j$，逐行积分）**：

```
向右积分（从中心到右边界）：
  kl_α[j, i+1] = kl_α[j, i] + √(½ · F̄₁₁) · |Δα|
  其中 F̄₁₁ = (F₁₁[j,i] + F₁₁[j,i+1]) / 2  （相邻两点的均值）

向左积分（从中心到左边界）：
  kl_α[j, i-1] = kl_α[j, i] - √(½ · F̄₁₁) · |Δα|
```

```python
# fim.py:294-302
for j in range(nb):
    # 向右
    for i in range(ca, na - 1):
        f_mid = max(0.5 * (F11[j, i] + F11[j, i + 1]), fim_floor)
        kl_alpha[j, i + 1] = kl_alpha[j, i] + np.sqrt(0.5 * f_mid) * abs(dalpha[i])
    # 向左
    for i in range(ca, 0, -1):
        f_mid = max(0.5 * (F11[j, i] + F11[j, i - 1]), fim_floor)
        kl_alpha[j, i - 1] = kl_alpha[j, i] - np.sqrt(0.5 * f_mid) * abs(dalpha[i - 1])
```

**沿 $\beta$ 方向（固定 $\alpha_i$，逐列积分）**：

```python
# fim.py:304-312
for i in range(na):
    for j in range(cb, nb - 1):
        f_mid = max(0.5 * (F22[j, i] + F22[j + 1, i]), fim_floor)
        kl_beta[j + 1, i] = kl_beta[j, i] + np.sqrt(0.5 * f_mid) * abs(dbeta[j])
    for j in range(cb, 0, -1):
        f_mid = max(0.5 * (F22[j, i] + F22[j - 1, i]), fim_floor)
        kl_beta[j - 1, i] = kl_beta[j, i] - np.sqrt(0.5 * f_mid) * abs(dbeta[j - 1])
```

#### 关键近似与假设

1. **对角近似**：假设 $F_{12} \approx 0$，即 $d_1, d_2$ 方向在 Fisher 度量下也近似正交。这使得 $\alpha$ 和 $\beta$ 方向的积分可以独立进行。

2. **FIM 下界**：`fim_floor = 1e-3`，防止 FIM 接近零时积分不稳定。

3. **梯形法则**：用相邻两点的 FIM 均值作为中间值，即 $F_\mathrm{mid} = \frac{1}{2}(F_k[i] + F_k[i+1])$，提高数值精度。

#### 变换的几何直觉

- **FIM 大的区域**：$\sqrt{F_{kk}}$ 大，单位欧氏距离映射到更长的 KL 距离 → 该区域在 KL 坐标下被**拉伸**
- **FIM 小的区域**：$\sqrt{F_{kk}}$ 小，单位欧氏距离映射到较短的 KL 距离 → 该区域在 KL 坐标下被**压缩**

物理意义：FIM 大 = 分布对参数变化敏感 = 信息密度高，因此在信息几何坐标下占更多"空间"。

### 2.4 轨迹点的 KL 坐标

PPO 轨迹点 $(\alpha_t, \beta_t)$ 也需要从欧氏坐标映射到 KL 坐标。这通过**一维插值**完成：

```python
# fim.py:394-395
traj_alpha_kl = np.interp(traj_alpha, alphas, kl_alpha_1d)
traj_beta_kl  = np.interp(traj_beta,  betas,  kl_beta_1d)
```

其中 `kl_alpha_1d` 和 `kl_beta_1d` 是沿各轴的平均 KL 坐标（对所有行/列取平均后积分）：

```python
# fim.py:366-392
F11_avg = F11.mean(axis=0)   # 对 β 方向平均，得到 F11 关于 α 的 1D 剖面
F22_avg = F22.mean(axis=1)   # 对 α 方向平均，得到 F22 关于 β 的 1D 剖面

# 从原点（α=0 处）开始积分
kl_alpha_1d = np.zeros(n_grid)
for i in range(ca, n_grid - 1):
    f_mid = max(0.5 * (F11_avg[i] + F11_avg[i + 1]), fim_floor)
    kl_alpha_1d[i + 1] = kl_alpha_1d[i] + np.sqrt(0.5 * f_mid) * abs(dalpha[i])
```

---

## 3. 新坐标下 Landscape 值的确定

### 3.1 问题描述

我们现在有了 KL 坐标系 $(\alpha_\mathrm{KL}, \beta_\mathrm{KL})$，但需要在这个坐标系下画出 $J(\theta)$ 的等高线图。问题在于：**在 KL 坐标下等间距的网格点，对应到欧氏空间中是非等间距的**，因此不能直接复用欧氏网格上的 $J$ 值。

### 3.2 方案：构建均匀 KL 网格 + 反向映射 + 重新采样

代码中使用的策略（`fim.py:397-411`）分为三步：

#### 步骤 1：在 KL 坐标下建立均匀网格

```python
# fim.py:397-398
kl_alphas_uniform = np.linspace(kl_alpha_1d.min(), kl_alpha_1d.max(), n_grid)
kl_betas_uniform  = np.linspace(kl_beta_1d.min(),  kl_beta_1d.max(),  n_grid)
```

这给出了 $n_\mathrm{grid} \times n_\mathrm{grid}$ 个在 KL 空间中**等间距**的点。

#### 步骤 2：反向映射——从 KL 坐标找到对应的欧氏坐标

通过已知的映射关系 $\alpha \mapsto \alpha_\mathrm{KL}(\alpha)$（单调递增函数），用插值求逆映射：

$$\alpha = (\alpha_\mathrm{KL})^{-1}(\alpha_\mathrm{KL,uniform})$$

```python
# fim.py:400-401
alpha_from_kl = np.interp(kl_alphas_uniform, kl_alpha_1d, alphas)
beta_from_kl  = np.interp(kl_betas_uniform,  kl_beta_1d,  betas)
```

这一步是整个方案的关键——`np.interp` 将均匀 KL 网格值作为查询点，在已有的 `(kl_alpha_1d[i], alphas[i])` 对应关系中插值，得到每个 KL 网格点对应的欧氏坐标值。

**为什么可以用 `np.interp`？** 因为 $\alpha \mapsto \alpha_\mathrm{KL}$ 是严格单调递增的（$\sqrt{F_{11}} > 0$），所以它的逆函数存在且连续。

#### 步骤 3：在反向映射得到的欧氏点处重新计算 $J(\theta)$

```python
# fim.py:403-411
J_kl_grid = np.zeros((n_grid, n_grid))
for i, kl_a in enumerate(kl_alphas_uniform):
    a = alpha_from_kl[i]                        # 对应的欧氏 α
    for j, kl_b in enumerate(kl_betas_uniform):
        b = beta_from_kl[j]                      # 对应的欧氏 β
        theta = center + a * d1 + b * d2         # 高维参数
        J_kl_grid[j, i] = evaluate_J(model, theta, n_eval_samples)  # 重新采样计算 J
```

对每个均匀 KL 网格点 $(\alpha_\mathrm{KL}^{(i)}, \beta_\mathrm{KL}^{(j)})$：
1. 反向映射找到欧氏坐标 $(\alpha^{(i)}, \beta^{(j)})$
2. 重建高维参数 $\theta = \theta_0 + \alpha^{(i)} d_1 + \beta^{(j)} d_2$
3. 从 $\pi_\theta$ 中采样 $N$ 个点，计算 $J(\theta) = \frac{1}{N}\sum r(x^{(k)})$

#### 为什么要重新采样而不是插值已有的 J 值？

虽然可以用双线性插值从 `J_grid`（欧氏网格上的 $J$ 值）插值得到非网格点的值，代码选择了**重新采样**的方式。原因：

1. **精度**：插值会引入误差，特别是当 landscape 有尖锐结构时
2. **一致性**：每个 KL 网格点的 $J$ 值都是用相同的采样数独立估计的
3. **可靠性**：避免了双线性插值在网格边界附近的外推问题

代价是计算量翻倍（需要做两次 $n_\mathrm{grid}^2$ 的 $J$ 扫描），但换来了更可靠的结果。

### 3.3 图示：变换前后的对比

最终绘图时（`visualize.py:77-115`），三个面板展示的是：

| 面板 | x 轴 | y 轴 | 颜色 | 网格 |
|------|-------|------|------|------|
| Panel 1 | $x_1$ | $x_2$ | $r(x)$ | 原始空间均匀 |
| Panel 2 | $\alpha$（欧氏） | $\beta$（欧氏） | $J(\theta)$ | 欧氏空间均匀 |
| Panel 3 | $\alpha_\mathrm{KL}$ | $\beta_\mathrm{KL}$ | $J(\theta)$ | **KL 空间均匀** |

在 Panel 3 中，还叠加了从欧氏网格变换过来的**非均匀白色网格线**：

```python
# visualize.py:89-100
kl_alpha_2d = theta_landscape['kl_alpha_2d']   # 2D 完整变换
kl_beta_2d  = theta_landscape['kl_beta_2d']
for j in range(0, n_grid, grid_step):
    axes[2].plot(kl_alpha_2d[j, :], kl_beta_2d[j, :], 'w-', linewidth=0.3)
for i in range(0, n_grid, grid_step):
    axes[2].plot(kl_alpha_2d[:, i], kl_beta_2d[:, i], 'w-', linewidth=0.3)
```

这些白色网格线在欧氏空间中是等间距的直线，但在 KL 坐标下变成了不均匀的曲线——直观展示了 FIM 引起的度量扭曲。

### 3.4 两套 KL 坐标的区别

代码中实际有**两套** KL 坐标，用途不同：

| 变量 | 维度 | 来源 | 用途 |
|------|------|------|------|
| `kl_alpha_2d, kl_beta_2d` | $(n_\beta, n_\alpha)$ | `build_kl_coords_2d()`：逐行逐列独立积分 | 画非均匀网格线（白色曲线叠加层） |
| `kl_alpha_1d, kl_beta_1d` | $(n_\mathrm{grid},)$ | 对 FIM 取行/列均值后积分 | 构建均匀 KL 网格、轨迹坐标变换 |

**为什么不直接用 2D 版本？**

2D 版本 `build_kl_coords_2d` 为每个 $(j, i)$ 单独积分，结果是一个**非规则网格**——同一行的 $\alpha_\mathrm{KL}$ 可能随 $\beta$ 变化。这无法直接用于 `contourf`（需要矩形网格）。

1D 版本取了行/列均值，将 2D 问题降维为两个独立的 1D 问题，产生的 KL 坐标是**可分离的**：$\alpha_\mathrm{KL}$ 只依赖 $\alpha$，$\beta_\mathrm{KL}$ 只依赖 $\beta$，从而构成规则的矩形网格。

---

## 4. 完整流水线总结

### 4.1 数据流

```
PPO 训练
  ↓ param_snapshots = [(iter₀, θ₀), (iter₁, θ₁), ..., (iter_T, θ_T)]
  
compute_pca_directions()
  ↓ d1, d2, center, traj_alpha, traj_beta
  
compute_theta_landscape()                        [可选，--scan 时运行]
  ├── 欧氏网格扫描 → J_grid[j,i]               [在 (αᵢ, βⱼ) 处计算 J]
  ├── compute_fim_field() → F11[j,i], F22[j,i]  [在 (αᵢ, βⱼ) 处计算 FIM]
  ├── build_kl_coords_2d() → kl_alpha_2d, kl_beta_2d  [2D 非均匀 KL 网格]
  ├── 1D 平均 FIM 积分 → kl_alpha_1d, kl_beta_1d       [1D 可分离 KL 坐标]
  ├── 反向映射 → alpha_from_kl, beta_from_kl           [KL→欧氏逆变换]
  └── 均匀 KL 网格扫描 → J_kl_grid[j,i]               [在逆映射点处重新计算 J]

可视化
  ├── fig1: 2D 等高线对比（欧氏 vs KL）
  ├── fig4: 3D 曲面对比
  └── fig11: 欧氏 J(θ) + FIM 椭圆叠加
```

### 4.2 关键公式速查

| 量 | 公式 | 代码位置 |
|----|------|----------|
| PCA 投影坐标 | $\alpha = (\theta - \theta_0) \cdot d_1$ | `fim.py:40` |
| FIM 投影 | $F_{kk} = \mathbb{E}[(\nabla_\theta \log\pi \cdot d_k)^2]$ | `fim.py:162-167` |
| KL 距离微元 | $ds_k = \sqrt{\frac{1}{2}F_{kk}} \cdot |d\alpha_k|$ | `fim.py:297-298` |
| KL 坐标积分 | $\alpha_\mathrm{KL} = \sum \sqrt{\frac{1}{2}\bar{F}_{11}} \cdot \|\Delta\alpha\|$ | `fim.py:379-384` |
| 逆映射 | $\alpha = \text{interp}^{-1}(\alpha_\mathrm{KL})$ | `fim.py:400` |
| KL 弧长 | $s_\mathrm{KL}(t) = \sqrt{\frac{1}{2}\mathbb{E}[(\nabla\log\pi \cdot \Delta\theta_t)^2]}$ | `fim.py:220` |
| 度量扭曲比 | $\rho(t) = s_\mathrm{KL}(t) / s_\mathrm{Euclid}(t)$ | `fim.py:237` |

### 4.3 设计决策总结

1. **为什么用投影而非完整 FIM？** 完整 $D \times D$ FIM 无法存储或求逆。投影到 2D 子空间的对角分量足以捕捉关键的各向异性信息。

2. **为什么假设 $F_{12} \approx 0$？** $d_1, d_2$ 是 PCA 方向，在欧氏空间中已经正交。虽然 Fisher 度量下的正交性不保证，但对角近似使得 $\alpha, \beta$ 方向可以独立积分，大大简化实现。

3. **为什么重新采样而非插值？** 插值在 landscape 有尖锐结构时不精确。重新采样保证每个 KL 网格点的 $J$ 值独立且一致。

4. **为什么用 trimmed mean 而非 plain mean？** Score 的平方值容易出现极端异常值（数值不稳定、rare extreme scores），trimmed mean + MAD 过滤提供了更鲁棒的估计。
