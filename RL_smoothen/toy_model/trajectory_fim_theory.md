# Fisher 信息度量与参数空间可视化：理论分析

## 1. FIM 的计算：二阶量，一阶算法

### 1.1 当前做法

现在的 `fim.py` 做了这件事：

1. 收集 PPO 训练过程中的参数快照 $\{\theta_0, \theta_1, \dots, \theta_T\}$
2. 对这条轨迹做 PCA，取前两个主方向 $d_1, d_2$
3. 以轨迹中点 $\theta_{\text{mid}}$ 为中心，在 $(\alpha, \beta)$ 网格上计算：

$$\theta(\alpha, \beta) = \theta_{\text{mid}} + \alpha \cdot d_1 + \beta \cdot d_2$$

两者等价来自 score 零均值性质：$\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$，对此再求一次 $\nabla_\theta$ 展开即得。

**实践意义**：一次 `log_prob(x).backward()` 就给出完整的 $\nabla_\theta \log \pi_\theta(x) \in \mathbb{R}^D$，对任意 $(\theta, x)$ 对都成立。成本与训练时每步相同。因此 $F$ 虽然是 $D \times D$ 矩阵（$D \sim 47\text{K}$），无法存储，但其中**任意一个矩阵元**和**任意方向的投影** $d^\top F\, d'$ 都可以轻松计算。

---

## 2. PCA 投影的含义与改进

### 2.1 原始做法的问题

原始做法对轨迹做标准 PCA，取前两个主成分 $d_1, d_2$。但标准 PCA 最大化方差，**不保证起止点在 2D 平面上**。轨迹的起点 $\theta_0$ 和终点 $\theta_T$ 只是被投影到平面上，存在残差。图上画的轨迹不经过真实的起终点。

### 2.2 改进：以起止方向为主轴

$$d_1 = \frac{\theta_T - \theta_0}{\|\theta_T - \theta_0\|}$$

$d_2$ = 轨迹在 $d_1$ 正交补空间中的最大方差方向（对残差做 PCA）。center = $\theta_0$。

这保证：
- $\theta_0$ 严格位于原点 $(0, 0)$
- $\theta_T$ 严格位于 $\alpha$ 轴上 $(\|\theta_T - \theta_0\|, 0)$
- $d_2$ 捕捉轨迹的最大横向偏移

中间的轨迹点仍可能有残差（偏离平面），但优化的主方向被忠实表示。

$$F_{11} = d_1^\top F\, d_1 = \mathbb{E}\left[(\nabla_\theta \log \pi \cdot d_1)^2\right]$$

要从 $(\theta_1, \theta_2)$ 空间变换到 KL 坐标，需要对 $d_1^\top F(\theta)\, d_1$ 做积分——但这要求在路径上每一点都知道 $F(\theta)$。

---

## 3. 2D 网格扫描为什么失败

### 3.1 OOD 问题的精确诊断

在网格点 $\theta(\alpha,\beta) = \theta_{\text{mid}} + \alpha d_1 + \beta d_2$ 上计算 FIM 时，当前代码从 $\pi_\theta$ **采样**：

```python
x, _ = model.sample(1)       # z ~ N(0,I), x = f_θ(z)
log_prob = model.log_prob(x)  # 再走一遍 inverse
log_prob.backward()           # 拿 score
```

当 $\theta$ 远离轨迹（OOD），$f_\theta$ 把正常的 $z$ 映射到极端的 $x$，极端 $x$ 的 score 方差爆炸。

### 3.2 修正方案：用固定参考点

$\nabla_\theta \log \pi_\theta(x)$ 对**任意** $(\theta, x)$ 都能 backward 算出，不依赖从 $\pi_\theta$ 采样。如果用固定的参考点集合 $\{x_i\} \subset [-5,5]^2$，避开了 OOD 采样，计算的是：

$$F(\theta) = \mathbb{E}_{x \sim \pi_\theta}\left[ \nabla_\theta \log \pi_\theta(x) \; \nabla_\theta \log \pi_\theta(x)^\top \right]$$

$F(\theta)$ 是一个 $D \times D$ 的半正定矩阵（$D$ = 参数维度），它定义了参数空间上的一个 **黎曼度量**（Riemannian metric）。

#### FIM 作为 KL 散度的局部度量

两个相邻参数配置之间的 KL 散度可以二阶展开：

$$D_{\text{KL}}(\pi_\theta \| \pi_{\theta + \delta}) \approx \frac{1}{2} \delta^\top F(\theta) \, \delta$$

这意味着 $F(\theta)$ 告诉我们：**在 $\theta$ 附近，沿不同方向走一小步，会导致多大的分布变化**。$F$ 大的方向，分布变化剧烈（"陡峭"）；$F$ 小的方向，分布几乎不变（"平坦"）。

#### 沿轨迹的弧长

在参数空间中，连接 $\theta_k$ 和 $\theta_{k+1}$ 的 **KL 弧长**（信息几何弧长）为：

$$s_{\text{KL}}(k \to k+1) = \sqrt{ \frac{1}{2}\, \Delta\theta_k^\top \, F(\theta_k) \, \Delta\theta_k }$$

其中 $\Delta\theta_k = \theta_{k+1} - \theta_k$。

对比 **欧氏弧长**：

$$s_{\text{Euclid}}(k \to k+1) = \|\Delta\theta_k\|_2$$

两者的比值反映了 **局部几何的扭曲程度**：

$$\rho_k = \frac{s_{\text{KL}}(k \to k+1)}{s_{\text{Euclid}}(k \to k+1)}$$

- $\rho_k$ 大：该步在分布空间走了"很远"，即使在参数空间看起来步长不大
- $\rho_k$ 小：该步在参数空间走了一段距离，但分布几乎没变

### 2.3 投影 FIM：从 $D \times D$ 降到有意义的维度

完整的 $F(\theta)$ 是 $D \times D$ 矩阵（$D = 35\text{K}$ 或 $47\text{K}$），不可能直接计算和存储。我们需要 **投影**。

#### 方法 1：投影到 PCA 方向

保留当前代码中的 PCA 方向 $d_1, d_2$，但 **只在轨迹点上** 计算投影 FIM：

$$F_{kk}^{(t)} = \mathbb{E}_{x \sim \pi_{\theta_t}} \left[ \left( \nabla_\theta \log \pi_{\theta_t}(x) \cdot d_k \right)^2 \right], \quad k = 1, 2$$

这给出两条曲线 $F_{11}(t)$ 和 $F_{22}(t)$，描述 **沿 PCA 方向的局部曲率如何随训练演化**。

#### 方法 2：投影到步进方向

更自然的选择是投影到 **实际的更新方向** $\hat{v}_t = \Delta\theta_t / \|\Delta\theta_t\|$：

$$F_{vv}^{(t)} = \mathbb{E}_{x \sim \pi_{\theta_t}} \left[ \left( \nabla_\theta \log \pi_{\theta_t}(x) \cdot \hat{v}_t \right)^2 \right]$$

以及 **垂直于更新方向** 的分量（从更新向量中减去 PCA 第一主成分后归一化），反映横向曲率。

#### 方法 3：随机投影取迹的估计（Hutchinson 估计）

如果需要 $\text{tr}(F)$（FIM 的迹，所有方向曲率之和），可以用 Hutchinson trick：

$$\text{tr}(F) = \mathbb{E}_v \left[ v^\top F v \right] \approx \frac{1}{M} \sum_{m=1}^{M} \left( \nabla_\theta \log \pi_\theta(x_m) \cdot v_m \right)^2$$

其中 $v_m \sim \mathcal{N}(0, I_D)$，$x_m \sim \pi_\theta$。但对于你的场景，投影到 PCA 方向已经足够。

### 2.4 累积 KL 距离

沿轨迹的 **累积 KL 距离** 定义为：

$$S_{\text{KL}}(t) = \sum_{k=0}^{t-1} s_{\text{KL}}(k \to k+1) = \sum_{k=0}^{t-1} \sqrt{ \frac{1}{2}\, \Delta\theta_k^\top \, F(\theta_k) \, \Delta\theta_k }$$

由于完整的 $\Delta\theta_k^\top F(\theta_k) \Delta\theta_k$ 需要矩阵-向量乘，我们可以用采样估计：

$$\Delta\theta_k^\top F(\theta_k) \Delta\theta_k = \mathbb{E}_{x \sim \pi_{\theta_k}} \left[ \left( \nabla_\theta \log \pi_{\theta_k}(x) \cdot \Delta\theta_k \right)^2 \right]$$

这只需要计算 score 在 $\Delta\theta_k$ 方向上的投影的二阶矩，和当前 `compute_fim_diagonal` 的计算方式完全一致，只是方向从 $d_1, d_2$ 换成了 $\Delta\theta_k$。

同样地，可以分解到 PCA 方向：

$$S_{\text{KL}}^{(k)}(t) = \sum_{i=0}^{t-1} \sqrt{ \frac{1}{2}\, (\Delta\theta_i \cdot d_k)^2 \, F_{kk}^{(i)} }$$

---

## 3. 物理直觉：这些量告诉我们什么

### 3.1 FIM 曲线 → 学习阶段的指纹

PPO 训练通常经历几个阶段：

| 阶段 | 特征 | FIM 行为 |
|------|------|----------|
| 初期探索 | 分布还很宽，reward 低 | $F$ 较小（分布对参数不敏感） |
| 快速收敛 | 分布开始集中到高奖励区域 | $F$ 增大（分布剧烈变化） |
| 精调/震荡 | 分布已锁定峰值，微调位置 | $F$ 可能下降（局部已稳定）或震荡 |

从 spline 的 PPO 日志可以看到：iter 80 时 mean_r=-3.09，iter 180 时反弹到 -4.23，再到 iter 200 的 -1.93——存在明显的震荡。FIM 曲线可以揭示这些震荡对应的是分布的什么变化。

### 3.2 欧氏步长 vs KL 步长 → 自然梯度的必要性

如果发现 $\rho_k = s_\text{KL} / s_\text{Euclid}$ 变化很大（比如从 0.01 到 100），说明：

- **欧氏空间中相同的学习率，在不同训练阶段对应完全不同的"分布空间步长"**
- 这解释了为什么固定 lr 的 Adam 优化器会导致训练不稳定
- 自然梯度方法（Natural Policy Gradient）就是用 $F^{-1}$ 修正步长，使得每步在 KL 空间走固定距离

### 3.3 方向各向异性 → 参数空间的"狭缝"

如果 $F_{11}(t) \gg F_{22}(t)$，说明：

- 沿 PC1 方向移动参数会**剧烈改变分布**，而沿 PC2 方向移动几乎不影响
- 优化景观在 PC1 方向是"陡坡"，在 PC2 方向是"平原"
- PPO 在 PC2 方向的更新基本是浪费（分布不变但参数变了）

反过来，如果 $F_{11} \approx F_{22}$，说明两个主方向同等重要。

### 3.4 累积 KL 距离 → 学习效率的度量

$S_\text{KL}(T)$ 是 PPO 在分布空间总共走了多远。将它与 reward 改善量对比：

$$\text{效率} = \frac{\Delta J}{S_\text{KL}} = \frac{J(\theta_T) - J(\theta_0)}{S_\text{KL}(T)}$$

这是一个 **归一化的学习效率指标**：每单位 KL 距离获得了多少 reward 提升。可以用来比较不同架构（affine vs spline vs diffusion）的 **信息效率**，而不仅仅是看 reward 收敛曲线。

---

## 4. 与 2D 网格方法的对比

| | 2D 网格扫描 | 沿轨迹计算 |
|---|---|---|
| 参数点来源 | 合成的 $\theta_\text{mid} + \alpha d_1 + \beta d_2$ | PPO 实际访问的 $\theta_t$ |
| OOD 风险 | 网格边缘严重 OOD | 无（都是训练时的参数） |
| 输出 | 2D 热力图（好看但数值不可信） | 1D 曲线（可信，可解释） |
| 计算量 | $O(N^2 \cdot M)$，$N$=网格边长 | $O(T \cdot M)$，$T$=快照数 |
| 告诉你什么 | "如果参数在这里，FIM 是多少" | "训练过程中 FIM 怎么变化" |
| 可信度 | 中心区域可信，边缘不可信 | 全部可信 |

关键差异：2D 网格试图回答 **"参数空间长什么样"**（全局问题），但流模型只在训练过的参数配置下可靠，所以这个问题本身就不适合用流模型来回答。沿轨迹计算回答的是 **"训练过程中发生了什么"**（局部问题），这是模型确实能可靠回答的。

---

## 5. 具体可以画哪些图

### 图 A：FIM 沿训练过程的演化

- x 轴：PPO iteration（或 function evaluations）
- y 轴（对数刻度）：$F_{11}(t)$、$F_{22}(t)$
- 叠加 reward 曲线（右 y 轴）
- **预期**：FIM 在 reward 快速提升阶段最大

### 图 B：欧氏步长 vs KL 步长

- x 轴：PPO iteration
- y 轴：$s_\text{Euclid}(t)$（蓝）和 $s_\text{KL}(t)$（红）
- 比值 $\rho_t$（绿，右 y 轴）
- **预期**：$\rho_t$ 在训练中期最大（分布变化最剧烈的阶段）

### 图 C：累积 KL 距离 vs Reward

- x 轴：$S_\text{KL}(t)$（累积 KL 弧长）
- y 轴：$J(\theta_t)$（期望 reward）
- 同一张图上画 affine / spline / diffusion
- **预期**：某个架构在相同 KL 距离下达到更高 reward = 更高效

### 图 D：PCA 空间中的轨迹 + 局部 FIM 椭圆

- 底图：当前 fig1 的面板2（J(θ) 欧氏热力图，只用轨迹附近的小范围）
- 在每隔 N 步的参数点上画 FIM 椭圆（$F_{11}, F_{22}$ 决定长短轴）
- **预期**：椭圆在不同阶段有不同的大小和朝向

### 图 E：各向异性比

- x 轴：PPO iteration
- y 轴：$F_{11}(t) / F_{22}(t)$
- **预期**：比值偏离 1 的程度反映优化方向的偏好嗯

---

## 6. 与当前代码的衔接

### 需要保留的

- `compute_fim_diagonal` 函数的核心逻辑（采样 + score 投影 + trimmed mean）
- PCA 方向的计算
- `evaluate_J` 函数（计算期望 reward）

### 需要修改的

- 去掉 2D 网格循环 `for alpha in alphas: for beta in betas:`
- 改为沿轨迹循环 `for t, theta_t in param_snapshots:`
- 增加步进方向的 FIM 投影
- 新增可视化函数

### 计算量估算

当前 2D 网格：$21 \times 21 = 441$ 个点 $\times$ 100 samples/点 = 44,100 次 score 计算

轨迹方法：$T = 201$ 个快照 $\times$ 100 samples/点 = 20,100 次 score 计算

**计算量减半，且结果全部可信。**

---

## 7. 更远的思考

### 7.1 Natural Policy Gradient 的联系

传统 PPO 用 Adam 在欧氏空间做梯度下降。如果沿轨迹的 FIM 分析显示 $\rho_k$ 变化巨大（比如跨 3 个数量级），这直接论证了 **Natural Policy Gradient** 的必要性：

$$\theta_{t+1} = \theta_t + \eta \, F(\theta_t)^{-1} \, \nabla_\theta J(\theta_t)$$

虽然在高维中直接计算 $F^{-1}$ 不现实，但 FIM 的方向性信息可以指导更好的优化策略（比如 per-layer 学习率调整）。

### 7.2 为什么 J(θ) 景观在信息几何下更"平滑"

即使我们不做 2D 网格扫描，沿轨迹的 FIM 已经暗含了信息几何景观的信息：

- 如果 FIM 大的地方 reward 也在快速变化 → 信息几何下景观梯度 $\frac{dJ}{dS_\text{KL}}$ **平稳**
- 如果 FIM 大的地方 reward 反而不变 → 模型在"空转"，分布变化大但 reward 不跟着变
- 如果 FIM 小的地方 reward 突然跳变 → 参数空间中存在"phase transition"

这些定性分析不需要 2D 热力图就能得到。
