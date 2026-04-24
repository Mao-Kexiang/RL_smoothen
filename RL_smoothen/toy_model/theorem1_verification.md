# Theorem 1 验证：$C_v$ 与 Natural Gradient Norm 的关系

## 不等式链

$$C_v \;\leq\; \|\tilde{\nabla}J\|_F^2 \;\leq\; \mathrm{Var}_{\pi_\theta}(r) \;\leq\; (R_{\max}^{\mathrm{eff}})^2$$

| 量 | 定义 | 是否可算 |
|----|------|----------|
| $C_v$ | $\dfrac{(\mathbb{E}[r \cdot (g \cdot v)])^2}{\mathbb{E}[(g \cdot v)^2]} = \dfrac{(v^\top \nabla J)^2}{v^\top F\,v}$ | 可算（配对样本） |
| $\|\tilde{\nabla}J\|_F^2$ | $(\nabla J)^\top F^{-1} (\nabla J)$ | 不可算（需 $F^{-1} \in \mathbb{R}^{D \times D}$） |
| $\mathrm{Var}(r)$ | $\mathbb{E}[(r - \bar{r})^2]$ | 可算（采样） |
| $R_{\max}^{\mathrm{eff}}$ | $\max_{x \in \mathrm{supp}(\pi_\theta)} |r(x)|$ | 可算（batch max，下界估计） |

其中 $g = \nabla_\theta \log \pi_\theta(x)$ 为 score，$v = \Delta\theta_t$ 为 PPO 步进方向。

## $C_v \leq \|\tilde{\nabla}J\|_F^2$ 的推导

对 $F$-内积做 Cauchy-Schwarz：

$$(v^\top \nabla J)^2 = \left(v^\top F \cdot F^{-1}\nabla J\right)^2 \leq \underbrace{(v^\top F\,v)}_{= \mathbb{E}[(g \cdot v)^2]}\;\cdot\;\underbrace{(\nabla J^\top F^{-1} \nabla J)}_{= \|\tilde{\nabla}J\|_F^2}$$

两边除以 $v^\top F\,v$：

$$C_v = \frac{(v^\top \nabla J)^2}{v^\top F\,v} \leq \|\tilde{\nabla}J\|_F^2 \qquad \blacksquare$$

**等号条件**：$v \propto F^{-1}\nabla J = \tilde{\nabla}J$（$v$ 恰好是 natural gradient 方向）。

## 几何意义

$$\frac{C_v}{\|\tilde{\nabla}J\|_F^2} = \cos^2\alpha$$

其中 $\alpha$ 是 $v$（PPO 步进方向）与 $\tilde{\nabla}J$（natural gradient）在 Fisher 度量下的夹角。

## Fig 12 下面板的解读

下面板画的是：

$$\frac{C_v}{\mathrm{Var}(r)} = \frac{C_v}{\|\tilde{\nabla}J\|_F^2} \;\cdot\; \frac{\|\tilde{\nabla}J\|_F^2}{\mathrm{Var}(r)} = \cos^2\alpha \;\cdot\; \frac{\|\tilde{\nabla}J\|_F^2}{\mathrm{Var}(r)}$$

这个比值混合了两个效应，无法分离：

1. **$\cos^2\alpha$**：PPO 方向与 natural gradient 的对齐程度（Adam vs. natural gradient 的差异）
2. **$\|\tilde{\nabla}J\|_F^2 / \mathrm{Var}(r)$**：Theorem 1 bound 本身的松紧度

如果 $C_v / \mathrm{Var}(r) \approx 1$：两个因子都接近 1，即 PPO 方向 $\approx$ natural gradient **且** bound 是紧的。

如果 $C_v / \mathrm{Var}(r) \ll 1$：可能是 PPO 方向偏离 natural gradient，也可能是 bound 本身很松，或两者皆有。

## 为什么不能直接算 $\|\tilde{\nabla}J\|_F^2$

$F^{-1}$ 是 $D \times D$ 矩阵（$D = 35\text{K}$），约 $5 \times 10^9$ 个元素，无法存储或求逆。即使用 $N$ 个样本估计 $\hat{F} = \frac{1}{N}\sum g_i g_i^\top$，当 $N < D$ 时 $\hat{F}$ 不满秩，伪逆给出的也只是投影到样本张成子空间后的下界。
