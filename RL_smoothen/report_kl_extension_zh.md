# KL 正则化作为主动景观重塑：平滑性理论的推广

## 0. 动机

主报告（`report.md`）建立了策略优化平滑崎岖景观的理论：定理 1 给出 $\|\tilde{\nabla}J\|_F^2 \leq \mathrm{Var}_{\pi_\theta}(r)$，预训练保证 $\mathrm{Var}(r)$ 有限。然而，KL 惩罚项 $\lambda D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}})$ 在原报告中仅被视为被动的安全机制——防止策略漂移进奇异区域。

本报告证明 KL 正则化扮演着远为深刻的角色：它**主动重塑优化景观**，使其在训练过程中逐步变得更平滑，并最终在 Fisher-Rao 几何下成为强凹函数。这提供了无正则化理论无法给出的指数收敛保证。

---

## 1. 有效奖励表示

### 1.1 KL 正则化目标的梯度

正则化目标为：

$$J_\lambda(\theta) = \mathbb{E}_{\pi_\theta}[r(x)] - \lambda D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}})$$

计算 $\nabla_\theta J_\lambda$ 需要 KL 项的梯度。利用 $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$ 展开：

$$\nabla_\theta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}}) = \nabla_\theta \int \pi_\theta(x) \log \frac{\pi_\theta(x)}{\pi_{\mathrm{base}}(x)} \, dx$$

对被积函数用乘积法则：

$$= \int \pi_\theta \nabla_\theta \log \pi_\theta \cdot \log \frac{\pi_\theta}{\pi_{\mathrm{base}}} \, dx + \int \underbrace{\nabla_\theta \pi_\theta}_{= \pi_\theta \nabla_\theta \log \pi_\theta} \, dx$$

第二个积分为 $\nabla_\theta \int \pi_\theta \, dx = \nabla_\theta \, 1 = 0$。因此：

$$\nabla_\theta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}}) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(x) \cdot \log \frac{\pi_\theta(x)}{\pi_{\mathrm{base}}(x)}\right]$$

与策略梯度 $\nabla_\theta \mathbb{E}[r] = \mathbb{E}[r \cdot \nabla_\theta \log \pi_\theta]$ 合并：

$$\boxed{\nabla_\theta J_\lambda = \mathbb{E}_{\pi_\theta}\!\left[\tilde{r}_\lambda(x;\theta) \cdot \nabla_\theta \log \pi_\theta(x)\right]}$$

其中**有效奖励**定义为：

$$\tilde{r}_\lambda(x;\theta) \;\triangleq\; r(x) - \lambda \log \frac{\pi_\theta(x)}{\pi_{\mathrm{base}}(x)}$$

这是一个已知结果（参见 Ziebart 2010; Levine 2018），但其对景观平滑性的含义尚未被充分挖掘。

**与无正则化情形的关键区别**：$\tilde{r}_\lambda$ 依赖于 $\theta$，因此在训练过程中与策略共同演化。

---

## 2. 定理 1 的推广

### 2.1 陈述

**定理 1$'$（KL 正则化自然梯度界）。** 对任意参数族 $\pi_\theta$，KL 正则化目标 $J_\lambda$ 满足：

$$\|\tilde{\nabla} J_\lambda(\theta)\|_F^2 \;\leq\; \mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda)$$

**证明。** 与主报告中定理 1 完全相同，仅将 $r$ 替换为 $\tilde{r}_\lambda$。令 $g(x) = \nabla_\theta \log \pi_\theta(x)$，由 $\mathbb{E}[g] = 0$：

$$\nabla_\theta J_\lambda = \mathbb{E}[\tilde{r}_\lambda \cdot g] = \mathrm{Cov}(\tilde{r}_\lambda, g)$$

由多元 Cauchy-Schwarz 不等式：

$$\|\tilde{\nabla}J_\lambda\|_F^2 = \mathrm{Cov}(\tilde{r}_\lambda, g)^\top [\mathrm{Var}(g)]^{-1} \mathrm{Cov}(\tilde{r}_\lambda, g) \leq \mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda) \qquad \blacksquare$$

### 2.2 方差分解

展开 $\mathrm{Var}(\tilde{r}_\lambda)$ 揭示其内部结构：

$$\mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda) = \mathrm{Var}(r) + \lambda^2 \mathrm{Var}\!\left(\log\frac{\pi_\theta}{\pi_{\mathrm{base}}}\right) - 2\lambda\,\mathrm{Cov}\!\left(r,\; \log\frac{\pi_\theta}{\pi_{\mathrm{base}}}\right)$$

| 项 | 符号 | 成功训练中的行为 |
|---|---|---|
| $\mathrm{Var}(r)$ | $+$ | 随 $\pi_\theta$ 集中到高奖励区域而下降 |
| $\lambda^2 \mathrm{Var}(\log \text{ratio})$ | $+$ | 随 $\pi_\theta$ 偏离 $\pi_{\mathrm{base}}$ 而上升 |
| $-2\lambda\,\mathrm{Cov}(r, \log\text{ratio})$ | $-$ | 当高奖励区域的 $\pi_\theta/\pi_{\mathrm{base}}$ 较大时为**负** |

协方差项至关重要：在正确的优化过程中，$\pi_\theta$ 会在高奖励区域赋予更高的概率密度，因此 $\mathrm{Cov}(r, \log(\pi_\theta/\pi_{\mathrm{base}})) > 0$，交叉项**降低**了总方差。在有利条件下：

$$\mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda) < \mathrm{Var}_{\pi_\theta}(r)$$

这意味着 KL 惩罚提供了**超越无正则化情形的额外平滑效果**。

---

## 3. 最优解处的景观坍缩

### 3.1 闭式最优策略

KL 正则化目标 $J_\lambda(\pi) = \mathbb{E}_\pi[r] - \lambda D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{base}})$ 在分布空间有唯一的全局最大化者：

$$\pi^*_\lambda(x) = \frac{1}{Z(\lambda)} \pi_{\mathrm{base}}(x) \exp\!\left(\frac{r(x)}{\lambda}\right), \qquad Z(\lambda) = \mathbb{E}_{\pi_{\mathrm{base}}}\!\left[\exp\!\left(\frac{r(x)}{\lambda}\right)\right]$$

这是从基分布出发的 Boltzmann（softmax）策略。

### 3.2 有效奖励变为常数

在 $\pi^*_\lambda$ 处，对数比值为：

$$\log \frac{\pi^*_\lambda(x)}{\pi_{\mathrm{base}}(x)} = \frac{r(x)}{\lambda} - \log Z(\lambda)$$

代入 $\tilde{r}_\lambda$：

$$\tilde{r}_\lambda(x;\theta^*) = r(x) - \lambda\!\left(\frac{r(x)}{\lambda} - \log Z(\lambda)\right) = \lambda \log Z(\lambda) = \text{常数}$$

因此：

$$\boxed{\mathrm{Var}_{\pi^*_\lambda}(\tilde{r}_\lambda) = 0 \qquad \Longrightarrow \qquad \tilde{\nabla} J_\lambda(\theta^*) = \mathbf{0}}$$

### 3.3 与无正则化情形的对比

| 性质 | 无 KL（$\lambda = 0$） | 有 KL（$\lambda > 0$） |
|------|------------------------|------------------------|
| 最优策略 | $\pi^* = \delta(x - x^*)$（Dirac delta） | $\pi^*_\lambda \propto \pi_{\mathrm{base}} \exp(r/\lambda)$（光滑分布） |
| 最优解处 $\mathrm{Var}(r)$ | $0$（平凡，单点支撑） | 一般 $> 0$ |
| 最优解处 $\mathrm{Var}(\tilde{r}_\lambda)$ | 不适用 | $= 0$（精确） |
| 有限参数模型可达？ | 否（无法表示 $\delta$ 分布） | 是（光滑目标分布） |
| 梯度消失 | 仅在 $\sigma \to 0$ 的极限下 | 在有限的 $\pi^*_\lambda$ 处精确成立 |

无正则化情形要求策略坍缩为点质量——这对任何有限参数模型都不可能。正则化情形要求匹配一个光滑的 Boltzmann 分布——normalizing flow 或扩散模型可以很好地逼近。这是本质性的改进：**正则化景观拥有一个可达的驻点，在该点自然梯度精确为零**。

---

## 4. Fisher-Rao 几何下的强凹性

### 4.1 陈述

**定理 2（强凹性）。** KL 正则化目标 $J_\lambda$ 在分布空间中关于 KL 散度是 $\lambda$-强凹的：

$$J_\lambda(\pi) \leq J_\lambda(\pi') + \langle \nabla_\pi J_\lambda(\pi'),\; \pi - \pi' \rangle - \lambda\, D_{\mathrm{KL}}(\pi \| \pi')$$

对所有定义域内的分布 $\pi, \pi'$ 成立。

**证明概要。** 分解 $J_\lambda$：

$$J_\lambda(\pi) = \underbrace{\mathbb{E}_\pi[r]}_{\text{关于 }\pi\text{ 线性}} - \lambda \underbrace{D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{base}})}_{\text{关于 }\pi\text{ 凸}}$$

第一项关于 $\pi$ 线性（因此既凸又凹）。KL 散度 $D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{base}})$ 关于 $\pi$ 严格凸，凸性模量为 1（以自身为距离度量）。因此 $-\lambda D_{\mathrm{KL}}$ 是 $\lambda$-强凹的。$\qquad \blacksquare$

### 4.2 推论：自然梯度的指数收敛

对于表达力足够强的参数族（Fisher-Rao 几何忠实地表示分布空间），Fisher 度量下的强凹性意味着自然梯度上升法指数收敛：

$$D_{\mathrm{KL}}(\pi^*_\lambda \| \pi_{\theta_t}) \leq (1 - \eta\lambda)^t \cdot D_{\mathrm{KL}}(\pi^*_\lambda \| \pi_{\theta_0})$$

其中 $\eta$ 是自然梯度步长（以 Fisher 弧长衡量）。

**与 $\lambda = 0$ 的对比**：无 KL 正则化时，$J(\theta) = \mathbb{E}_{\pi_\theta}[r]$ 关于 $\pi$ 线性——在分布空间中既不凸也不凹。不存在一般性的收敛速率保证，景观可能有鞍点、平台或其他病态结构。

### 4.3 物理意义

在最优解 $\pi^*_\lambda$ 附近：
- 景观在 Fisher 空间的**每个**方向上都向下弯曲（凹）。
- 除 $\pi^*_\lambda$ 外不存在其他局部极大值（唯一全局最大值）。
- 曲率至少为 $\lambda$——$\lambda$ 越大收敛越快，但偏差也越大（见第 6 节）。

这是整个理论框架中最强的保证：**KL 正则化将优化问题从非凸景观导航转化为具有指数收敛的强凹优化**。

---

## 5. KL 的三重角色——统一视角

主报告为 KL 约束辨识了两个角色。有效奖励分析揭示了第三个：

| 角色 | 机制 | 数学内容 |
|------|------|---------|
| **信赖域** | 限制每步 $D_{\mathrm{KL}}(\pi_\theta \| \pi_{\theta_{\mathrm{old}}})$ | 防止灾难性策略更新；保证重要性采样可靠 |
| **安全性** | Pinsker: $\mathrm{TV} \leq \sqrt{\delta/2}$ | 维持 $R_{\max}^{\mathrm{eff}}(\pi_\theta) \approx R_{\max}^{\mathrm{eff}}(\pi_{\mathrm{base}}) < \infty$ |
| **主动平滑**（新） | $\mathrm{Var}(\tilde{r}_\lambda) \to 0$（$\pi_\theta \to \pi^*_\lambda$ 时） | 将景观重塑为强凹；保证指数收敛 |

第三个角色最为深刻：KL 惩罚不仅仅是约束优化——它从根本上改变了目标函数的几何结构，在原本不存在良好景观的地方创造出一个。

---

## 6. 偏差-平滑性权衡

### 6.1 正则化引入的偏差

KL 正则化的最优奖励严格小于全局最优：

$$\mathbb{E}_{\pi^*_\lambda}[r] < r(x^*) = \max_x r(x)$$

差距可表示为：

$$r(x^*) - \mathbb{E}_{\pi^*_\lambda}[r] = \lambda\!\left(\log Z(\lambda) + D_{\mathrm{KL}}(\pi^*_\lambda \| \pi_{\mathrm{base}})\right)$$

### 6.2 两个极限区间

**$\lambda \to 0$（弱正则化）：**
- $\pi^*_\lambda \to \delta(x^*)$：无偏差，最大奖励
- $\mathrm{Var}(\tilde{r}_\lambda) \to \mathrm{Var}(r)$：无额外平滑
- 无强凹性保证
- 优化可能缓慢或陷入局部极值

**$\lambda \to \infty$（强正则化）：**
- $\pi^*_\lambda \to \pi_{\mathrm{base}}$：最大偏差，无奖励提升
- $\mathrm{Var}(\tilde{r}_\lambda) \to 0$：极度平滑（平坦景观）
- 强凹性模量 $\lambda$ 极大
- 优化瞬间收敛到 $\pi_{\mathrm{base}}$（平凡解）

### 6.3 最优 $\lambda$

存在最优 $\lambda^*$ 平衡优化质量与收敛速度：

$$\lambda^* = \arg\min_\lambda \left[\underbrace{\text{优化误差}(\lambda)}_{\sim \exp(-\lambda \cdot T)} + \underbrace{\text{偏差}(\lambda)}_{\sim \lambda \cdot (\log Z + D_{\mathrm{KL}})}\right]$$

其中 $T$ 为计算预算（自然梯度步数）。优化误差随 $\lambda$ 指数下降（来自强凹性），而偏差随 $\lambda$ 线性增长。此权衡尖锐且依赖于具体问题：

- **崎岖景观**（大量局部极值，$L_r$ 大）：较大的 $\lambda^*$ 有利——平滑收益超过偏差代价。
- **简单景观**（少量局部极值）：较小的 $\lambda^*$ 即可——不需要太多平滑。
- **计算资源有限**（$T$ 小）：较大的 $\lambda^*$——需要更快收敛，即使牺牲偏差。

---

## 7. 更新后的理论链

纳入 KL 正则化后，完整的理论链变为：

$$\boxed{\text{预训练} \xrightarrow{R_{\max}^{\mathrm{eff}} < \infty} \text{定理 1' 成立} \xrightarrow[\mathrm{Var}(\tilde{r}_\lambda) \to 0]{\text{强凹性}} \text{指数收敛} \xrightarrow{D_{\mathrm{KL}} \leq \delta} \text{维持安全区域}}$$

| 步骤 | 无 KL（$\lambda = 0$） | 有 KL（$\lambda > 0$） |
|------|------------------------|------------------------|
| 景观界 | $\|\tilde{\nabla}J\|_F \leq \sqrt{\mathrm{Var}(r)}$ | $\|\tilde{\nabla}J_\lambda\|_F \leq \sqrt{\mathrm{Var}(\tilde{r}_\lambda)} \leq \sqrt{\mathrm{Var}(r)}$ |
| 近最优行为 | $\mathrm{Var}(r)$ 有限但非零 | $\mathrm{Var}(\tilde{r}_\lambda) \to 0$（景观坍缩） |
| 凸性 | 无保证 | $\lambda$-强凹 |
| 收敛速率 | 无保证 | $O(e^{-\lambda t})$（Fisher 度量下） |
| 最优目标 | $\delta(x^*)$（不可达） | $\pi^*_\lambda$（光滑，可达） |
| 代价 | — | 偏差 $\sim \lambda(\log Z + D_{\mathrm{KL}})$ |

---

## 8. 对晶体结构预测的启示

在晶体结构预测场景中，$r(x)$ 具有 Coulomb 奇异性：

1. **预训练**将 $\pi_{\mathrm{base}}$ 限制在物理合理的构型上，使 $R_{\max}^{\mathrm{eff}} < \infty$。

2. **KL 正则化**同时完成三件事：
   - 防止 $\pi_\theta$ 漂移进奇异区域（安全性）。
   - 将有效奖励方差降至 $\mathrm{Var}(r)$ 以下（额外平滑）。
   - 使景观在 Boltzmann 最优解附近强凹（收敛保证）。

3. **最优 $\lambda$** 对晶体应用可能取中等值：能量景观高度崎岖（有利于较大 $\lambda$ 以获得平滑），但全局最小值定义尖锐（惩罚过大偏差）。

4. **对称性破缺**导致 $|r(x)| \to \infty$ 的问题被 KL 约束自动处理：即使 $r$ 在某些区域发散，$\tilde{r}_\lambda = r - \lambda \log(\pi_\theta/\pi_{\mathrm{base}})$ 会自我修正，因为 $\pi_\theta$ 无法在那些区域赋予显著概率而不承担巨大的 KL 惩罚。

---

## 9. 实验验证方案

以下量应在 toy model 中跟踪以验证理论：

### 9.1 需要新记录的量

在 `ppo.py` 中，每次迭代额外计算：

- $\mathrm{Var}(\tilde{r}_\lambda)$：有效奖励 $r(x) - \lambda \log(\pi_\theta(x)/\pi_{\mathrm{base}}(x))$ 的方差
- $\mathrm{Cov}(r, \log(\pi_\theta/\pi_{\mathrm{base}}))$：驱动额外平滑的交叉项
- 方差分解的三项分别记录

### 9.2 实验设计

| 实验 | 预期验证 |
|------|---------|
| **沿轨迹追踪 $\mathrm{Var}(\tilde{r}_\lambda)$ vs $\mathrm{Var}(r)$** | 训练中 $\mathrm{Var}(\tilde{r}_\lambda) < \mathrm{Var}(r)$（额外平滑） |
| **$\mathrm{Var}(\tilde{r}_\lambda)$ 衰减曲线** | 单调下降趋近 0（景观坍缩） |
| **$\lambda$ 扫描：$\{0, 0.1, 0.5, 1.0, 2.0\}$** | 偏差-平滑性权衡；最优 $\lambda^*$ |
| **收敛速度 vs $\lambda$** | 较大 $\lambda$ 初期收敛更快 |
| **$C_v^\lambda / \mathrm{Var}(\tilde{r}_\lambda)$** | 推广版定理 1' 的松紧度检验 |

### 9.3 预期结果

对 Boltzmann 预训练的 flow 模型在 2D Rastrigin 上：

1. $\lambda = 0$ 时：$\mathrm{Var}(r)$ 下降但保持在远离 0 的水平。
2. $\lambda > 0$ 时：$\mathrm{Var}(\tilde{r}_\lambda)$ 下降更快且趋近于 0。
3. 最终平均奖励：随 $\lambda$ 增大而下降（偏差效应）。
4. 收敛速度（达到最终奖励 90% 所需迭代数）：随 $\lambda$ 增大而加快（平滑效应）。
5. 最优 $\lambda^*$：平衡以上两个效应，对此问题可能在 $[0.1, 1.0]$ 范围内。
