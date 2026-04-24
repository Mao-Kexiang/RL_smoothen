# RealNVP 的数值稳定性问题与局限性

## 问题现象

在 RL_smoothen 项目中，使用 RealNVP（仿射耦合流）进行 PPO 微调时，FIM（Fisher Information Matrix）计算出现极端值：

```
F11 range: [0.147313, 11977463030.129091]  # 比值 8e10
F22 range: [0.735484, 51894343064.452660]  # 比值 7e10
KL-beta range: [-111703, 105746]            # 跨度 21 万
```

这导致 KL 坐标系统崩溃，3D 可视化中轨迹与曲面不匹配。

---

## 根本原因

### 1. 仿射变换的表达力限制

RealNVP 的核心变换是：

```
y = x * exp(s(x_cond)) + t(x_cond)
log_det = s(x_cond)
```

**问题**：
- `s` 被 clamp 到 `[-2, 2]`，限制了单层的缩放范围在 `[e^-2, e^2] ≈ [0.14, 7.4]`
- 8 层累积后，总缩放范围是 `[0.14^8, 7.4^8] ≈ [3.7e-8, 1.6e6]`
- 当模型需要表达**多峰分布**（如 Rastrigin 的多个局部极值）时，仿射变换力不从心
- 为了"挤出"足够的表达力，网络会让 `s` 饱和在 ±2 边界，导致梯度病态

### 2. 参数空间的"病态"区域

在 FIM 扫描时，我们沿 PCA 方向移动参数：

```
θ = θ_center + α·d1 + β·d2
```

**问题**：
- 训练时模型只见过 `θ_center` 附近的参数配置
- 当 `(α, β)` 偏离中心时，模型进入**未训练的参数空间**：
  - 条件网络 `s(x_cond)` 输出极端值（超出训练时的分布）
  - 耦合层的 clamp 饱和，`s = ±2` 成为常数
  - `log_prob` 的梯度 `∇_θ log π(x)` 数值爆炸或消失

**类比**：就像把训练好的神经网络输入 OOD（out-of-distribution）数据，输出不可信。

### 3. 雅可比行列式的累积误差

RealNVP 的 `log_prob` 计算：

```python
z, log_det_inv = model.inverse(x)  # x → z
log_pz = -0.5 * (z^2 + log(2π)).sum()
log_prob = log_pz + log_det_inv
```

**问题**：
- `log_det_inv = -Σ s_i`，是 8 层 scale 的累加
- 如果某层的 `s_i` 很大（接近 +2），`log_det_inv` 会是大负数
- 当 `log_prob` 对 `θ` 求导时，链式法则会放大这个误差
- 在参数空间边缘，`∇_θ log_det` 可能爆炸到 1e10 量级

### 4. 采样的尾部事件

FIM 计算需要从模型采样：

```python
z ~ N(0, I)
x = model.forward(z)
```

**问题**：
- 小概率采到 `|z| > 4` 的极端值（概率 ~0.01%）
- 经过非线性流变换后，可能映射到：
  - 分布的极端尾部（`log π(x) → -∞`）
  - 数值不稳定区域（雅可比接近奇异）
- 在这些点，`∇_θ log π(x)` 接近无穷，污染 FIM 估计

---

## 为什么 Spline Flow 更好

Neural Spline Flow 使用**有理二次样条**替代仿射变换：

```
y = RQS(x; widths, heights, derivatives)
```

**优势**：

1. **更强的表达力**：
   - 单层可以做复杂的非线性单调映射
   - 不需要饱和在边界来"挤出"表达力
   - 8 bins 的样条 ≈ 仿射的 10+ 层

2. **更平滑的梯度**：
   - 样条的导数是连续的（C^1 光滑）
   - 不会像 clamp 那样产生梯度截断
   - `log_det` 的计算更稳定（解析公式，无指数运算）

3. **更好的数值稳定性**：
   - `tail_bound` 外自动退化为恒等变换（安全边界）
   - 二次方程求解有判别式保护
   - 不会出现 `exp(s)` 的指数爆炸

**代价**：
- 计算量稍大（每层需要 searchsorted + 二次方程）
- 参数量更多（3K+1 个参数 vs 仿射的 2 个）

---

## 解决方案对比

### 方案 A：鲁棒统计估计（已采用）

```python
# 用 trimmed mean 过滤离群值
projections = [compute_grad_projection(sample_i) for i in range(100)]
f11 = trimmed_mean([p^2 for p in projections], trim=0.1)
```

**优点**：
- 数据驱动，不依赖硬编码阈值
- 自动过滤极端样本（去掉最大最小 10%）
- 保留了 FIM 的相对大小关系

**缺点**：
- 治标不治本，模型数值问题依然存在
- 增加采样数（100 vs 30）会拖慢 FIM 扫描

### 方案 B：切换到 Spline Flow

```python
model = FlowModel(coupling='spline', n_bins=8)
```

**优点**：
- 从根源上解决表达力和数值稳定性问题
- 训练收敛更快（更少的层数）
- FIM 不会出现极端值

**缺点**：
- 需要重新训练（预训练的 RealNVP 参数不能迁移）
- 计算稍慢（~1.5x）

### 方案 C：更保守的 RealNVP

```python
# 在 AffineCoupling 中
s = s.clamp(-1.5, 1.5)  # 更严格的 clamp
log_det = log_det.clamp(-15, 15)  # 防止累积爆炸
```

**优点**：
- 简单，不需要重新训练
- 减少极端值的概率

**缺点**：
- 进一步限制了表达力
- 可能影响模型拟合多峰分布的能力

---

## 理论背景：为什么 FIM 会爆炸

Fisher Information Matrix 的定义：

```
F_ij = E_x~π_θ [ (∂log π_θ(x)/∂θ_i) * (∂log π_θ(x)/∂θ_j) ]
```

对于流模型：

```
log π_θ(x) = log p_z(f^-1(x)) + log|det J_f^-1(x)|
```

求导：

```
∂log π_θ/∂θ = ∂log p_z/∂z · ∂z/∂θ + ∂log|det J|/∂θ
```

**爆炸的来源**：

1. **第一项**：当 `z = f^-1(x)` 落在高斯尾部（|z| > 5），`∂log p_z/∂z = -z` 很大
2. **第二项**：当 `log|det J|` 累积到极端值（±20），对 `θ` 的导数会放大这个信号
3. **链式法则**：8 层的链式求导会指数级放大误差

**为什么 Spline 不爆炸**：
- 样条的 `log|det J|` 有界（由 tail_bound 和 softmax 归一化保证）
- 不需要 clamp，梯度流更平滑
- 单层表达力强，不需要深层累积

---

## 实验建议

### 短期（保持 RealNVP）

1. 使用方案 A（trimmed mean + n_samples=100）
2. 减小 PCA 扫描范围（`grid_range=2.0` 而不是 3.0）
3. 增加预训练 epochs（让模型更稳定）

### 长期（切换到 Spline）

1. 用 `coupling='spline'` 重新训练
2. 对比 RealNVP vs Spline 的：
   - 训练收敛速度
   - PPO 最终性能
   - FIM 数值稳定性
3. 如果 Spline 显著更好，写进论文的 ablation study

---

## 参考文献

1. **RealNVP**: Dinh et al., "Density estimation using Real NVP", ICLR 2017
2. **Neural Spline Flows**: Durkan et al., "Neural Spline Flows", NeurIPS 2019
3. **FIM in RL**: Martens & Grosse, "Optimizing Neural Networks with Kronecker-factored Approximate Curvature", ICML 2015

---

## 附录：诊断工具

如果想定位哪些参数点导致 FIM 爆炸，可以在 `compute_fim_diagonal` 中加日志：

```python
f11, f22 = compute_fim_diagonal(...)
if f11 > 1e4 or f22 > 1e4:
    print(f"  WARNING: FIM explosion at (α={alpha:.2f}, β={beta:.2f}): "
          f"F11={f11:.2e}, F22={f22:.2e}")
```

然后可视化这些"病态"点在参数空间的分布，看是否有规律（比如都在边界附近）。
