# 流模型单元测试报告

## 1. 什么是单元测试

单元测试（Unit Test）是软件工程中最基础的测试方法：**把代码拆成最小的可测试单元，为每个单元编写独立的测试用例，验证其行为是否符合预期。**

以流模型为例，"单元"可以是：
- 一个样条耦合层（SplineCoupling）的 forward/inverse
- 整个 FlowModel 的 `log_prob` 函数
- `sample()` 与 `log_prob()` 之间的数学一致性

单元测试的核心思想是：

| 原则 | 说明 |
|------|------|
| **独立性** | 每个测试用例只检验一个性质，互不依赖 |
| **可重复** | 固定随机种子，每次运行结果一致 |
| **自动判定** | 通过/失败由程序自动判断，无需人工检查 |
| **快速反馈** | 几秒内跑完，随时可以运行 |

在实践中，我们使用 **pytest** 框架：把测试函数写在 `test_*.py` 文件中，用 `assert` 语句声明期望条件，然后一行命令运行全部测试：

```bash
cd toy_model/exp_2d && python -m pytest test_model.py -v
```

---

## 2. 为什么流模型需要单元测试

流模型（Normalizing Flow）的训练和推理依赖于严格的数学性质。如果这些性质被破坏，模型不会报错，但会**静默地产生错误的概率密度和梯度**，导致 PPO 训练行为异常却难以排查。

具体来说，流模型的正确性建立在以下数学链条上：

$$z \sim \mathcal{N}(0, I) \xrightarrow{f_\theta} x, \quad \log p_\theta(x) = \log p_z(f_\theta^{-1}(x)) + \log \left|\det \frac{\partial f_\theta^{-1}}{\partial x}\right|$$

这个链条中任何一环出错——逆变换不精确、log-det 计算有误、前向和逆向不匹配——都会导致 `log_prob` 的值错误，进而使 PPO 的重要性权重 $\frac{\pi_\theta(x)}{\pi_{\theta_{\text{old}}}(x)}$ 偏离真实值。

单元测试的目的就是**逐一验证这条链条的每个环节**。

---

## 3. 测试文件结构

`test_model.py` 包含 **5 类共 16 个测试用例**，覆盖流模型的所有关键数学性质：

```
test_model.py
├── Fixtures（测试夹具）
│   ├── spline_model    — 8 层 Neural Spline Flow
│   ├── affine_model    — 8 层 RealNVP（仿射耦合）
│   ├── spline_layer    — 单个 SplineCoupling 层
│   └── affine_layer    — 单个 AffineCoupling 层
│
├── TestInvertibility（可逆性，5 个测试）
├── TestLogDet（Log-det 正确性，5 个测试）
├── TestLogProbConsistency（Log-prob 自洽性，3 个测试）
├── TestNormalization（归一化，1 个测试）
└── TestLogDetSymmetry（Log-det 对称性，2 个测试）
```

**Fixtures**（夹具）是 pytest 的机制：用 `@pytest.fixture` 标记的函数会在测试前自动执行，构造好模型实例并传入测试函数。所有 fixture 都固定 `torch.manual_seed(42)`，保证每次运行结果一致。

---

## 4. 五类测试详解

### 4.1 可逆性测试（TestInvertibility）

**数学原理：** 流模型的变换 $f_\theta$ 必须是严格可逆的。即对任意输入，先前向再逆向（或反过来）必须恢复原值：

$$f_\theta^{-1}(f_\theta(z)) = z, \quad f_\theta(f_\theta^{-1}(x)) = x$$

**测试方法：** 生成随机输入，做 round-trip，检查最大绝对误差 < $10^{-5}$。

| 测试名 | 做了什么 |
|--------|---------|
| `test_spline_coupling_invertibility` | 单个 SplineCoupling 层：$x \to y \to x'$，检查 $x \approx x'$ |
| `test_affine_coupling_invertibility` | 单个 AffineCoupling 层：同上 |
| `test_flow_model_spline_invertibility` | 完整 Spline 模型：$z \to x \to z'$，检查 $z \approx z'$ |
| `test_flow_model_affine_invertibility` | 完整 Affine 模型：同上 |
| `test_inverse_then_forward` | 反方向 round-trip：$x \to z \to x'$，检查 $x \approx x'$ |

**如果此测试失败：** 说明 forward 和 inverse 的实现不匹配，很可能是样条求逆的二次方程求解有数值问题。

### 4.2 Log-det 正确性测试（TestLogDet）

**数学原理：** 每一层耦合变换 $y = g(x)$ 都会输出一个 $\log|\det J|$，其中 $J = \partial y / \partial x$ 是 Jacobian 矩阵。这个值必须精确，因为它直接参与 `log_prob` 的计算。

**测试方法：** 用 PyTorch 的 `torch.autograd.functional.jacobian` 自动微分计算完整的 $2 \times 2$ Jacobian 矩阵，然后取 $\log|\det(\cdot)|$ 作为"真值"，与模型输出的 `log_det` 对比：

```python
# 对每个样本 z_i，计算完整 Jacobian
J = torch.autograd.functional.jacobian(f, z_i)  # shape (2, 2)
log_det_true = torch.log(torch.abs(torch.det(J)))

# 与模型声称的 log_det 对比
assert |log_det_model - log_det_true| < 1e-4
```

| 测试名 | 做了什么 |
|--------|---------|
| `test_spline_coupling_logdet` | 单个 SplineCoupling 层的 forward log-det |
| `test_affine_coupling_logdet` | 单个 AffineCoupling 层的 forward log-det |
| `test_flow_model_forward_logdet` | 完整 Spline 模型 forward 的 log-det |
| `test_flow_model_inverse_logdet` | 完整 Spline 模型 inverse 的 log-det |
| `test_affine_flow_model_forward_logdet` | 完整 Affine 模型 forward 的 log-det |

**为什么用 autograd 而不是解析推导？** autograd 计算的是"暴力"的数值真值——直接对变换函数求导组装出完整 Jacobian——不依赖任何关于耦合层结构的假设。如果模型代码中的 log-det 公式写错了（比如少了一个负号、漏了一项），autograd 一定能检测出来。

**如果此测试失败：** 说明 `rational_quadratic_spline` 或 `AffineCoupling` 中的 log-det 公式有误。这是最危险的 bug，因为 log-det 错误不会导致 NaN，但会让 `log_prob` 系统性地偏离真实值。

### 4.3 Log-prob 自洽性测试（TestLogProbConsistency）

**数学原理：** `log_prob` 的值可以通过两条独立路径得到，两条路径必须给出相同结果：

- **路径 A**（`sample` 方法）：采样 $z$，前向得到 $x$，计算 $\log p_z(z) - \log|\det J_{\text{fwd}}|$
- **路径 B**（`log_prob` 方法）：给定 $x$，逆向得到 $z$，计算 $\log p_z(z) + \log|\det J_{\text{inv}}|$

| 测试名 | 做了什么 |
|--------|---------|
| `test_sample_logprob_matches_logprob` | 用 Spline 模型 sample 出 $(x, \log p)$，再用 `log_prob(x)` 重新算，两者应一致 |
| `test_sample_logprob_matches_logprob_affine` | 同上，Affine 模型 |
| `test_logprob_decomposition` | 手动执行 inverse → 计算 $\log p_z$ → 加 log-det，与 `log_prob()` 输出对比 |

**如果此测试失败：** 说明 `sample()` 和 `log_prob()` 对 log-det 的符号处理不一致（forward 的 log-det 应该被减去，inverse 的应该被加上）。

### 4.4 归一化测试（TestNormalization）

**数学原理：** 任何概率密度函数必须满足 $\int p_\theta(x) \, dx = 1$。

**测试方法：** 在 $[-6, 6]^2$ 上铺一个 $400 \times 400$ 的均匀网格（共 160,000 个点），对每个点计算 `exp(log_prob)`，然后用矩形法数值积分：

$$\hat{I} = \sum_{i,j} \exp(\log p_\theta(x_{ij})) \cdot \Delta x^2 \approx 1.0$$

容差设为 0.05，因为网格积分有离散化误差，且 $[-6,6]^2$ 外的尾部被截断。

| 测试名 | 做了什么 |
|--------|---------|
| `test_density_integrates_to_one` | 网格积分验证概率密度总和 $\approx 1$ |

**如果此测试失败：** 说明 log-det 有系统性偏差（比如全局偏移了一个常数），导致密度被整体放大或缩小。

### 4.5 Log-det 对称性测试（TestLogDetSymmetry）

**数学原理：** 对同一组数据，forward 的 log-det 和 inverse 的 log-det 应该互相抵消：

$$\log|\det J_{\text{fwd}}(z)| + \log|\det J_{\text{inv}}(f(z))| = 0$$

这是因为 $J_{\text{inv}} = J_{\text{fwd}}^{-1}$，所以 $\det(J_{\text{inv}}) = 1/\det(J_{\text{fwd}})$。

| 测试名 | 做了什么 |
|--------|---------|
| `test_forward_inverse_logdet_cancel` | Spline 模型：$z \xrightarrow{\text{fwd}} (x, \ell_1)$，$x \xrightarrow{\text{inv}} (z', \ell_2)$，检查 $\ell_1 + \ell_2 \approx 0$ |
| `test_forward_inverse_logdet_cancel_affine` | 同上，Affine 模型 |

**如果此测试失败：** 和 4.2 类似，但这个测试更直接——它不依赖 autograd，而是检查 forward 和 inverse 两个代码路径是否在 log-det 的符号和数值上完全对称。

---

## 5. 测试结果

```
16 passed, 1 warning in 2.85s
```

所有 16 个测试全部通过。唯一的 warning 是 `torch.searchsorted` 对非连续 tensor 的性能提示，不影响正确性。

| 测试类别 | 测试数 | 状态 |
|----------|--------|------|
| 可逆性 | 5 | 全部通过 |
| Log-det 正确性 | 5 | 全部通过 |
| Log-prob 自洽性 | 3 | 全部通过 |
| 归一化 | 1 | 全部通过 |
| Log-det 对称性 | 2 | 全部通过 |

**结论：** FlowModel 的 Spline 和 Affine 两种耦合实现在数学上是正确的——可逆变换精确、log-det Jacobian 与自动微分一致、概率密度正确归一化。可以放心用于后续的 PPO 训练。

---

## 6. 如何运行

```bash
# 激活环境
conda activate RLtoy

# 运行全部测试（在 exp_2d 目录下）
cd toy_model/exp_2d && python -m pytest test_model.py -v

# 只运行某一类测试
python -m pytest test_model.py::TestLogDet -v

# 只运行单个测试
python -m pytest test_model.py::TestNormalization::test_density_integrates_to_one -v
```
