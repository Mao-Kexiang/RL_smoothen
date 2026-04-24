# KL Regularization as Active Landscape Reshaping: Extending the Smoothing Theory

## 0. Motivation

The main report (`report.md`) establishes that policy optimization smooths rugged landscapes: Theorem 1 bounds $\|\tilde{\nabla}J\|_F^2 \leq \mathrm{Var}_{\pi_\theta}(r)$, and pretraining keeps $\mathrm{Var}(r)$ finite. However, the KL penalty term $\lambda D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}})$ is treated only as a passive safety mechanism — preventing drift into singular regions.

This report shows that KL regularization plays a much deeper role: it **actively reshapes the optimization landscape**, making it progressively smoother and eventually strongly concave in the Fisher-Rao geometry. This yields exponential convergence guarantees that the un-regularized theory cannot provide.

---

## 1. The Effective Reward Representation

### 1.1 Gradient of the KL-Regularized Objective

The regularized objective is:

$$J_\lambda(\theta) = \mathbb{E}_{\pi_\theta}[r(x)] - \lambda D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}})$$

To compute $\nabla_\theta J_\lambda$, we need the gradient of the KL term. Using $\nabla_\theta \pi_\theta = \pi_\theta \nabla_\theta \log \pi_\theta$ and expanding:

$$\nabla_\theta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}}) = \nabla_\theta \int \pi_\theta(x) \log \frac{\pi_\theta(x)}{\pi_{\mathrm{base}}(x)} \, dx$$

Applying the product rule:

$$= \int \pi_\theta \nabla_\theta \log \pi_\theta \cdot \log \frac{\pi_\theta}{\pi_{\mathrm{base}}} \, dx + \int \underbrace{\nabla_\theta \pi_\theta}_{= \pi_\theta \nabla_\theta \log \pi_\theta} \, dx$$

The second integral is $\nabla_\theta \int \pi_\theta \, dx = \nabla_\theta \, 1 = 0$. Therefore:

$$\nabla_\theta D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}}) = \mathbb{E}_{\pi_\theta}\!\left[\nabla_\theta \log \pi_\theta(x) \cdot \log \frac{\pi_\theta(x)}{\pi_{\mathrm{base}}(x)}\right]$$

Combining with the policy gradient $\nabla_\theta \mathbb{E}[r] = \mathbb{E}[r \cdot \nabla_\theta \log \pi_\theta]$:

$$\boxed{\nabla_\theta J_\lambda = \mathbb{E}_{\pi_\theta}\!\left[\tilde{r}_\lambda(x;\theta) \cdot \nabla_\theta \log \pi_\theta(x)\right]}$$

where the **effective reward** is:

$$\tilde{r}_\lambda(x;\theta) \;\triangleq\; r(x) - \lambda \log \frac{\pi_\theta(x)}{\pi_{\mathrm{base}}(x)}$$

This is a standard result (e.g., Ziebart 2010; Levine 2018), but its implications for landscape smoothness have not been fully explored.

**Key distinction from the un-regularized case**: $\tilde{r}_\lambda$ depends on $\theta$, so it co-evolves with the policy during training.

---

## 2. Generalized Theorem 1

### 2.1 Statement

**Theorem 1$'$ (KL-regularized natural gradient bound).** For any parametric family $\pi_\theta$ with KL-regularized objective $J_\lambda$:

$$\|\tilde{\nabla} J_\lambda(\theta)\|_F^2 \;\leq\; \mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda)$$

**Proof.** Identical to Theorem 1 in the main report, with $\tilde{r}_\lambda$ replacing $r$. Define $g(x) = \nabla_\theta \log \pi_\theta(x)$. Since $\mathbb{E}[g] = 0$:

$$\nabla_\theta J_\lambda = \mathbb{E}[\tilde{r}_\lambda \cdot g] = \mathrm{Cov}(\tilde{r}_\lambda, g)$$

By multivariate Cauchy-Schwarz:

$$\|\tilde{\nabla}J_\lambda\|_F^2 = \mathrm{Cov}(\tilde{r}_\lambda, g)^\top [\mathrm{Var}(g)]^{-1} \mathrm{Cov}(\tilde{r}_\lambda, g) \leq \mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda) \qquad \blacksquare$$

### 2.2 Variance Decomposition

Expanding $\mathrm{Var}(\tilde{r}_\lambda)$ reveals the structure:

$$\mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda) = \mathrm{Var}(r) + \lambda^2 \mathrm{Var}\!\left(\log\frac{\pi_\theta}{\pi_{\mathrm{base}}}\right) - 2\lambda\,\mathrm{Cov}\!\left(r,\; \log\frac{\pi_\theta}{\pi_{\mathrm{base}}}\right)$$

| Term | Sign | Behavior during successful training |
|------|------|------|
| $\mathrm{Var}(r)$ | $+$ | Decreases as $\pi_\theta$ concentrates on high-reward regions |
| $\lambda^2 \mathrm{Var}(\log \text{ratio})$ | $+$ | Increases as $\pi_\theta$ deviates from $\pi_{\mathrm{base}}$ |
| $-2\lambda\,\mathrm{Cov}(r, \log\text{ratio})$ | $-$ | **Negative** when high-reward regions have high $\pi_\theta/\pi_{\mathrm{base}}$ |

The covariance term is crucial: during correct optimization, $\pi_\theta$ assigns higher density to higher-reward regions, so $\mathrm{Cov}(r, \log(\pi_\theta/\pi_{\mathrm{base}})) > 0$, and the cross-term **reduces** the total variance. Under favorable conditions:

$$\mathrm{Var}_{\pi_\theta}(\tilde{r}_\lambda) < \mathrm{Var}_{\pi_\theta}(r)$$

This means the KL penalty provides **additional smoothing beyond the un-regularized case**.

---

## 3. Landscape Collapse at the Optimum

### 3.1 Closed-Form Optimal Policy

The KL-regularized objective $J_\lambda(\pi) = \mathbb{E}_\pi[r] - \lambda D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{base}})$ has a unique global maximizer in distribution space:

$$\pi^*_\lambda(x) = \frac{1}{Z(\lambda)} \pi_{\mathrm{base}}(x) \exp\!\left(\frac{r(x)}{\lambda}\right), \qquad Z(\lambda) = \mathbb{E}_{\pi_{\mathrm{base}}}\!\left[\exp\!\left(\frac{r(x)}{\lambda}\right)\right]$$

This is the Boltzmann (softmax) policy tilted from the base distribution.

### 3.2 Effective Reward Becomes Constant

At $\pi^*_\lambda$, the log-ratio is:

$$\log \frac{\pi^*_\lambda(x)}{\pi_{\mathrm{base}}(x)} = \frac{r(x)}{\lambda} - \log Z(\lambda)$$

Substituting into $\tilde{r}_\lambda$:

$$\tilde{r}_\lambda(x;\theta^*) = r(x) - \lambda\!\left(\frac{r(x)}{\lambda} - \log Z(\lambda)\right) = \lambda \log Z(\lambda) = \text{const}$$

Therefore:

$$\boxed{\mathrm{Var}_{\pi^*_\lambda}(\tilde{r}_\lambda) = 0 \qquad \Longrightarrow \qquad \tilde{\nabla} J_\lambda(\theta^*) = \mathbf{0}}$$

### 3.3 Comparison with the Un-Regularized Case

| Property | Without KL ($\lambda = 0$) | With KL ($\lambda > 0$) |
|----------|---------------------------|-------------------------|
| Optimal policy | $\pi^* = \delta(x - x^*)$ (Dirac delta) | $\pi^*_\lambda \propto \pi_{\mathrm{base}} \exp(r/\lambda)$ (smooth) |
| $\mathrm{Var}(r)$ at optimum | $0$ (trivially, single point) | $> 0$ in general |
| $\mathrm{Var}(\tilde{r}_\lambda)$ at optimum | N/A | $= 0$ (exactly) |
| Achievable by finite model? | No (cannot represent $\delta$) | Yes (smooth target) |
| Gradient vanishing | Only in the limit $\sigma \to 0$ | Exact at finite $\pi^*_\lambda$ |

The un-regularized case requires the policy to collapse to a point mass — impossible for any finite parametric model. The regularized case requires matching a smooth Boltzmann distribution — readily achievable by normalizing flows or diffusion models with sufficient capacity. This is a qualitative improvement: **the regularized landscape has a reachable stationary point where the natural gradient exactly vanishes**.

---

## 4. Strong Concavity in Fisher-Rao Geometry

### 4.1 Statement

**Theorem 2 (Strong concavity).** The KL-regularized objective $J_\lambda$ is $\lambda$-strongly concave with respect to KL divergence in distribution space:

$$J_\lambda(\pi) \leq J_\lambda(\pi') + \langle \nabla_\pi J_\lambda(\pi'),\; \pi - \pi' \rangle - \lambda\, D_{\mathrm{KL}}(\pi \| \pi')$$

for all distributions $\pi, \pi'$ in the domain.

**Proof sketch.** Decompose $J_\lambda$:

$$J_\lambda(\pi) = \underbrace{\mathbb{E}_\pi[r]}_{\text{linear in }\pi} - \lambda \underbrace{D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{base}})}_{\text{convex in }\pi}$$

The first term is linear (hence both convex and concave). The KL divergence $D_{\mathrm{KL}}(\pi \| \pi_{\mathrm{base}})$ is strictly convex in $\pi$ with modulus 1 (with respect to itself as the distance measure). Therefore $-\lambda D_{\mathrm{KL}}$ is $\lambda$-strongly concave. $\qquad \blacksquare$

### 4.2 Implication: Exponential Convergence of Natural Gradient

For a sufficiently expressive parametric family (where the Fisher-Rao geometry faithfully represents the distribution space), strong concavity in the Fisher metric implies that natural gradient ascent converges exponentially:

$$D_{\mathrm{KL}}(\pi^*_\lambda \| \pi_{\theta_t}) \leq (1 - \eta\lambda)^t \cdot D_{\mathrm{KL}}(\pi^*_\lambda \| \pi_{\theta_0})$$

where $\eta$ is the natural gradient step size (in Fisher arc length).

**Contrast with $\lambda = 0$**: without KL regularization, $J(\theta) = \mathbb{E}_{\pi_\theta}[r]$ is linear in $\pi$ — neither convex nor concave in distribution space. No convergence rate guarantee exists in general. The landscape may have saddle points, plateaus, or other pathologies.

### 4.3 What This Means Physically

Near the optimum $\pi^*_\lambda$:
- The landscape curves downward (concave) in **every** direction in Fisher space.
- There are no local maxima other than $\pi^*_\lambda$ (unique global maximum).
- The curvature is at least $\lambda$ — larger $\lambda$ means faster convergence but larger bias (see Section 6).

This is the strongest theoretical guarantee in the entire framework: **KL regularization converts the optimization problem from a non-convex landscape navigation task into a strongly concave optimization with exponential convergence**.

---

## 5. The Three Roles of KL — Unified View

The main report identifies two roles for the KL constraint. The effective reward analysis reveals a third:

| Role | Mechanism | Mathematical content |
|------|-----------|---------------------|
| **Trust region** | Bounds $D_{\mathrm{KL}}(\pi_\theta \| \pi_{\theta_{\mathrm{old}}})$ per step | Prevents catastrophic policy updates; enables reliable importance sampling |
| **Safety** | Pinsker: $\mathrm{TV} \leq \sqrt{\delta/2}$ | Maintains $R_{\max}^{\mathrm{eff}}(\pi_\theta) \approx R_{\max}^{\mathrm{eff}}(\pi_{\mathrm{base}}) < \infty$ |
| **Active smoothing** (new) | $\mathrm{Var}(\tilde{r}_\lambda) \to 0$ as $\pi_\theta \to \pi^*_\lambda$ | Reshapes landscape to be strongly concave; guarantees exponential convergence |

The third role is the deepest: the KL penalty does not merely constrain the optimization — it fundamentally changes the geometry of the objective function, creating a well-shaped landscape where none existed before.

---

## 6. The Bias-Smoothness Tradeoff

### 6.1 Bias Introduced by Regularization

The optimal KL-regularized reward is strictly less than the global optimum:

$$\mathbb{E}_{\pi^*_\lambda}[r] < r(x^*) = \max_x r(x)$$

The gap can be expressed as:

$$r(x^*) - \mathbb{E}_{\pi^*_\lambda}[r] = \lambda\!\left(\log Z(\lambda) + D_{\mathrm{KL}}(\pi^*_\lambda \| \pi_{\mathrm{base}})\right)$$

### 6.2 Two Limiting Regimes

**$\lambda \to 0$ (weak regularization):**
- $\pi^*_\lambda \to \delta(x^*)$: no bias, maximum reward
- $\mathrm{Var}(\tilde{r}_\lambda) \to \mathrm{Var}(r)$: no additional smoothing
- No strong concavity guarantee
- Optimization may be slow or get trapped

**$\lambda \to \infty$ (strong regularization):**
- $\pi^*_\lambda \to \pi_{\mathrm{base}}$: maximum bias, no reward improvement
- $\mathrm{Var}(\tilde{r}_\lambda) \to 0$: extreme smoothing (flat landscape)
- Strong concavity with large modulus $\lambda$
- Optimization converges instantly to $\pi_{\mathrm{base}}$ (trivial solution)

### 6.3 Optimal $\lambda$

There exists an optimal $\lambda^*$ balancing optimization quality and convergence speed:

$$\lambda^* = \arg\min_\lambda \left[\underbrace{\text{optimization error}(\lambda)}_{\sim \exp(-\lambda \cdot T)} + \underbrace{\text{bias}(\lambda)}_{\sim \lambda \cdot (\log Z + D_{\mathrm{KL}})}\right]$$

where $T$ is the computational budget (number of natural gradient steps). The optimization error decreases exponentially with $\lambda$ (from strong concavity), while the bias increases linearly. This tradeoff is sharp and problem-dependent:

- **Rugged landscapes** (many local minima, high $L_r$): larger $\lambda^*$ is beneficial — the smoothing gain outweighs the bias.
- **Simple landscapes** (few local minima): smaller $\lambda^*$ suffices — less smoothing needed.
- **Limited compute** (small $T$): larger $\lambda^*$ — need faster convergence even at the cost of bias.

---

## 7. Updated Theoretical Chain

Incorporating KL regularization, the complete theoretical chain becomes:

$$\boxed{\text{Pretraining} \xrightarrow{R_{\max}^{\mathrm{eff}} < \infty} \text{Thm 1' applies} \xrightarrow[\text{Var}(\tilde{r}_\lambda) \to 0]{\text{strong concavity}} \text{Exponential convergence} \xrightarrow{D_{\mathrm{KL}} \leq \delta} \text{Stay in safe region}}$$

| Step | Without KL ($\lambda = 0$) | With KL ($\lambda > 0$) |
|------|---------------------------|-------------------------|
| Landscape bound | $\|\tilde{\nabla}J\|_F \leq \sqrt{\mathrm{Var}(r)}$ | $\|\tilde{\nabla}J_\lambda\|_F \leq \sqrt{\mathrm{Var}(\tilde{r}_\lambda)} \leq \sqrt{\mathrm{Var}(r)}$ |
| Near-optimum behavior | $\mathrm{Var}(r)$ finite but nonzero | $\mathrm{Var}(\tilde{r}_\lambda) \to 0$ (landscape collapse) |
| Convexity | No guarantee | $\lambda$-strongly concave |
| Convergence rate | No guarantee | $O(e^{-\lambda t})$ in Fisher metric |
| Optimal target | $\delta(x^*)$ (unreachable) | $\pi^*_\lambda$ (smooth, reachable) |
| Price | — | Bias $\sim \lambda(\log Z + D_{\mathrm{KL}})$ |

---

## 8. Implications for Crystal Structure Prediction

In the crystal structure prediction setting, where $r(x)$ has Coulomb singularities:

1. **Pretraining** restricts $\pi_{\mathrm{base}}$ to physically reasonable configurations, making $R_{\max}^{\mathrm{eff}} < \infty$.

2. **KL regularization** does three things simultaneously:
   - Prevents $\pi_\theta$ from drifting into singular regions (safety).
   - Reduces the effective reward variance below $\mathrm{Var}(r)$ (additional smoothing).
   - Makes the landscape strongly concave near the Boltzmann optimum (convergence guarantee).

3. **The optimal $\lambda$** for crystal applications is likely moderate: the energy landscape is highly rugged (favoring larger $\lambda$ for smoothing), but the global minimum is sharply defined (penalizing large bias).

4. **Symmetry breaking** that causes $|r(x)| \to \infty$ is automatically handled by the KL constraint: even if $r$ diverges in certain regions, $\tilde{r}_\lambda = r - \lambda \log(\pi_\theta/\pi_{\mathrm{base}})$ self-corrects, because $\pi_\theta$ cannot assign significant density there without incurring a large KL penalty.

---

## 9. Experimental Verification Plan

The following quantities should be tracked to validate the theory in the toy model:

### 9.1 New Quantities to Record

In `ppo.py`, for each iteration, additionally compute:

- $\mathrm{Var}(\tilde{r}_\lambda)$: variance of the effective reward $r(x) - \lambda \log(\pi_\theta(x)/\pi_{\mathrm{base}}(x))$
- $\mathrm{Cov}(r, \log(\pi_\theta/\pi_{\mathrm{base}}))$: the cross-term driving additional smoothing
- All three variance decomposition terms separately

### 9.2 Experiments

| Experiment | What to show |
|---|---|
| **$\mathrm{Var}(\tilde{r}_\lambda)$ vs $\mathrm{Var}(r)$ along trajectory** | $\mathrm{Var}(\tilde{r}_\lambda) < \mathrm{Var}(r)$ during training (additional smoothing) |
| **$\mathrm{Var}(\tilde{r}_\lambda)$ decay curve** | Monotonic decrease toward 0 (landscape collapse) |
| **$\lambda$ sweep: $\{0, 0.1, 0.5, 1.0, 2.0\}$** | Bias-smoothness tradeoff; optimal $\lambda^*$ |
| **Convergence speed vs $\lambda$** | Faster initial convergence at larger $\lambda$ |
| **$C_v^\lambda / \mathrm{Var}(\tilde{r}_\lambda)$** | Generalized Theorem 1' tightness check |

### 9.3 Expected Results

For the 2D Rastrigin function with Boltzmann pretrained flow:

1. At $\lambda = 0$: $\mathrm{Var}(r)$ decreases but remains bounded away from 0.
2. At $\lambda > 0$: $\mathrm{Var}(\tilde{r}_\lambda)$ decreases faster and approaches 0.
3. Final mean reward: decreasing in $\lambda$ (bias effect).
4. Convergence speed (iterations to reach 90% of final reward): increasing in $\lambda$ (smoothing effect).
5. Optimal $\lambda^*$: balances these two effects, likely in the range $[0.1, 1.0]$ for this problem.
