# From KL Divergence to Fisher Information: Why Policy Optimization Smooths Rugged Landscapes

## 1. The Core Question

Given an objective function $r(x)$ over a design space $x \in \mathbb{R}^n$, the standard approach is direct optimization:

$$\max_{x} r(x)$$

When $r$ is highly non-convex (e.g., crystal formation energy with many local minima), gradient-based methods get trapped and sampling-based methods scale poorly. Reinforcement learning (RL) takes a fundamentally different path: instead of optimizing $x$ directly, we parameterize a probability distribution $\pi_\theta(x)$ and optimize its parameters $\theta \in \mathbb{R}^D$:

$$\max_{\theta}    J(\theta) = \mathbb{E}_{x \sim \pi_\theta}[r(x)] = \int r(x)   \pi_\theta(x)   dx$$

This raises an immediate question: **why should lifting a low-dimensional problem into a vastly higher-dimensional parameter space help?** The answer lies in the interplay between KL divergence, Fisher information geometry, and the trust-region mechanics of PPO.

---

## 2. KL Divergence: Measuring Distribution Change

### 2.1 Definition

The Kullback-Leibler divergence from distribution $q$ to $p$ is:

$$D_{\mathrm{KL}}(p \lVert q) = \int p(x) \log \frac{p(x)}{q(x)}     dx = \mathbb{E}_{x \sim p}  \left[\log \frac{p(x)}{q(x)}\right]$$

Key properties:

- $D_{\mathrm{KL}}(p \lVert q) \geq 0$, with equality iff $p = q$ a.e.
- **Not symmetric**: $D_{\mathrm{KL}}(p \lVert q) \neq D_{\mathrm{KL}}(q \lVert p)$ in general.
- **Not a metric**, but its local structure induces one (the Fisher metric).

### 2.2 KL Divergence Between Nearby Distributions (FIM)

Consider a parametric family $\pi_\theta$. For an infinitesimal perturbation $\theta \to \theta + \delta\theta$, expand $\log \pi_{\theta+\delta\theta}(x)$ to second order:

$$\log \pi_{\theta+\delta\theta}(x) = \log \pi_\theta(x) + \delta\theta^\top \nabla_\theta \log \pi_\theta(x) + \frac{1}{2}   \delta\theta^\top \nabla_\theta^2 \log \pi_\theta(x)   \delta\theta + O(\|\delta\theta\|^3)$$

Substituting into the KL divergence $D_{\mathrm{KL}}(\pi_\theta \lVert \pi_{\theta+\delta\theta})$:

$$D_{\mathrm{KL}}(\pi_\theta   \lVert \pi_{\theta+\delta\theta}) = -\mathbb{E}_{\pi_\theta}  \left[\delta\theta^\top \nabla_\theta \log \pi_\theta(x) + \frac{1}{2}   \delta\theta^\top \nabla^2_\theta \log \pi_\theta(x)   \delta\theta\right] + O(\|\delta\theta\|^3)$$

The first-order term vanishes because the score function has zero mean:

$$\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(x)] = \int \pi_\theta(x) \cdot \frac{\nabla_\theta \pi_\theta(x)}{\pi_\theta(x)}   dx = \nabla_\theta   \int \pi_\theta(x)   dx = \nabla_\theta    1 = 0$$

For the second-order term, use the identity (derived by differentiating $\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$):
Start from $\mathbb{E}[\nabla_\theta \log \pi_\theta] = 0$
and differentiate both sides with respect to $\theta$:

$$
\begin{align}
0 &= \nabla_\theta \int \pi_\theta(x)   \nabla_\theta \log \pi_\theta(x)   dx\\
&= \int \pi_\theta(x)  \left[\nabla_\theta \log \pi_\theta(x)   \nabla_\theta \log \pi_\theta(x)^\top + \nabla^2_\theta \log \pi_\theta(x)\right]dx \\
&= \mathbb{E}_{\pi_\theta} \left[\nabla_\theta \log \pi_\theta  \nabla_\theta \log \pi_\theta^\top\right] + \mathbb{E}_{\pi_\theta} \left[\nabla^2_\theta \log \pi_\theta\right]
\end{align}
$$

$$\implies\mathbb{E}_{\pi_\theta}  \left[-\nabla^2_\theta \log \pi_\theta(x)\right] = \mathbb{E}_{\pi_\theta}  \left[\nabla_\theta \log \pi_\theta(x)  \nabla_\theta \log \pi_\theta(x)^\top\right]$$

Set

$$F(\theta) = \mathbb{E}_{x\sim\pi_\theta}  \left[\nabla_\theta \log \pi_\theta(x)  \nabla_\theta \log \pi_\theta(x)^\top\right]$$

Therefore:

$$\boxed{D_{\mathrm{KL}}(\pi_\theta   \lVert \pi_{\theta+\delta\theta}) = \frac{1}{2}   \delta\theta^\top F(\theta)   \delta\theta + O(\|\delta\theta\|^3)}$$

where $F(\theta)$ is the **Fisher Information Matrix**.

$F(\theta)$ is a $D \times D$ positive semi-definite matrix. It encodes the **intrinsic geometry of parameter space**: how much the output distribution changes per unit parameter perturbation.

### 2.3 FIM as a Riemannian Metric Tensor

The second-order expansion of KL divergence reveals that $F(\theta)$ serves as a **Riemannian metric tensor** on the parameter manifold. The infinitesimal "distance" between $\pi_\theta$ and $\pi_{\theta+d\theta}$ is:

$$ds^2 = d\theta^\top F(\theta)   d\theta$$

This is the **Fisher-Rao metric**. It turns the parameter space into a Riemannian manifold where distances measure statistical distinguishability rather than Euclidean displacement.

**Geodesic distance** (the length of the shortest path between two distributions in this geometry) approximates $\sqrt{2   D_{\mathrm{KL}}}$ locally, giving a principled notion of "how far apart two policies are."

---

## 3. Why KL Divergence Is the Right Lens for PPO

### 3.1 The Trust-Region Idea

Policy gradient methods update $\theta$ by following $\nabla_\theta J(\theta)$. But naive gradient ascent is unreliable: a large step in $\theta$ may cause a catastrophic change in behavior. The key insight of trust-region methods is to constrain the **distribution change**, not the parameter change:

$$\max_\theta    J(\theta) \quad \text{subject to} \quad D_{\mathrm{KL}}(\pi_{\theta_{\mathrm{old}}} \lVert \pi_\theta) \leq \delta$$

This constraint is expressed in KL divergence because:

1. **Euclidean constraints are meaningless in policy space.** Two parameter vectors $\theta_A$ and $\theta_B$ with $\|\theta_A - \theta_B\| = \epsilon$ might produce identical distributions (redundant parameterization) or wildly different ones. KL divergence measures actual behavioral change.

2. **KL divergence controls the validity of importance sampling.** The policy ratio $\rho = \pi_\theta(x)/\pi_{\theta_{\mathrm{old}}}(x)$ must stay bounded for the surrogate objective to be a reliable estimate of $J(\theta)$. By Pinsker's inequality, bounding KL bounds the total variation, which bounds how extreme $\rho$ can get.

### 3.2 From TRPO to PPO

**TRPO** (Trust Region Policy Optimization) solves the constrained problem above using a second-order approximation. The constraint $D_{\mathrm{KL}} \leq \delta$ is locally equivalent to:

$$\delta\theta^\top F(\theta_{\mathrm{old}})   \delta\theta \leq 2\delta$$

This is an ellipsoidal constraint shaped by the Fisher matrix. TRPO computes $F^{-1}\nabla J$ (the **natural gradient**) via conjugate gradient methods.

**PPO** replaces the hard KL constraint with two practical mechanisms:

**(a) Clipped surrogate objective:**

$$L^{\mathrm{CLIP}}(\theta) = \mathbb{E}  \left[\min  \Big(\rho   \hat{A},  \mathrm{clip}(\rho,   1{-}\epsilon,   1{+}\epsilon)   \hat{A}\Big)\right]$$

where:

$$\rho = \frac{\pi_\theta(x)}{\pi_{\theta_{\mathrm{old}}}(x)}, \qquad \hat{A} = \frac{r - \bar{r}}{\mathrm{std}(r) + \varepsilon}$$

The clipping at $1 \pm \epsilon$ implicitly constrains $\rho$, which constrains how far $\pi_\theta$ can drift from $\pi_{\theta_{\mathrm{old}}}$.

**(b) KL penalty (used in fine-tuning variants):**

$$L(\theta) = \mathbb{E}  \left[\rho   \hat{A}\right] - \lambda   D_{\mathrm{KL}}(\pi_\theta \lVert \pi_{\theta_{\mathrm{base}}})$$

where $\lambda$ controls how far the fine-tuned policy can deviate from the pretrained base. This is directly a Lagrangian relaxation of the trust-region constraint.

### 3.3 The Connection: PPO Performs Approximate Natural Gradient Descent

Within the trust region, the optimal update direction is the **natural gradient**:

$$\tilde{\nabla}_\theta J = F(\theta)^{-1}   \nabla_\theta J$$

This is not ordinary gradient ascent in Euclidean space — it is gradient ascent along the steepest direction **as measured by the Fisher-Rao metric**. PPO's clipping mechanism approximates this: by preventing $\rho$ from deviating too far, each PPO step approximately follows the natural gradient within an adaptive trust region.

The equivalence can be seen from the Lagrangian of the trust-region problem:

$$\max_{\delta\theta}  \delta\theta^\top \nabla_\theta J - \frac{\lambda}{2}   \delta\theta^\top F(\theta)   \delta\theta$$

Taking the gradient and setting to zero:

$$\nabla_\theta J - \lambda   F(\theta)   \delta\theta^* = 0 \implies \delta\theta^* = \frac{1}{\lambda}   F(\theta)^{-1}   \nabla_\theta J = \frac{1}{\lambda}   \tilde{\nabla}_\theta J$$

### 3.4 The norm of Natural Gradient Descent

**Remark: Why $F^{-1}$ and not $F^{-2}$ in the norm.** A natural question is why $\|\tilde{\nabla}J\|_F^2 = (\nabla J)^\top F^{-1}(\nabla J)$ rather than $(\nabla J)^\top F^{-2}(\nabla J)$. The answer is that the natural gradient $\tilde{\nabla}J = F^{-1}\nabla J$ is a tangent vector on a Riemannian manifold, and its length must be measured with the Riemannian inner product $\langle u, v \rangle_F = u^\top F v$, not the Euclidean one. Expanding:

$$\|\tilde{\nabla}J\|_F^2 = (F^{-1}\nabla J)^\top  \cdot  F  \cdot  (F^{-1}\nabla J) = \nabla J^\top \underbrace{F^{-1} \cdot F \cdot F^{-1}}_{= F^{-1}} \nabla J$$

So the norm of Natural Gradient Descent is:

$$\|\tilde{\nabla}J\|_F^2 = \nabla J^\top F^{-1} \nabla J$$

---

## 4. Why the Lifted Landscape Is Smoother

### 4.1 The Policy Gradient as a Weighted Average

The policy gradient theorem gives:

$$\nabla_\theta J(\theta) = \mathbb{E}_{x \sim \pi_\theta}  \left[r(x)   \nabla_\theta \log \pi_\theta(x)\right]$$

This gradient is a **global weighted average** over the distribution $\pi_\theta$, not a local derivative $\nabla_x r(x)$ at a single point. Even if $r$ has sharp local gradients pointing toward nearby traps, the averaged gradient tends toward globally high-reward regions.

### 4.2 Theorem (Natural Gradient Bound via Cauchy-Schwarz)

**Theorem 1.** For any parametric family $\pi_\theta$ and bounded reward $|r(x)| \leq R_{\max}$:

$$\|\tilde{\nabla}_\theta J(\theta)\|_F^2   \leq   \mathrm{Var}_{\pi_\theta}(r)   \leq   \max \mathcal{supp}_{\pi_\theta(x)}  r^2(x) \leq R^2_{\max}$$

**Proof.** Define the score vector $g(x) = \nabla_\theta \log \pi_\theta(x)$. We have $\mathbb{E}[g] = 0$ and $F = \mathrm{Var}(g) = \mathbb{E}[gg^\top]$. (proved in 2.2)

By the policy gradient theorem:

$$\nabla_\theta J = \mathbb{E}[r \cdot g] = \mathrm{Cov}(r, g)$$

Because when $\mathbb{E}[g] = 0$, we have:

$$ \mathrm{Cov}(r, g) =  \mathbb{E}[(r- \mathbb{E}[r]) \cdot g] = \mathbb{E}[r \cdot g] - \mathbb{E}[r]\mathbb{E}[g] = \mathbb{E}[r \cdot g]$$

The natural gradient norm squared is:

$$\|\tilde{\nabla}J\|_F^2 = (\nabla J)^\top F^{-1}(\nabla J) = \mathrm{Cov}(r, g)^\top   [\mathrm{Var}(g)]^{-1}   \mathrm{Cov}(r, g)$$

By the multivariate Cauchy-Schwarz inequality (equivalently, $\mathrm{Var}(r \mid g) \geq 0$): https://zhuanlan.zhihu.com/p/2005644725047279995

$$\mathrm{Cov}(r, g)^\top   [\mathrm{Var}(g)]^{-1}   \mathrm{Cov}(r, g)   \leq   \mathrm{Var}(r) \qquad \blacksquare$$

**Interpretation.** Along Fisher-Rao geodesics (one unit of $\sqrt{D_{\mathrm{KL}}}$), the rate of change of $J$ is bounded by $\mathrm{Var}_ {\pi_\theta}(r)$ . This quantity depends on the **distribution** $\pi_\theta$, not just on $r$ alone.

**If $\pi_\theta(x)$ is good enough to avoid the unbonded area of r(x) and restrict x sampled from it has a small |r(x)|, then the inequality guarantees RL smoothen the landscape!**

**Smoothing gain:**

$$\eta = \frac{L_r}{\sup_\theta \|\tilde{\nabla}_\theta J\|_F}   \geq   \frac{L_r}{\sqrt{\mathrm{Var}(r)}}$$

For high-frequency oscillations ($L_r \to \infty$ with bounded $\mathrm{Var}(r)$ ), $\eta \to \infty$.

---

## 5. The Complete Picture: Why PPO Works on Rugged Objectives -- A GOOD BASE MODEL

### 5.1 The Problem with Global Bounds: Singular Objectives

The theorems above assume $|r(x)| \leq R_{\max}$ or $\|\nabla r\| \leq L_r$ globally. In many real applications — notably crystal structure generation — neither condition holds. Coulomb potentials $V \propto 1/|r_{ij}|$ diverge when atoms overlap, and interatomic repulsion $\propto 1/|r_{ij}|^{12}$ makes $L_r = \infty$ near singularities. If taken at face value, $R_{\max} = \infty$ renders Theorem 1's amplitude corollary and vacuous.

In csp, the symmetry broken also makes $|r(x)|\to\infty$, which makes RL less useful.

### 5.2 Pretraining Provides Effective Bounds

The resolution lies in a key observation: **Theorem 1 itself does not require $r$ to be globally bounded — it requires only $\mathrm{Var}_ {\pi_\theta}(r) < \infty$.** And a pretrained generative model ensures exactly this.

Define the **effective bound** of $r$ under a distribution $\pi$:

$$R_{\max}^{\mathrm{eff}}(\pi)  =  \mathrm{\max sup}_{x \sim \pi} |r(x)|$$

A base model pretrained on known crystal structures learns to generate physically plausible configurations: atoms at reasonable separations, lattice parameters within stable ranges. On the support of this pretrained distribution $\pi_{\mathrm{base}}$, interatomic distances satisfy $|r_{ij}| > r_{\min} > 0$, so:

$$R_{\max}^{\mathrm{eff}}(\pi_{\mathrm{base}})  \ll  R_{\max}^{\mathrm{global}} = \infty$$

All theorems then apply with $R_{\max}^{\mathrm{eff}}$ replacing $R_{\max}$:

$$\|\tilde{\nabla}J\|_F \leq \sqrt{\mathrm{Var}_{\pi_\theta}(r)} \leq R_{\max}^{\mathrm{eff}}(\pi_\theta)$$

The effective bound is not just finite — it is **small**, because the pretrained model concentrates on configurations where formation energies span a modest range (typically a few eV/atom). This is where the smoothing theorems acquire real force.

### 5.3 KL Constraints Maintain Effective Bounds During Training

PPO's KL penalty serves a dual role:

$$L(\theta) = \mathbb{E}[\rho \hat{A}] - \lambda D_{\mathrm{KL}}(\pi_\theta \| \pi_{\theta_{\mathrm{base}}})$$

**Role 1 (trust region):** Controls the rate of policy change, preventing premature collapse of the distribution width $\sigma$ and maintaining smoothness.

**Role 2 (safety):** Keeps $\pi_\theta$ close to $\pi_{\mathrm{base}}$ in distribution space, ensuring that $\pi_\theta$ does not drift into regions where $r$ diverges. Since $D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}}) \leq \delta$ implies $\mathrm{TV}(\pi_\theta, \pi_{\mathrm{base}}) \leq \sqrt{\delta/2}$ (Pinsker), the fine-tuned policy cannot assign significant probability to configurations far outside the pretrained support. This guarantees:

$$R_{\max}^{\mathrm{eff}}(\pi_\theta)  \approx  R_{\max}^{\mathrm{eff}}(\pi_{\mathrm{base}})$$

throughout training. The effective bound is inherited from pretraining and preserved by the KL constraint — RL never needs to "discover" that singular configurations are bad, because the base model already avoids them.

### 5.4 The Virtuous Cycle: Pretraining, Smoothing, and RL

The complete mechanism is a three-stage pipeline where each stage enables the next:

**Stage 1: Pretraining $\to$ effective bounds.** Training on known crystal data restricts $\pi_{\mathrm{base}}$ to physically reasonable configurations. This makes $R_{\max}^{\mathrm{eff}}$ finite and small, and makes $\mathrm{Var}_ {\pi_\theta}(r)$ well-controlled.

**Stage 2: Effective bounds $\to$ smooth landscape.** With finite $R_{\max}^{\mathrm{eff}}$, the smoothing theorems guarantee that $J(\theta)$ is well-behaved in FIM metric: $\|\tilde{\nabla}J\|_ F \leq R_{\max}^{\mathrm{eff}}$, gradients are reliable, and the landscape is navigable.

**Stage 3: Smooth landscape $\to$ RL succeeds.** PPO follows approximate natural gradients on this smooth landscape, with KL constraints preventing drift out of the safe region. The policy concentrates on higher-reward structures while remaining within the domain where the theorems apply.

$$\boxed{\text{Pretraining}  \xrightarrow{R_{\max}^{\mathrm{eff}} < \infty}  \text{Theorems apply}  \xrightarrow{\tilde{L}_J \leq R_{\max}^{\mathrm{eff}}}  \text{RL navigates smooth landscape}  \xrightarrow{D_{\mathrm{KL}} \leq \delta}  \text{Stay in safe region}}$$

Without pretraining, $R_{\max}^{\mathrm{eff}} = \infty$ and the smoothing guarantee collapses. Without KL constraints, the policy could drift into singular regions during training, violating the effective bound. Both are essential.

---
## 6. Summary of the Theoretical Chain

$$\boxed{\text{Pretraining}  \xrightarrow{R_{\max}^{\mathrm{eff}} < \infty}  J(\theta) = \mathbb{E}_{\pi_\theta}[r]  \xrightarrow{\text{FIM metric}}  \text{smooth landscape}  \xrightarrow{D_{\mathrm{KL}} \leq \delta}  \text{safe RL}}$$

| Step | Mathematical content | What it buys |
|---|---|---|
| Pretraining | $\pi_{\mathrm{base}}$ avoids singularities of $r$ | $R_{\max}^{\mathrm{eff}}(\pi) < \infty$; theorems become applicable |
| FIM from KL | $D_{\mathrm{KL}} \approx \frac{1}{2}\delta\theta^\top F(\theta) \delta\theta$ | Principled metric that measures actual distribution change |
| Natural gradient bound | $\|\tilde{\nabla}J\|_ F \leq \min(L_r\sigma\sqrt{n},  R_{\max}^{\mathrm{eff}})$ | FIM-roughness strictly less than Euclidean roughness |
| KL constraint in PPO | $D_{\mathrm{KL}}(\pi_\theta \| \pi_{\mathrm{base}}) \leq \delta$ | Controls annealing rate **and** maintains $R_{\max}^{\mathrm{eff}}$ |

The central message: **For objectives with singularities (like crystal energies), smoothing theorems are not unconditionally valid — they require the policy to stay in well-behaved regions. Pretraining provides this precondition by restricting the distribution to physically reasonable configurations, and KL-constrained RL preserves it. The power of RL fine-tuning emerges precisely at the intersection of a good base model and a trust-region optimizer: neither alone suffices.**


