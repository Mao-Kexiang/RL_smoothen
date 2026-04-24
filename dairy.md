# 2026.4.03
I start the trial on visualizing parameters space by drawing a 2D subspace, and figure out that we should use FIM (Fisher Information Matrix) to avoid the ambiguous parameter settings.

Here, the main ambiguous thing is the definition of roughness. Many people believes that pretraining a model actually finds a good representation of the dataset (or the problem), and the parameter space is a better space compared to the orgin space.
So one can do RL to finetune the paramaters of the pretrained network to find the solution of the optimization problem.
And why it works? People usually argues that parameter space is more smooth than origin space. And we want to visualize this.
However, if one zooms in any surface seems smooth, it will discover roughness details (skin as an example). And one can also do the oppposite to make some surface seems smooth.

Which means if in $\theta$ parameter space, a surface is roughness. One can find it smooth use $100\theta$.

$f(x)=\sin (100 x)$, use $y = 100x$ then $g(y)=\sin y = f(x)$

To avoid this kind of ambiguous. FIM will help, because

$$F(\theta)=\mathbb{E}_\theta(\nabla_\theta \log \pi_\theta(x)\nabla_\theta \log \pi_\theta(x)^T)$$

And if $\theta$ is scaled by $\alpha$, $\varphi=\alpha\theta$ then:

$${\rm d}\theta ^2 \to {\rm d}\varphi ^2 = \alpha ^2 {\rm d}\theta ^2$$

while

$${\rm d}\theta^T F(\theta) {\rm d}\theta \to {\rm d}\varphi^T F(\varphi) {\rm d}\varphi = \alpha{\rm d}\theta^T \frac{1}{\alpha^2}F(\theta) \alpha{\rm d}\theta ={\rm d}\theta^T F(\theta) {\rm d}\theta $$

We get a dimensionless (scaling free) view of the parameter space.

# 2026.4.10

The first version code (toy model) of the idea above is realized. Using PPO to rl a mlp. The aim is to generate better x s.t. Rastrigin function get minimized.

$$f(x)= A \cdot d + \sum_{i=1}^d (x^2_i-A\cos(2\pi x_i))$$

Our toy model sets $A=10, d=2$

The base model is a mlp convert a $z\sim \mathcal{N}(0,\mathbb{I}_8)$ to a $(\mu_1, \mu_2, \log \sigma_1, log \sigma_2 ) \in \mathcal{R}^4$, so the mlp predicts a Guassian distribution $\pi _\theta(x|z)$

$$\pi_\theta(x|z) = \frac{1}{2\pi\sigma_1\sigma_2}\exp\Big(-\frac{(x_1-\mu_1(z;\theta))^2}{2\sigma^2_1(z;\theta)}-\frac{(x_2-\mu_2(z;\theta))^2}{2\sigma^2_2(z;\theta)}\Big)$$

And the whole model is:

$$\pi_\theta(x)=\int p(z)\pi_\theta(x|z){\rm d}z$$

# 2026.4.13
The base model actually is a GMM. Advisor suggests that this model is ambigous when considering $\nabla_\theta \log \pi_\theta(x)$

pretrain:

$$\mathcal{L} = -\sum_{i\in D}\log \pi(x_i) = -\sum_{i\in D}\log \Big(\int p(z)\pi_\theta(x_i|z){\rm d}z\Big)$$

When doing SGD (actually Adam), we should calculate the gradient:

$$\nabla_\theta \log \Big(\int p(z)\pi_\theta(x_i|z){\rm d}z\Big) = \frac{\int p(z)\nabla_\theta\pi_\theta(x_i|z){\rm d}z}{\int p(z)\pi_\theta(x_i|z){\rm d}z}$$

The intergal is hard for the 8 dimension, we have to sample a lot of points in the 8 dimensional space and do forward propogation to get the parameters. And gadients even needs BP to do it. To slow.

To deal this, we usually choose to optimize the upper bound of L.

$$- \log \Big(\int p(z)\pi_\theta(x_i|z){\rm d}z\Big) \leq - \int p(z) \log \pi_\theta(x_i|z){\rm d}z$$

And now the gradient becomes to 

$$\int p(z)\nabla_\theta \log\pi_\theta(x_i|z){\rm d}z$$

still should use sampling to get the gradient, though $\nabla_\theta\log\pi_\theta(x_i|z)$ can be calculated, easier than $\nabla_\theta\pi_\theta(x_i|z)$

And even worse, we have to use $\nabla_\theta \log \pi_\theta(x)$ to do PPO, but the code always uses $\nabla_\theta \log \pi_\theta(x|z)$. The error of sampling always impacts the iters, which makes results ambigous.

# 2026.4.20
Change the base model to Normalizing Flow Model, using RealNVP to realize it.




# 2026.4.24

