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

# 2026.4.24

