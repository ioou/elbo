# elbo
elbo in vae
$$
用z的后验分布去拟合z的分布
min_{q(z)} KL(q(z) ||p(z|x) ) =E_{q(z)}\[log\frac{q(z)\cdot p(x)}{p(z|x)\cdot p(x)}] \\
 = E_{q(z)}\[log\frac{q(z) }{p(x|z)\cdot p(x)} + logp(x)]   \\
 = E_{q(z)}\[ -logp(x|z) + log\frac{q(z) }{p(x)}]+logp(x)  \\
左边括号中的为ELBO = -variational free energy \\
 ELBO = E_{q(z)}\[log\[p(x|z)] + log( \frac{p(z)} {q(z)} ] \\
 = E_{q(z)}\[log\[p(x|z)]  - KL(q(z)||p(z)) ]
$$

$ p(x)$: real distribution of images
$ p(z)$ : real distribution of latent variables.
architecture of vae Image -> ($z$, $\mu$),  ($z$, $\mu$)  -> Image
