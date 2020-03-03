[gitee]https://gitee.com/ioou/vaeelbo
# elbo
用模型预测的z的后验分布去近似实际的后验分布。两个版本都涉及到z的先验后验相似的问题。

1 原始版本 micheal jordan，这里的生硬的假设是，z的实际后验分布和x无关且各个变量之间独立。
$$
min_{q(z)} KL(q(z) ||p(z|x) ) =E_{q(z)}[log\frac{q(z)\cdot p(x)}{p(z|x)\cdot p(x)}] \\
 = E_{q(z)}[log\frac{q(z) }{p(x|z)\cdot p(z)} + logp(x)]   \\
 = E_{q(z)}[ -logp(x|z) + log\frac{q(z) }{p(z)}]+logp(x)  \\
左边括号中的为ELBO = -variational ~free ~energy \\
 = E_{q(z)}[log[p(x|z)] + log( \frac{p(z)} {q(z)} ] \\
 = E_{q(z)}[log[p(x|z)]  - KL(q(z)||p(z)) ]
 >=左边项
$$
2 这里的假设是z模型中的后验分布和z实际的先验分布是接近的，(最后一项)
$$
min_{q(z|x)} KL(q(z) ||p(z|x) ) =E_{q(z)}[log\frac{q(z)\cdot p(x)}{p(z|x)\cdot p(x)}] \\
 = E_{q(z|x)}[log\frac{q(z|x) }{p(x|z)\cdot p(z)} + logp(x)]   \\
 = E_{q(z|x)}[ -logp(x|z) + log\frac{q(z|x) }{p(z)}]+logp(x)  \\
左边括号中的为ELBO = -variational ~free ~energy \\
 = E_{q(z|x)}[log[p(x|z)] + log( \frac{p(z)} {q(z|x)} ] \\
 = E_{q(z|x)}[log[p(x|z)]  - KL(q(z|x)||p(z)) ]
 >=左边项
$$

loss function 

1对于交叉熵项，是elbo。为KLLoss，一般是z( $\mu$, $\sigma$)是去预测x的类别和方差，即预测标签。
2 对于左边项，为模型第二阶段有高斯假设，化简后为$(\frac{x-\mu_{x}}{  \sigma_{x}})^2$， 假设$\sigma_{x} = 1$ 。为了最小化这个项，使得模型输入输出一致，即重建损失。

$q(z)$:  z(latent variable)的预测分布。 
$p(x) $: x(Image)的实际分布。
$ p(z)$ : z的实际分布。

 x----$q_{\theta}(z|x)$------> ( $\mu$, $\sigma$)   $N(0,1)$ 采样-----$p_{\phi}(x|z)$-----> $\mu_{x},  \sigma_{x}$
 模型的优化就是去优化$\theta, \phi$

采样技巧：重参数化 -- 假设z的后验分布满足高斯分布，使用N(0,1)来采样$zz$，对$q_{\theta}(z|x)$ 预测的( $\mu$, $\sigma$) 进行重参数化便于反传训练，即$z_{sample} = \mu+zz*\sigma$   
