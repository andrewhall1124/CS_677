## Problem 2.1: MLE for the Bernoulli/Binomial Model

**Given:** The Bernoulli PMF is $p(D|\theta) = \theta^{N_1}(1-\theta)^{N_0}$ where $N = N_0 + N_1$

**Derive:** $\hat{\theta}_{MLE} = \frac{N_1}{N}$

**Solution:**

Write the log-likelihood:
$$\log p(D|\theta) = \log(\theta^{N_1}(1-\theta)^{N_0})$$

$$= \log(\theta^{N_1}) + \log((1-\theta)^{N_0})$$

$$= N_1 \log(\theta) + N_0 \log(1-\theta)$$

Take the derivative with respect to $\theta$:
$$\frac{d}{d\theta} \log p(D|\theta) = \frac{N_1}{\theta} - \frac{N_0}{1-\theta}$$

Set equal to zero:
$$0 = \frac{N_1}{\theta} - \frac{N_0}{1-\theta}$$

$$\frac{N_1}{\theta} = \frac{N_0}{1-\theta}$$

Cross-multiply:
$$N_1(1-\theta) = N_0\theta$$

$$N_1 - N_1\theta = N_0\theta$$

$$N_1 = (N_0 + N_1)\theta$$

$$N_1 = N\theta$$

Therefore:
$$\boxed{\theta = \frac{N_1}{N}}$$



