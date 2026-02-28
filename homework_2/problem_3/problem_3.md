## Problem 2.3: MLE for the Poisson Distribution

**Given:** $\text{Poi}(x|\lambda) = e^{-\lambda}\frac{\lambda^x}{x!}$ for $x \in \{0, 1, 2, ...\}$ where $\lambda > 0$

**Derive:** The MLE for $\lambda$

**Solution:**

Write the log-likelihood:
$$\log(\text{Poi}(x|\lambda)) = \log\left(e^{-\lambda}\frac{\lambda^x}{x!}\right)$$

$$= -\lambda + \log\left(\frac{\lambda^x}{x!}\right)$$

$$= -\lambda + \log(\lambda^x) - \log(x!)$$

$$= -\lambda + x\log(\lambda) - \log(x!)$$

Take the derivative with respect to $\lambda$:
$$\frac{d}{d\lambda}\log(\text{Poi}(x|\lambda)) = -1 + \frac{x}{\lambda}$$

Set equal to zero:
$$-1 + \frac{x}{\lambda} = 0$$

$$\frac{x}{\lambda} = 1$$

Therefore:
$$\boxed{\lambda = x}$$