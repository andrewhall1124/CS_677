## Problem 2.4: Bayesian Analysis of the Poisson Distribution

### Part 1: Derive the Posterior

**Given:** 
- Prior: $p(\lambda) = Ga(\lambda|a,b) \propto \lambda^{a-1}e^{-b\lambda}$
- Likelihood: $p(D|\lambda) = e^{-\lambda}\frac{\lambda^x}{x!} \propto e^{-\lambda}\lambda^x$

**Solution:**

By Bayes' theorem:
$$p(\lambda|D) \propto p(D|\lambda) \cdot p(\lambda)$$

$$p(\lambda|D) \propto \left(e^{-\lambda}\lambda^x\right) \cdot \left(\lambda^{a-1}e^{-b\lambda}\right)$$

$$\propto e^{-\lambda} \lambda^x \lambda^{a-1} e^{-b\lambda}$$

$$\propto e^{-(b+1)\lambda} \lambda^{a+x-1}$$

This is the kernel of a Gamma distribution:
$$\boxed{p(\lambda|D) = Ga(\lambda | a+x, b+1)}$$

### Part 2: Posterior Mean as $a \to 0$ and $b \to 0$

**Given:** The mean of $Ga(a,b)$ is $\frac{a}{b}$

**Solution:**

The posterior mean is:
$$E[\lambda|D] = \frac{a+x}{b+1}$$

Taking the limit:
$$\lim_{a \to 0, b \to 0} E[\lambda|D] = \lim_{a \to 0, b \to 0} \frac{a+x}{b+1} = \frac{0+x}{0+1} = \boxed{x}$$

This equals the MLE from Problem 2.3, showing that as the prior becomes uninformative, the posterior mean converges to the maximum likelihood estimate.