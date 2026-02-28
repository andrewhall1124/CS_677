## Problem 2.2: Posterior Predictive Distribution for the Beta-Binomial

**Given:** The posterior predictive distribution
$$p(x|n, D) = Bb(x|\alpha'_0, \alpha'_1, n) = \frac{B(x + \alpha'_1, n - x + \alpha'_0)}{B(\alpha'_1, \alpha'_0)} \binom{n}{x}$$

**Prove:** $Bb(1|\alpha'_1, \alpha'_0, 1) = \frac{\alpha'_1}{\alpha'_1 + \alpha'_0}$

**Hint:** $\Gamma(\alpha'_0 + \alpha'_1 + 1) = (\alpha_0 + \alpha_1)\Gamma(\alpha_0 + \alpha_1)$

**Solution:**

Substitute $n = 1$ and $x = 1$:
$$Bb(1|\alpha'_0, \alpha'_1, 1) = \frac{B(1 + \alpha'_1, 0 + \alpha'_0)}{B(\alpha'_1, \alpha'_0)} \binom{1}{1}$$

$$= \frac{B(1 + \alpha'_1, \alpha'_0)}{B(\alpha'_1, \alpha'_0)}$$

Using $B(a,b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$:

$$= \frac{\Gamma(1 + \alpha'_1)\Gamma(\alpha'_0)}{\Gamma(1 + \alpha'_1 + \alpha'_0)} \cdot \frac{\Gamma(\alpha'_1 + \alpha'_0)}{\Gamma(\alpha'_1)\Gamma(\alpha'_0)}$$

The $\Gamma(\alpha'_0)$ terms cancel:
$$= \frac{\Gamma(1 + \alpha'_1)}{\Gamma(\alpha'_1)} \cdot \frac{\Gamma(\alpha'_1 + \alpha'_0)}{\Gamma(1 + \alpha'_1 + \alpha'_0)}$$

Using $\Gamma(1 + \alpha'_1) = \alpha'_1 \Gamma(\alpha'_1)$ and the hint $\Gamma(\alpha'_0 + \alpha'_1 + 1) = (\alpha_0 + \alpha_1)\Gamma(\alpha_0 + \alpha_1)$:

$$= \frac{\alpha'_1 \Gamma(\alpha'_1)}{\Gamma(\alpha'_1)} \cdot \frac{\Gamma(\alpha'_1 + \alpha'_0)}{(\alpha'_1 + \alpha'_0)\Gamma(\alpha'_1 + \alpha'_0)}$$

$$= \alpha'_1 \cdot \frac{1}{\alpha'_1 + \alpha'_0}$$

$$= \boxed{\frac{\alpha'_1}{\alpha'_1 + \alpha'_0}}$$