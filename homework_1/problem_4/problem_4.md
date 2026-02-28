## Problem 4

Let $P = \text{Positive}$, $D = \text{Disease}$

Given:

$$P(P|D) = 0.99$$
$$P(D) = 0.0001$$
$$P(P|\overline{D}) = 0.01$$
$$P(\overline{D}) = 0.9999$$

Then:
$$P(P) = P(P|D)P(D) + P(P|\overline{D})P(\overline{D})$$
$$= (0.99)(0.0001) + (0.01)(0.9999)$$
$$= 0.010098$$

Then:
$$P(D|P) = \frac{P(P|D)P(D)}{P(P)}$$
$$= \frac{(0.99)(0.0001)}{0.010098}$$
$$= 0.0098$$

Thus: 0.98% chance of having the disease.
