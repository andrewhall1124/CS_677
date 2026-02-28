## Problem 1

**Proposition:** $P(E_1|E_2) = \frac{P(E_2|E_1)P(E_1)}{P(E_2)}$

**Proof:** We will work directly.

Consider events $E_1, E_2$ where $P(E_1) \neq 0, P(E_2) \neq 0$.

By conditional probability:
$$P(E_1|E_2) = \frac{P(E_1 \cap E_2)}{P(E_2)}$$

By the multiplication rule:
$$P(E_1 \cap E_2) = P(E_1|E_2)P(E_2) = P(E_2|E_1)P(E_1)$$

Thus:
$$P(E_1|E_2) = \frac{P(E_1 \cap E_2)}{P(E_2)}$$
