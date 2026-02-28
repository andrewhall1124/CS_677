## Problem 5

Let $P = \text{Prize}$, $D = \text{Door Open}$

$$P(P|D_3) = \frac{P(D_3|P)P(P)}{P(D_3)}$$

$$\begin{aligned}
P(D_3) &= P(D_3|P_1)P(P_1) + P(D_3|P_2)P(P_2) + P(D_3|P_1)P(P_1) \\
&= (0)\left(\frac{1}{3}\right) + (1)\left(\frac{1}{3}\right) + \left(\frac{1}{2}\right)\left(\frac{1}{3}\right) \\
&= \frac{1}{2}
\end{aligned}$$

$$P(P_1|D_3) = \frac{\left(\frac{1}{2}\right)\left(\frac{1}{3}\right)}{\left(\frac{1}{2}\right)} = \frac{1}{3}$$

$$P(P_2|D_3) = \frac{(1)\left(\frac{1}{3}\right)}{\left(\frac{1}{2}\right)} = \frac{2}{3}$$

**(b)** Switch to Door 2 since $P(P_2|D_3) > P(P_1|D_3)$, i.e., $\frac{2}{3} > \frac{1}{3}$
