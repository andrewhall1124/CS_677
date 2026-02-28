## Problem 6

**(a)**

$$\begin{aligned}
\text{Ber}(y|\theta) &= \theta^y(1-\theta)^{1-y}\\
\log(\text{Ber}(y|\theta)) &= \log\left(\theta^y(1-\theta)^{1-y}\right) \\
&= \log(\theta^y) + \log((1-\theta)^{1-y}) \\
&= y\log(\theta) + (1-y)\log(1-\theta) \\
-\log(\text{Ber}(y|\theta)) &= -[y\log(\theta) + (1-y)\log(1-\theta)]
\end{aligned}$$

**(b)**

$$\begin{aligned}
\text{Cat}(y|\theta) &= \prod_{j=1}^{K} \theta_j^{y_j} \\
\log(\text{Cat}(y|\theta)) &= \log\left(\prod_{j=1}^{K} \theta_j^{y_j}\right) \\
&= \sum_{j=1}^{K} \log(\theta_j^{y_j}) \\
&= \sum_{j=1}^{K} y_j \log(\theta_j) \\
-\log(\text{Cat}(y|\theta)) &= -\sum_{j=1}^{K} y_j \log(\theta_j)
\end{aligned}$$
