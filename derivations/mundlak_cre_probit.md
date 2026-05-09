# Correlated Random Effects Probit: The Mundlak Device

**Reference:** Mundlak (1978), *Econometrica* 46(1), 69–85.

> Mundlak (1978) derives the device for a *linear* model. The extension to probit follows Wooldridge (2010, Ch. 15).

---

## The problem with standard panel estimators

The basic panel model with individual effects is (Mundlak eq. 2.1):

$$
Y = X\beta + Z\alpha + u, \qquad u \sim (0,\, \sigma^2 I_n)
$$

where $Z\alpha$ collects the individual fixed effects. Two standard approaches fail for binary outcomes:

- **Fixed effects**: sweeping out $\alpha_i$ by within-transformation works in linear models but not nonlinear ones — estimating $N$ individual intercepts alongside $\beta$ leads to the incidental parameters problem (Neyman & Scott 1948), causing inconsistency as $N \to \infty$ with $T$ fixed.
- **Random effects**: assumes $\text{Cov}(\alpha_i, X_{it}) = 0$, which is implausible in labour market panels where unobserved ability or preferences correlate with education, income, and work hours. Imposing this yields inconsistent estimates of $\beta$.

---

## The Mundlak projection

Mundlak's solution is to model the correlation between $\alpha_i$ and $X_{it}$ explicitly, rather than assuming it away or trying to eliminate it. He introduces the auxiliary regression of the individual effect onto the individual time-means (his eq. 2.4):

$$
\alpha_i = X_{i\cdot}\pi + w_i
$$

where $X_{i\cdot} = T^{-1}\sum_{t=1}^T X_{it}$ is the vector of individual time-means, $\pi$ is the vector of projection coefficients, and $w_i$ is the projection residual. The projection coefficient is defined as the population OLS minimiser:

$$
\pi \equiv \bigl[E(X_{i\cdot}^{\top} X_{i\cdot})\bigr]^{-1} E(X_{i\cdot}^{\top} \alpha_i)
$$

By the normal equations of the linear projection, $E[X_{i\cdot}^{\top} w_i] = 0$ holds by construction — no distributional assumption on $\alpha_i$ is required. Mundlak (p. 72) notes that $\pi = 0$ if and only if the effects are uncorrelated with the regressors, i.e. the RE assumption holds.

---

## The augmented model

Substituting the projection into the model yields:

$$
Y = X\beta + X_{i\cdot}\pi + w_i + u_{it}
$$

This is a standard regression on the augmented regressor set $(X_{it},\, X_{i\cdot})$. For the linear case, Mundlak (his eq. 3.3) shows that the estimator of $\beta$ from this augmented model equals the within estimator — RE and FE coincide once the auxiliary regression is included.

---

## Extension to probit

For a binary outcome we work with the latent variable model:

$$
y_{it}^* = X_{it}\beta + \alpha_i + \varepsilon_{it}, \qquad y_{it} = \mathbf{1}[y_{it}^* > 0], \qquad \varepsilon_{it} \overset{iid}{\sim} N(0,1)
$$

Applying the Mundlak projection gives the augmented latent equation:

$$
y_{it}^* = X_{it}\beta + X_{i\cdot}\pi + w_i + \varepsilon_{it}
$$

The composite error $e_{it} = w_i + \varepsilon_{it}$ is normal if $w_i \sim N(0, \omega^2)$, since the normal distribution is closed under convolution. This is why CRE applies to probit but not logit — a normal plus a logistic is not logistic.

After normalising $\text{Var}(e_{it}) = 1$, the model reduces to a **pooled probit** on $(X_{it},\, X_{i\cdot})$, estimated by MLE. Since observations within individual $i$ share $w_i$, cluster-robust standard errors (clustered at the individual level) are required.

Average marginal effects are reported rather than raw coefficients, since the latter are scaled by $1/\sqrt{\omega^2 + 1}$ and not directly interpretable in probability units.

---

## Specification test and interpretation

The joint Wald test $H_0: \pi = 0$ tests whether the RE assumption is tenable. Rejection confirms that unobserved heterogeneity is correlated with the regressors and the Mundlak augmentation is necessary.

| Coefficient | Interpretation |
|---|---|
| $\hat{\beta}_k$ | Within-individual effect, controlling for time-invariant heterogeneity |
| $\hat{\pi}_k$ | Difference between between- and within-individual effects |
| $\hat{\beta}_k + \hat{\pi}_k$ | Between-individual (long-run) effect |

---

## References

- Mundlak, Y. (1978). On the pooling of time series and cross section data. *Econometrica*, 46(1), 69–85.
- Wooldridge, J. M. (2010). *Econometric Analysis of Cross Section and Panel Data* (2nd ed.), Ch. 15. MIT Press.
- Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika*, 73(1), 13–22.
- Neyman, J., & Scott, E. L. (1948). Consistent estimates based on partially consistent observations. *Econometrica*, 16(1), 1–32.
