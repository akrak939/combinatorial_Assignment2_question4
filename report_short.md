# Assignment 2: Volatility

## 1 Introduction

Volatility is the most common measure of the risk of an asset and a key input in option pricing and Value-at-Risk. It cannot be observed directly, and the variance of financial returns is not constant over time: calm periods alternate with turbulent periods in which large returns follow each other, a pattern known as volatility clustering (Brooks, 2019, §9.1). Brooks (2019, Ch. 9) discusses several ways of measuring such time-varying volatility, and the aim of this report is to find out which of them works well in practice.

For this purpose, the daily returns of two ETFs listed on the SIX Swiss Exchange, IWDC.SW and SPICHA.SW, are analysed from January 2021 to December 2025. The sample includes the market turbulence after the US tariff announcement of 2 April 2025 ("Liberation Day"), which tests how quickly the models react to a sudden shock. Three baseline models are estimated: GARCH(1,1) (Bollerslev, 1986) and its special cases EWMA and RiskMetrics. They are complemented by the asymmetric GJR model (Glosten, Jagannathan and Runkle, 1993), and the models are compared on their parameters, fit and ability to track volatility. Finally, the correlation between the two series is analysed with a dynamic conditional correlation (DCC) model.

Section 2 describes the data, Section 3 the baseline models, and the subsequent sections present the advanced model, the evaluation and the correlation analysis.

## 2 Data

The raw data set, `etf_full-SW.csv`, contains daily closing prices of 2,314 ETFs listed on the SIX Swiss Exchange, for 1,280 dates from 4 January 2021 to 30 December 2025. The data are cleaned in three steps, using the script `extract_etf.py`.

First, since the assignment requires returns until at least June 2025, only ETFs with a price on the first sample date and a last price on or after 30 June 2025 are kept, which leaves 1,053 ETFs. Second, some ETFs display constant prices over specific periods, which results in daily returns of zero and makes their measured volatility lower than their true volatility. Every ETF for which more than 5% of the daily returns are exactly zero is therefore dropped, leaving 384 ETFs. Third, the 22 dates on which no ETF has a price are removed, leaving 1,258 trading days.

For every ETF $i$, the daily simple return in percent is computed as

$$
y_{i,t} = 100 \times \left( \frac{P_{i,t}}{P_{i,t-1}} - 1 \right),
$$

where $P_{i,t}$ is the closing price on trading day $t$. Prices are re-adjusted for dividends and splits at each year-end, so the first return of each year may reflect this adjustment rather than a real price change. To avoid artificial shocks in the volatility models, these returns are set to missing.

Of the 384 ETFs, 175 have the maximum number of 1,253 daily returns, and out of this group two ETFs were selected at random: **IWDC.SW** and **SPICHA.SW**. Both have returns until 30 December 2025, and their samples overlap completely. Their returns are stored in `2etfs_retuns.csv`, on which all further estimation is based.

**Table 1: Descriptive statistics of daily returns (in %)**

| Statistic | IWDC.SW | SPICHA.SW |
| --- | --- | --- |
| Observations | 1,253 | 1,253 |
| Mean | 0.042 | 0.026 |
| Standard deviation | 0.943 | 0.815 |
| Minimum | −4.866 | −5.168 |
| Maximum | 4.770 | 4.366 |
| Skewness | −0.41 | −0.53 |
| Excess kurtosis | 2.84 | 4.70 |

Table 1 shows that both series are negatively skewed and have clearly fatter tails than the normal distribution. Such leptokurtosis, together with volatility clustering, cannot be explained by models with a constant variance (Brooks, 2019, §9.1), which motivates the models of the next section. The two series have a sample correlation of 0.73.

## 3 Baseline models

Following the course slides, the returns are modelled by a regression equation for the mean and a GARCH(1,1) equation for the variance (Brooks, 2019, Box 9.2),

$$
y_t = x_t'\beta + u_t, \qquad u_t \sim \mathcal{N}(0, \sigma_t^2), \qquad
\sigma_{t+1}^2 = \omega + \alpha u_t^2 + \delta \sigma_t^2 ,
$$

where the only regressor is a constant, $x_t = 1$ and $\beta = \mu$, which gives the assignment's specification $y_t \sim \mathcal{N}(\mu, \sigma_t^2)$. The parameter $\alpha$ measures the reaction of the variance to new shocks and $\delta$ its persistence. The restrictions $\omega > 0$, $0 < \alpha, \delta < 1$ and $\alpha + \delta < 1$ keep the variance positive and ensure the unconditional variance $\mathrm{E}\,\sigma^2 = \omega / (1 - \alpha - \delta)$, to which the variance reverts (Brooks, 2019, eq. 9.40).

The EWMA model with decay factor $\lambda$ (Brooks, 2019, §9.5) is

$$
\sigma_{t+1}^2 = (1 - \lambda) \sum_{j=0}^{\infty} \lambda^j u_{t-j}^2 = (1 - \lambda)\, u_t^2 + \lambda \sigma_t^2 ,
$$

which is a GARCH(1,1) model with $\omega = 0$, $\alpha = 1 - \lambda$ and $\delta = \lambda$, the restrictions of the assignment. It is an integrated GARCH model without mean reversion. RiskMetrics fixes $\lambda = 0.94$ and, following the assignment, $\mu = \bar{y}$, so that no parameter is left to estimate.

**Table 2: Baseline models as special cases of GARCH(1,1)**

| Model | Restrictions | Estimated parameters |
| --- | --- | --- |
| GARCH(1,1) | $\omega > 0$, $0 < \alpha, \delta < 1$, $\alpha + \delta < 1$ | $\omega, \alpha, \delta, \mu$ |
| EWMA | $\omega = 0$, $\alpha = 1 - \delta$, $\delta = \lambda$ | $\delta, \mu$ |
| RiskMetrics | $\omega = 0$, $\delta = \lambda = 0.94$, $\alpha = 0.06$, $\mu = \bar{y}$ | none |

The parameters are estimated by maximum likelihood under conditional normality (Brooks, 2019, §9.9). Following the course slides, the log-likelihood is

$$
l_t = -\frac{1}{2} \left( \log 2\pi + \log \sigma_t^2 + \frac{u_t^2}{\sigma_t^2} \right), \qquad
l = \sum_{t=1}^{T} l_t .
$$

Following the course example code, $-l/T$ is minimised with the BFGS algorithm. The restrictions are imposed by transformation: $\omega$ is estimated as $\log \omega$, and $\alpha$ and $\delta$ on the logit scale, which keeps them between zero and one. The joint restriction $\alpha + \delta < 1$ cannot be imposed in this way and is enforced by assigning an infinite objective value to inadmissible parameter values.

Since the likelihood may have local maxima, starting values away from zero are used (Brooks, 2019, Box 9.3): $\mu_0 = \bar{y}$, $(\alpha_0, \delta_0) = (0.05, 0.90)$ and $\omega_0 = s^2(1 - \alpha_0 - \delta_0)$, with $s^2 = \frac{1}{T}\sum_{t=1}^{T}(y_t - \bar{y})^2$ the sample variance. The optimisation is repeated from $(0.10, 0.85)$ and $(0.03, 0.95)$, and for EWMA from $\delta_0 = 0.94$, $0.90$ and $0.97$; the run with the highest log-likelihood is kept.

The GARCH filter is initialised at $\sigma_1^2 = \omega / (1 - \alpha - \delta)$. For EWMA and RiskMetrics this expression is undefined, so both start at $\sigma_1^2 = s^2$ rather than at the first squared return, which is a very noisy variance estimate. After estimation, the filter is run at the optimal parameters to obtain the conditional variances $\hat{\sigma}_t^2$, which are evaluated later in the report.

## References

Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics* 31.3, pp. 307–327.

Brooks, C. (2019). *Introductory Econometrics for Finance*. 4th ed. Cambridge: Cambridge University Press.

Glosten, L. R., R. Jagannathan and D. E. Runkle (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance* 48.5, pp. 1779–1801.
