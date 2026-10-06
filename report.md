# Assignment 2: Volatility

## 1 Introduction

Volatility is one of the central concepts in finance. It is the most common measure of the risk of an asset and a key input in option pricing and in risk measures such as Value-at-Risk. Unlike returns, however, volatility cannot be observed directly and has to be estimated from the data. A well-known feature of financial returns makes this task both important and challenging: their variance is not constant over time. Calm periods with small price changes alternate with turbulent periods in which large positive and negative returns follow each other, a pattern known as volatility clustering (Brooks, 2019, §9.1). A model that assumes a constant variance therefore underestimates risk in turbulent times and overestimates it in calm times.

Brooks (2019, Ch. 9) discusses several ways of measuring and modelling such time-varying volatility, ranging from simple weighting schemes to models estimated by maximum likelihood. The aim of this report is to find out which of these measures works well in practice. For this purpose, the daily returns of two exchange-traded funds (ETFs) listed on the SIX Swiss Exchange, IWDC.SW and SPICHA.SW, are analysed over the period from January 2021 to December 2025. This sample covers several episodes of market stress, among them the turbulence that followed the announcement of US import tariffs on 2 April 2025 ("Liberation Day"), which provides a natural test of how quickly the models react to a sudden shock.

Three baseline models are estimated for each series: the GARCH(1,1) model of Bollerslev (1986) and two restricted special cases of it, the exponentially weighted moving average (EWMA) model and the RiskMetrics model. These are complemented by an advanced asymmetric model, the GJR model of Glosten, Jagannathan and Runkle (1993), which allows negative shocks to raise volatility more than positive shocks of the same size. The models are compared in terms of their estimated parameters, their fit and their ability to track the observed volatility, both over the full sample and around Liberation Day. Finally, the correlation between the two series is investigated, first over sub-samples and then with a dynamic conditional correlation (DCC) model.

The report is structured as follows. Section 2 describes the data, the cleaning procedure and the selection of the two ETFs. Section 3 introduces the baseline models and their estimation. The subsequent sections present the advanced model, the evaluation of all models and the correlation analysis.

## 2 Data

### 2.1 Data source

The analysis is based on ETFs listed on the SIX Swiss Exchange. The raw data set, `etf_full-SW.csv`, contains daily closing prices of 2,314 ETFs, all identified by the ticker suffix .SW. The prices cover 1,280 dates, from 4 January 2021 to 30 December 2025. Since many ETFs were launched or delisted during this period, or are traded only rarely, the data are cleaned before the returns are computed, using the script `extract_etf.py`.

### 2.2 Data cleaning

First, the sample period is restricted. Since the assignment requires returns until at least June 2025, only ETFs that have a price on the first date of the sample (4 January 2021) and whose last available price falls on or after 30 June 2025 are kept. After this step, 1,053 of the original 2,314 ETFs remain.

Second, some of the ETFs display constant prices over specific periods of time, meaning that consecutive closing prices are the same, which results in daily returns of zero. Such zero returns make the measured volatility of an ETF lower than its true volatility, so these ETFs are not appropriate for volatility modelling. Therefore, for each ETF the daily returns are computed and those that are exactly zero are counted, and if their share of all daily returns exceeds 5%, the ETF is dropped. After this step, 384 ETFs remain.

Third, the data contain 22 dates on which none of the remaining ETFs has a recorded price, for instance exchange holidays. These dates carry no information and are removed, which leaves 1,258 trading days.

### 2.3 Construction of the returns

For every ETF $i$, the daily simple return in percent is computed as

$$
y_{i,t} = 100 \times \left( \frac{P_{i,t}}{P_{i,t-1}} - 1 \right),
$$

where $P_{i,t}$ is the closing price of ETF $i$ on trading day $t$. A return is only computed when the prices on both days are available.

Lastly, prices are re-adjusted for dividends and splits at each year-end, so the first return of each year may reflect this adjustment rather than a real price change. To avoid artificial shocks in the volatility models, these returns, on 3 January 2022, 2023, 2024 and 2025, are set to missing. The remaining returns are treated as a consecutive daily series.

### 2.4 Selection of two ETFs

After all cleaning steps were completed, the remaining ETFs were sorted in descending order by their number of daily returns. In total, 175 of the 384 ETFs have the maximum number of 1,253 daily returns, and out of this group two ETFs were selected at random: **IWDC.SW** and **SPICHA.SW**. Both IWDC.SW and SPICHA.SW have returns until 30 December 2025, and their samples overlap completely because both have a return on every trading day, making the comparison of their volatility models straightforward. The returns of the two ETFs are stored in a new file, `2etfs_retuns.csv`, and all subsequent estimation is based on this file only.

*[Open question: add the full fund names and investment focus of IWDC.SW and SPICHA.SW.]*

### 2.5 Descriptive statistics

Table 1 summarises the main properties of the two return series.

**Table 1: Descriptive statistics of daily returns (in %), 5 January 2021 – 30 December 2025**

| Statistic | IWDC.SW | SPICHA.SW |
| --- | --- | --- |
| Observations | 1,253 | 1,253 |
| Mean | 0.042 | 0.026 |
| Standard deviation | 0.943 | 0.815 |
| Minimum | −4.866 | −5.168 |
| Maximum | 4.770 | 4.366 |
| Skewness | −0.41 | −0.53 |
| Excess kurtosis | 2.84 | 4.70 |

For both ETFs, the average daily return is an order of magnitude smaller than the standard deviation, which is typical for daily returns. IWDC.SW is the more volatile series, while SPICHA.SW shows the larger extreme movements, with a minimum return of −5.17%.

Both series are negatively skewed and show clear excess kurtosis (2.84 and 4.70), so their distributions have fatter tails than the normal distribution. Brooks (2019, §9.1) lists such leptokurtosis, together with volatility clustering, among the stylised features of financial returns that models with a constant variance cannot explain, which motivates the conditional volatility models of the next section. Finally, the two series are strongly related, with a sample correlation of 0.73.

## 3 Baseline models

Three baseline models are estimated for each ETF: GARCH(1,1), EWMA and RiskMetrics. The latter two are restricted versions of GARCH(1,1), so all three share the same mean equation, variance equation and likelihood, and differ only in which parameters are estimated and which are fixed in advance.

### 3.1 GARCH(1,1)

Following the course slides, the model consists of a regression equation for the mean and a GARCH equation for the variance of the disturbances (Brooks, 2019, Box 9.2):

$$
y_t = x_t'\beta + u_t, \qquad u_t \sim \mathcal{N}(0, \sigma_t^2), \qquad
\sigma_{t+1}^2 = \omega + \alpha u_t^2 + \delta \sigma_t^2 .
$$

In this assignment the mean is constant, so the only regressor is a constant, $x_t = 1$, and $\beta = \mu$. The model then reduces to the specification of the assignment, $y_t \sim \mathcal{N}(\mu, \sigma_t^2)$ with $u_t = y_t - \mu$. In the notation of Brooks (2019, eq. 9.22), $\omega$, $\alpha$ and $\delta$ correspond to $\alpha_0$, $\alpha_1$ and $\beta$.

The parameter $\alpha$ measures how strongly the variance reacts to new shocks, and $\delta$ measures how persistent it is. Together they allow the model to capture volatility clustering (Brooks, 2019, §9.7). Taking expectations of the variance equation under weak stationarity gives the unconditional variance

$$
\mathrm{E}\,\sigma^2 = \frac{\omega}{1 - \alpha - \delta},
$$

to which the variance reverts after a shock (Brooks, 2019, eq. 9.40). This requires the restrictions $\omega > 0$, $0 < \alpha < 1$, $0 < \delta < 1$ and $\alpha + \delta < 1$, which also keep the variance positive (Brooks, 2019, §9.7.2 and §9.8.1). The GARCH(1,1) model has four parameters to estimate: $\omega$, $\alpha$, $\delta$ and $\mu$.

### 3.2 EWMA and RiskMetrics

The EWMA model weights past squared shocks with exponentially declining weights. Following Brooks (2019, §9.5) and the course slides, with decay factor $\lambda$,

$$
\sigma_{t+1}^2 = (1 - \lambda) \sum_{j=0}^{\infty} \lambda^j u_{t-j}^2 = (1 - \lambda)\, u_t^2 + \lambda \sigma_t^2 ,
$$

where the second form follows by substituting the recursion for $\sigma_t^2$. Comparing this with the GARCH equation shows that EWMA is a GARCH(1,1) model with $\omega = 0$, $\alpha = 1 - \lambda$ and $\delta = \lambda$, which are exactly the restrictions of the assignment, $\omega = 0$ and $\alpha = 1 - \delta$. Since $\alpha + \delta = 1$, EWMA is an integrated GARCH (IGARCH) model and has no unconditional variance to revert to. Two parameters remain to be estimated, $\delta$ and $\mu$; Brooks (2019, §9.5 and §9.22.3) notes that the decay factor is often fixed but can also be estimated by maximum likelihood.

The RiskMetrics model is an EWMA model with the decay factor fixed at $\lambda = 0.94$, the value recommended for daily data, so that $\alpha = 0.06$ and $\delta = 0.94$. Brooks (2019) and the course slides describe RiskMetrics with a mean return of zero; following the assignment, the mean is instead set to the sample mean, $\mu = \bar{y}$. As a result, RiskMetrics has no parameters to estimate. Table 2 summarises the three models.

**Table 2: Baseline models as special cases of GARCH(1,1)**

| Model | Restrictions | Estimated parameters |
| --- | --- | --- |
| GARCH(1,1) | $\omega > 0$, $0 < \alpha, \delta < 1$, $\alpha + \delta < 1$ | $\omega, \alpha, \delta, \mu$ |
| EWMA | $\omega = 0$, $\alpha = 1 - \delta$, $\delta = \lambda$ | $\delta, \mu$ |
| RiskMetrics | $\omega = 0$, $\delta = \lambda = 0.94$, $\alpha = 0.06$, $\mu = \bar{y}$ | none |

### 3.3 Estimation

The parameters are estimated by maximum likelihood under conditional normality (Brooks, 2019, §9.9). Following the course slides, the log-likelihood contribution of day $t$ and the total log-likelihood are

$$
l_t = -\frac{1}{2} \left( \log 2\pi + \log \sigma_t^2 + \frac{u_t^2}{\sigma_t^2} \right), \qquad
l = \sum_{t=1}^{T} l_t ,
$$

which is the time-varying variance version of Brooks (2019, eq. 9.43). For given parameter values, the regression filter computes the residuals $u_t = y_t - x_t'\beta$, the GARCH filter runs the variance equation through the sample to obtain $\sigma_1^2, \dots, \sigma_T^2$, and finally $l$ is evaluated. Following the course example code, the negative average log-likelihood, $-l/T$, is minimised with the BFGS algorithm.

The parameter restrictions are imposed by transformation, so that the optimiser can search over unrestricted values without leaving the admissible region. The constant $\omega$ is estimated as $\log \omega$, which keeps it positive, and $\alpha$ and $\delta$ are estimated on the logit scale, $\log(x / (1 - x))$, which keeps them between zero and one. The logit transformation of $\delta$ is added to the course example code, which transforms only $\omega$ and $\alpha$. The joint restriction $\alpha + \delta < 1$ cannot be imposed by transforming each parameter separately, since two parameters that each lie between zero and one can still sum to more than one. It is therefore enforced directly in the objective function, which returns an infinite value for inadmissible parameter values, so that the optimiser moves back into the admissible region. For EWMA, this check is not needed, since $\alpha + \delta = 1$ holds by construction.

Since the likelihood of GARCH models may have local maxima, the starting values matter (Brooks, 2019, §9.9.1). Following Brooks (2019, Box 9.3), the mean starts at the regression estimate $\mu_0 = \bar{y}$, and the variance parameters start at plausible values away from zero, $(\alpha_0, \delta_0) = (0.05, 0.90)$, with $\omega_0 = s^2(1 - \alpha_0 - \delta_0)$, where

$$
s^2 = \frac{1}{T} \sum_{t=1}^{T} (y_t - \bar{y})^2
$$

is the sample variance, so that the implied unconditional variance equals $s^2$. As the course slides advise restarting the optimisation when local maxima are a concern, the estimation is repeated from $(\alpha_0, \delta_0) = (0.10, 0.85)$ and $(0.03, 0.95)$, and for EWMA from $\delta_0 = 0.94$, $0.90$ and $0.97$. The run with the highest log-likelihood is kept.

### 3.4 Initial variance and extraction

The variance equation is recursive, so a value for $\sigma_1^2$ has to be chosen; it is not prescribed by the assignment or the course material. For GARCH(1,1), the filter is initialised at the unconditional variance, $\sigma_1^2 = \omega / (1 - \alpha - \delta)$, as in the course example code. For EWMA and RiskMetrics this expression is undefined, since $\omega = 0$ and $\alpha + \delta = 1$. Both are therefore initialised at the sample variance, $\sigma_1^2 = s^2$, rather than at the first squared return used in the course's EWMA example, because a single squared return is a very noisy estimate of the variance. Using the same initial value also keeps RiskMetrics an exact special case of EWMA.

After estimation, the variance equation is run once more at the optimal parameters, which yields the conditional variance $\hat{\sigma}_t^2$ for every trading day, ETF and model. These series form the basis of the evaluation of the models later in the report. All models are implemented in Python in the script `baseline.py`, using NumPy, pandas and SciPy.

## References

Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics* 31.3, pp. 307–327.

Brooks, C. (2019). *Introductory Econometrics for Finance*. 4th ed. Cambridge: Cambridge University Press.

Glosten, L. R., R. Jagannathan and D. E. Runkle (1993). On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks. *Journal of Finance* 48.5, pp. 1779–1801.
