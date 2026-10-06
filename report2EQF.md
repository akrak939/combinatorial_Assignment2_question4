# Assignment 2: Volatility — Report

## 1 Data

### 1.1 Data source

The empirical analysis in this report is based on exchange-traded funds (ETFs) listed on the SIX Swiss Exchange. The raw data set, `etf_full-SW.csv`, contains daily closing prices for 2,314 ETFs, all identified by the ticker suffix .SW. The prices cover 1,280 dates, starting on 4 January 2021 and ending on 30 December 2025, and thereby span roughly five years of trading. This period includes several episodes of market stress, among them the inflation and interest-rate shock of 2022 and the turbulence following the announcement of US import tariffs on 2 April 2025 ("Liberation Day"). Such episodes make the sample well suited for a study of time-varying volatility.

Not every ETF in the raw file is suitable for this purpose. Many funds were launched or delisted during the sample period, so their price histories cover only part of it. Others are traded so rarely that their recorded price barely changes from day to day. Before any returns can be modelled, the data therefore have to be cleaned. All cleaning and extraction steps are carried out in the script `extract_etf.py` and are described below.

### 1.2 Data cleaning

The cleaning proceeds in four steps, each of which addresses a different problem in the raw prices.

**Coverage of the sample period.** The assignment requires both series to have returns until at least June 2025, and the models in this report benefit from a long and uninterrupted history. In the first step, we therefore keep only those ETFs that have a price on the very first date of the sample (4 January 2021) and whose last available price falls on or after 30 June 2025. Funds that started trading later, or stopped trading before mid-2025, are removed. After this step, 1,053 of the original 2,314 ETFs remain.

**Isolated price spikes.** Daily price data occasionally contain recording errors, in which a single price is far away from both its neighbours while the prices before and after it are close to each other. Such a pattern produces one very large positive and one very large negative return on consecutive days, which would distort any volatility model. We identify a price as an isolated spike when it differs by more than a factor three from both the previous and the next available price, while the previous and next prices differ from each other by less than a factor 1.5. These conditions are strict enough that genuine market moves, which do not immediately reverse, are not affected. The detected spikes are set to missing. In total, 10 prices in 10 different ETFs were removed in this way.

**Stale prices.** A fund that is traded only rarely often shows the same closing price on many consecutive days, because no new transaction took place. The resulting returns are exactly zero, which does not reflect the true variation in the value of the fund and would artificially lower its estimated volatility. We therefore compute, for each ETF, the share of daily returns that are exactly zero, and drop all funds for which this share exceeds 5%. This step removes a large number of illiquid funds and leaves 384 ETFs.

**Dates without any prices.** Finally, the data set contains 22 dates on which none of the remaining ETFs has a recorded price, for instance exchange holidays. These dates carry no information and are deleted, which leaves 1,258 trading days.

### 1.3 Construction of the returns

From the cleaned prices, daily returns are computed in percentage terms as simple returns,

$$
y_t = 100 \cdot \frac{P_t - P_{t-1}}{P_{t-1}},
$$

where $P_t$ denotes the closing price on trading day $t$. Expressing the returns in percent keeps the numbers in a convenient range, so that daily variances are of order one, and it follows the convention used in the course material.

One further adjustment is made. The return on the first trading day of each new year would measure the price change over the year-end holiday period rather than over a single trading day. These returns, on 3 January 2022, 2023, 2024 and 2025, are therefore set to missing, so that every return in the sample covers one regular trading day. The remaining returns are treated as a consecutive daily series.

### 1.4 Choice of the two ETFs

From the 384 cleaned ETFs we select two funds, IWDC.SW and SPICHA.SW, for the remainder of the analysis. The main criterion for this choice is the length and overlap of the available data. Both funds have a valid return on every one of the 1,253 trading days that remain after cleaning, which is the maximum number possible. Their samples therefore overlap completely, without a single missing observation, and both run until 30 December 2025, well beyond the June 2025 requirement. The final sample is also evenly spread over time, with between 248 and 253 returns in each calendar year from 2021 to 2025.

The returns of the two ETFs are stored in a new file, `2etfs_retuns.csv`. In line with the assignment, all subsequent estimation is based on this file only, so that the analysis can be reproduced from the return data alone.

*[Open question: add the full fund names and investment focus of IWDC.SW and SPICHA.SW.]*

### 1.5 Descriptive statistics

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

The average daily return is small for both funds, 0.042% for IWDC.SW and 0.026% for SPICHA.SW, and is an order of magnitude smaller than the daily standard deviation of 0.94% and 0.82%, respectively. This is typical for daily financial returns: over short horizons, the variation of returns dominates their mean. IWDC.SW is the more volatile of the two series, while SPICHA.SW shows the larger extreme movements, with a minimum daily return of −5.17%.

Both series are negatively skewed, which means that large negative returns occur more often than large positive returns of the same size. Both also show clear excess kurtosis, of 2.84 for IWDC.SW and 4.70 for SPICHA.SW, compared with zero for a normal distribution. The return distributions are therefore leptokurtic: they have more mass in the tails and around the mean than a normal distribution with the same variance. Brooks (2019, §9.1) lists leptokurtosis, together with volatility clustering, as one of the stylised features of financial returns that linear models with constant variance cannot explain. One explanation is that the variance itself changes over time, so that calm periods with small returns alternate with turbulent periods with large returns. This motivates the conditional volatility models introduced in the next section. Finally, the two series are strongly related, with a sample correlation of 0.73, which will be relevant for the correlation analysis later in the report.

## 2 Baseline models

### 2.1 Overview

To describe how the volatility of the two ETFs changes over time, we start from three baseline models: the GARCH(1,1) model, the exponentially weighted moving average (EWMA) model and the RiskMetrics model. The GARCH(1,1) model serves as the most general of the three. The EWMA and RiskMetrics models are obtained from it by fixing some of its parameters, so they can be seen as simplified special cases. This nesting is convenient both conceptually and for the implementation: all three models share the same mean equation, the same variance equation and the same likelihood function, and they differ only in which parameters are estimated and which are fixed in advance.

### 2.2 The GARCH(1,1) model

The generalised autoregressive conditional heteroskedasticity (GARCH) model was developed by Bollerslev (1986) and is discussed in Brooks (2019, §9.8). In its GARCH(1,1) form, as specified in the assignment, the return on day $t$ is assumed to be normally distributed around a constant mean $\mu$, with a variance $\sigma_t^2$ that changes from day to day:

$$
y_t \sim \mathcal{N}(\mu, \sigma_t^2), \qquad \sigma_{t+1}^2 = \omega + \alpha (y_t - \mu)^2 + \delta \sigma_t^2 .
$$

The model consists of two parts. The mean equation describes the expected return. Following the general set-up of a regression with GARCH disturbances (Brooks, 2019, Box 9.2), it can be written as a regression,

$$
y_t = x_t' \beta + u_t ,
$$

where $u_t$ is the residual or shock on day $t$. In this assignment the mean is constant, so the only regressor is a constant, $x_t = 1$, and the regression coefficient equals the mean, $\beta = \mu$. The residual is then simply the deviation of the return from its mean, $u_t = y_t - \mu$. Daily returns show little predictable structure in their mean, while their variance changes strongly over time, so a constant mean is a natural choice for the purpose of volatility modelling.

The variance equation states that tomorrow's variance is a weighted combination of three components: a constant $\omega$, the squared shock of today, $u_t^2$, and today's variance $\sigma_t^2$. The parameter $\alpha$ measures how strongly the variance reacts to new information: a large return of either sign raises the variance of the next day. The parameter $\delta$ measures how persistent the variance is: a high value of $\delta$ means that a period of high (or low) variance tends to last. Together, these two mechanisms allow the model to reproduce volatility clustering, the tendency of large returns to be followed by large returns and small returns by small returns (Brooks, 2019, §9.7).

Because the variance must be positive, the parameters have to satisfy $\omega > 0$, $\alpha \geq 0$ and $\delta \geq 0$ (Brooks, 2019, §9.7.2). In addition, we impose $\alpha + \delta < 1$. Under this condition the process is stationary in variance, and the variance fluctuates around a finite long-run level, the unconditional variance

$$
\mathrm{E}(\sigma_t^2) = \frac{\omega}{1 - \alpha - \delta}
$$

(Brooks, 2019, eq. 9.40). After a shock, the variance gradually returns to this level, at a speed that is determined by the persistence $\alpha + \delta$. This mean reversion is a well-documented property of financial volatility, and it is the main feature that distinguishes GARCH from the two simpler models below. The GARCH(1,1) model has four parameters to estimate: $\omega$, $\alpha$, $\delta$ and the mean $\mu$.

### 2.3 The EWMA model

The EWMA model (Brooks, 2019, §9.5) is obtained from GARCH(1,1) by imposing two restrictions, $\omega = 0$ and $\alpha = 1 - \delta$. The variance equation then becomes

$$
\sigma_{t+1}^2 = (1 - \delta)(y_t - \mu)^2 + \delta \sigma_t^2 .
$$

By repeatedly substituting for the lagged variance, $\sigma_{t+1}^2$ can be written as a weighted average of all past squared shocks, in which the weights $(1 - \delta)\delta^j$ decline exponentially with the lag $j$. The most recent observation therefore receives the largest weight, and the influence of older observations fades out gradually. The parameter $\delta$ plays the role of the decay factor, denoted $\lambda$ in Brooks (2019): the closer it is to one, the more slowly old information is forgotten.

Compared with GARCH, the EWMA model has a clear limitation. Since $\alpha + \delta = 1$ and $\omega = 0$, it corresponds to an integrated GARCH (IGARCH) model without a constant. Its variance therefore has no long-run level to return to, and forecasts from the model do not revert to an average variance (Brooks, 2019, §9.5). Whether this restriction matters in practice is one of the questions addressed later in the report. In the EWMA model, two parameters remain to be estimated: the decay factor $\delta$ and the mean $\mu$. Brooks (2019, §9.5 and §9.22.3) notes that the decay factor is often fixed but can also be estimated by maximum likelihood, which is the approach taken here.

### 2.4 The RiskMetrics model

The RiskMetrics model goes one step further and fixes all parameters in advance. It is an EWMA model with the decay factor set to $\delta = 0.94$, so that $\alpha = 0.06$, and with the mean set equal to the sample mean of the returns, $\mu = \bar{y}$. The value 0.94 was proposed by RiskMetrics for daily data and is widely used in practice (Brooks, 2019, §9.5). Brooks (2019) and the course slides describe RiskMetrics with a mean return of zero; we follow the assignment and use the sample mean instead. As a consequence, the RiskMetrics model has no parameters to estimate, and its variance can be computed directly from the data.

The relation between the three models is summarised in Table 2.

**Table 2: Baseline models as special cases of GARCH(1,1)**

| Model | Restrictions | Estimated parameters |
| --- | --- | --- |
| GARCH(1,1) | $\omega > 0$, $0 < \alpha < 1$, $0 < \delta < 1$, $\alpha + \delta < 1$ | $\omega, \alpha, \delta, \mu$ |
| EWMA | $\omega = 0$, $\alpha = 1 - \delta$, $0 < \delta < 1$ | $\delta, \mu$ |
| RiskMetrics | $\omega = 0$, $\delta = 0.94$, $\alpha = 0.06$, $\mu = \bar{y}$ | none |

### 2.5 Estimation by maximum likelihood

Because the variance equation is non-linear in the parameters and the variance is not directly observed, the models cannot be estimated by ordinary least squares. Instead, the parameters are estimated by maximum likelihood (Brooks, 2019, §9.9). Under the assumption that the returns are conditionally normally distributed, the contribution of day $t$ to the log-likelihood is

$$
\ell_t = -\frac{1}{2} \left( \log 2\pi + \log \sigma_t^2 + \frac{u_t^2}{\sigma_t^2} \right),
$$

and the log-likelihood of the full sample is the sum of these contributions, $\ell = \sum_{t=1}^{T} \ell_t$ (Brooks, 2019, eq. 9.43). For any given set of parameter values, the log-likelihood is evaluated in three steps. First, the regression filter computes the residuals of the mean equation, $u_t = y_t - x_t'\beta$, which here equal $y_t - \mu$. Second, the GARCH filter runs the variance equation through the sample to obtain the variances $\sigma_1^2, \dots, \sigma_T^2$. Third, the log-likelihood contributions are computed from the residuals and the variances. A numerical optimiser then searches for the parameter values that maximise $\ell$. In practice, we minimise the negative average log-likelihood, $-\ell / T$, using the quasi-Newton BFGS algorithm; dividing by $T$ does not change the optimum but keeps the objective on a convenient scale.

**Imposing the parameter restrictions.** The restrictions on the individual parameters are imposed by reparameterising the model, so that the optimiser can search over unrestricted values without ever leaving the admissible region. Instead of $\omega$, the optimiser works with $\log \omega$, which can take any real value while $\omega = \exp(\log \omega)$ is always positive. Similarly, $\alpha$ and $\delta$ are replaced by their logit transformations, $\log(\alpha / (1 - \alpha))$ and $\log(\delta / (1 - \delta))$, whose inverse always lies between zero and one. The mean $\mu$ is not restricted and is estimated directly. This follows the approach of the course example code, which transforms $\omega$ and $\alpha$ in the same way; we extend it with the logit transformation of $\delta$, so that all three restrictions $\omega > 0$, $0 < \alpha < 1$ and $0 < \delta < 1$ hold at every step of the search.

The remaining restriction for GARCH, $\alpha + \delta < 1$, involves two parameters jointly and cannot be imposed by transforming each parameter separately: even with $\alpha$ and $\delta$ each between zero and one, their sum can exceed one. It is therefore imposed directly in the objective function. Whenever a trial parameter vector violates the restriction, the objective function returns an infinite value, so that the point is rejected and the optimiser moves back into the admissible region. Without this check, a trial value with $\alpha + \delta \geq 1$ would make the starting variance $\omega / (1 - \alpha - \delta)$ negative or infinite, and the log-likelihood could not be evaluated. For the EWMA model the condition does not need to be checked, since $\alpha + \delta = 1$ holds by construction and the logit transformation of $\delta$ keeps both $\delta$ and $\alpha = 1 - \delta$ between zero and one.

**Starting values.** As Brooks (2019, §9.9.1) points out, the likelihood of GARCH-type models can have several local maxima, so the outcome of a numerical optimisation may depend on where it starts. Brooks (2019, Box 9.3) advises using a regression to obtain starting values for the mean parameters, and choosing plausible starting values away from zero for the variance parameters, because zero values often lead to a local maximum. We follow this advice. The mean starts at the sample mean, $\mu_0 = \bar{y}$, which is the least-squares estimate of a regression on a constant. For GARCH(1,1), the first set of starting values for the variance parameters is $\alpha_0 = 0.05$ and $\delta_0 = 0.90$, which are typical magnitudes for daily returns and are also used in the course example code. The constant is set to $\omega_0 = s^2 (1 - \alpha_0 - \delta_0)$, where $s^2$ is the sample variance of the returns, so that the unconditional variance implied by the starting values equals the sample variance. Since the course slides advise restarting the optimisation when local maxima are a concern, the estimation is repeated from two further starting points, $(\alpha_0, \delta_0) = (0.10, 0.85)$ and $(0.03, 0.95)$. The EWMA model is likewise estimated from three starting values for the decay factor, $\delta_0 = 0.94$, $0.90$ and $0.97$, each combined with $\mu_0 = \bar{y}$. For both models, the run that attains the highest log-likelihood is retained as the final estimate.

### 2.6 Initial value of the variance

The GARCH filter is recursive: to compute $\sigma_2^2$, a value for $\sigma_1^2$ is needed, and this initial value has to be chosen. Neither the assignment nor the course material prescribes it. For the GARCH(1,1) model, we follow the course example code and initialise the filter at the unconditional variance, $\sigma_1^2 = \omega / (1 - \alpha - \delta)$, which is the natural starting point for a stationary process that fluctuates around this level.

For the EWMA and RiskMetrics models, this formula cannot be used: with $\omega = 0$ and $\alpha + \delta = 1$, it reduces to $0/0$ and is undefined, which reflects the fact that these models have no long-run variance. Both models are therefore initialised at the sample variance of the returns, $\sigma_1^2 = s^2$, the historical variance estimate of Brooks (2019, §9.3). An alternative, used in the EWMA function of the course example code, is to start from the first squared return. We prefer the sample variance, because a single squared return is a very noisy estimate of the variance: a small first return would make the filter start far below the typical variance level, and the optimiser could then adjust the estimated mean partly to influence this starting value. Because EWMA and RiskMetrics use the same initial value, the RiskMetrics model remains an exact special case of the EWMA model, obtained by setting $\delta = 0.94$ and $\mu = \bar{y}$. In all three models, the influence of the initial value fades as the filter moves through the sample, since its weight declines geometrically with each step.

### 2.7 Extraction of the conditional variances

Once the optimal parameters have been found, the regression and GARCH filters are run one final time at these parameter values. This produces the estimated conditional variance $\hat{\sigma}_t^2$ for every trading day in the sample, separately for each ETF and each model. For the RiskMetrics model, which has no estimated parameters, the filters are run directly at the fixed values. The resulting variance series form the basis for the evaluation of the models in Section 4. They are saved in three files, `sigma2_garch.csv`, `sigma2_ewma.csv` and `sigma2_riskmetrics.csv`, each of which contains one column per ETF.

### 2.8 Implementation

All models are implemented in Python, using NumPy and pandas for the data handling and SciPy (`scipy.optimize.minimize`) for the numerical optimisation. The code is contained in the script `baseline.py` and follows the structure of the regression-GARCH example code provided in the course. The parameter vector is ordered as $(\omega, \alpha, \delta, \beta)$. The function `FiltRegr` computes the residuals of the mean equation, with a column of ones as the only regressor, and `FiltGARCH` implements the variance filter. The function `LnLRegrGARCH` combines both filters and returns the log-likelihood contributions. The functions `TransPar` and `TransBackPar` carry out the parameter transformations, and `AvgNLnLRegrGARCHTr` is the objective function, which also checks the restriction $\alpha + \delta < 1$. The routine `EstRegrGARCH` performs the maximum likelihood estimation, including the repeated optimisation from different starting values.

Compared with the course example code, the implementation adds the logit transformation of $\delta$, the check on $\alpha + \delta < 1$ and the additional starting values, and it allows the GARCH filter to start from a given initial variance. Because EWMA and RiskMetrics are special cases of GARCH(1,1), they do not require their own filter or likelihood: their parameters are mapped into the GARCH parameter vector, $(0, 1 - \delta, \delta, \mu)$ for EWMA and $(0, 0.06, 0.94, \bar{y})$ for RiskMetrics, and the same `FiltRegr`, `FiltGARCH` and `LnLRegrGARCH` functions are used. This keeps the three models directly comparable and ensures that any difference between them stems from the restrictions alone.

## References

Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity. *Journal of Econometrics* 31.3, pp. 307–327.

Brooks, C. (2019). *Introductory Econometrics for Finance*. 4th ed. Cambridge: Cambridge University Press.
