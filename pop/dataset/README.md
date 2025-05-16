# Dataset Management

This document describes the dataset creation and management process for the portfolio optimization project. It outlines the step-by-step approach to transform raw stock data into a structured dataset containing returns, volatility, and other financial metrics.

## Overview

The dataset generation process transforms raw stock price data into a comprehensive set of financial metrics used for portfolio optimization. The system processes historical stock prices to calculate daily returns, excess returns, volatility, Sharpe ratios, and correlation matrices.

## Data Processing Steps

### 1. Data Collection and Preparation

The process begins with CSV files containing historical stock data, with each file representing a single stock. Each stock file should contain the following columns:
- Date
- Open
- High
- Low
- Close
- Adj Close
- Volume

### 2. Calculation of Daily Returns

For each stock, we calculate the daily returns using the adjusted closing prices:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

Where:
- $r_t$ is the return on day $t$
- $P_t$ is the adjusted closing price on day $t$
- $P_{t-1}$ is the adjusted closing price on day $t-1$

This calculation is performed using the pandas `pct_change()` method.

**Output File**: `daily_returns_companies.csv`

### 3. Calculation of Excess Returns

Excess returns are calculated by subtracting the risk-free rate from the daily returns:

$$r_{excess} = r_t - r_f$$

Where:
- $r_{excess}$ is the excess return
- $r_t$ is the daily return on day $t$
- $r_f$ is the daily risk-free rate

**Output File**: `daily_excess_return_companies.csv`

### 4. Calculation of Mean Excess Returns

The mean excess return for each stock over the specified period is calculated as:

$$\mu = \frac{\sum r_{excess}}{n}$$

Where:
- $\mu$ is the mean excess return
- $\sum r_{excess}$ is the sum of all excess returns
- $n$ is the number of trading days in the period

**Output File**: `mean_daily_excess_return_companies.csv`

### 5. Calculation of Volatility

Volatility (standard deviation of excess returns) is calculated as:

$$\sigma = \sqrt{\frac{\sum (r_{excess} - \mu)^2}{n}}$$

Where:
- $\sigma$ is the volatility (standard deviation)
- $r_{excess}$ is the excess return
- $\mu$ is the mean excess return
- $n$ is the number of trading days in the period

**Output File**: `daily_volatility_companies.csv`

### 6. Calculation of Sharpe Ratio

The Sharpe ratio measures the risk-adjusted return and is calculated as:

$$Sharpe = \frac{\mu}{\sigma}$$

Where:
- $\mu$ is the mean excess return
- $\sigma$ is the volatility
- $trading\_days$ is the number of trading days in a year (typically 252)

This annualizes the Sharpe ratio to provide a yearly performance metric.

**Output File**: `daily_sharpe_ratio_companies.csv`

### 7. Creation of Annual Summary Statistics

An annual summary is created by annualizing the daily metrics:

- Annual Excess Return = Daily Mean Excess Return × Trading Days
- Annual Volatility = Daily Volatility × √Trading Days
- Annual Sharpe Ratio = Annual Excess Return / Annual Volatility

**Output File**: `annual_resume_companies.csv`

### 8. Calculation of Correlation Matrix

The correlation matrix between all stocks is calculated based on their excess returns:

$$\rho_{i,j} = \frac{cov(r_i, r_j)}{\sigma_i \sigma_j}$$

Where:
- $\rho_{i,j}$ is the correlation between stocks $i$ and $j$
- $cov(r_i, r_j)$ is the covariance between the excess returns of stocks $i$ and $j$
- $\sigma_i$ and $\sigma_j$ are the standard deviations of excess returns for stocks $i$ and $j$

**Output File**: `correlation_companies.csv`

## Dataset Directory Structure

After running the data processing script, the following directory structure will be created:

```
dataset/
├── daily_returns_companies.csv
├── symbols_valid_meta.csv
├── {risk_free_rate}-risk-free-rate/
│   ├── daily_excess_return_companies.csv
│   ├── period-from-{start_date}-to-{end_date}/
│   │   ├── annual_resume_companies.csv
│   │   ├── correlation_companies.csv
│   │   ├── daily_sharpe_ratio_companies.csv
│   │   ├── daily_volatility_companies.csv
│   │   └── mean_daily_excess_return_companies.csv
```

Where:
- `{risk_free_rate}` is the annual risk-free rate with decimals replaced by hyphens (e.g., "4-2" for 4.2%)
- `{start_date}` is the start date of the analysis period in YYYY-MM-DD format
- `{end_date}` is the end date of the analysis period in YYYY-MM-DD format

## Usage Notes

1. The default risk-free rate is set to 4.2% annually.
2. The default number of trading days per year is set to 252.
3. The script automatically handles the creation of necessary directories and files.
4. Stocks with missing data during the specified period are excluded from the analysis.
5. All calculations are performed using pandas and numpy for numerical precision.