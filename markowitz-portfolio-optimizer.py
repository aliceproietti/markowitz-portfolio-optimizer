import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Parameters
tickers = ['SPY', 'BND', 'GLD', 'QQQ', 'VTI']
end_date = datetime.today()
start_date = end_date - timedelta(days=5*365)
risk_free_rate = 0.02  # annuale
num_portfolios = 10000  # simulazioni

# Download adjusted prices
adj_close_df = pd.DataFrame()
for ticker in tickers:
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True
    )
    adj_close_df[ticker] = data['Close']

# Data cleaning
adj_close_df = adj_close_df.dropna(how='any')

# Daily log returns
log_returns = np.log1p(adj_close_df.pct_change()).dropna()

# Annualized covariance matrix
cov_matrix = log_returns.cov() * 252

# Portfolio functions
def portfolio_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def portfolio_std(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    ret = portfolio_return(weights, log_returns)
    vol = portfolio_std(weights, cov_matrix)
    return (ret - risk_free_rate) / vol

def neg_sharpe(weights, log_returns, cov_matrix, risk_free_rate):
    return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

# Constraints and bounds
n = len(tickers)
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0.0, 0.40)] * n  # max 40% per asset
initial_weights = np.full(n, 1.0 / n)

# Portfolio Optimization
res = minimize(
    neg_sharpe,
    initial_weights,
    args=(log_returns, cov_matrix, risk_free_rate),
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False}
)
optimal_weights = res.x

# Random portfolios simulation
simulated_returns = []
simulated_vols = []
simulated_sharpes = []

for _ in range(num_portfolios):
    weights = np.random.random(n)
    weights /= np.sum(weights)
    simulated_returns.append(portfolio_return(weights, log_returns))
    simulated_vols.append(portfolio_std(weights, cov_matrix))
    simulated_sharpes.append((simulated_returns[-1] - risk_free_rate) / simulated_vols[-1])

simulated_returns = np.array(simulated_returns)
simulated_vols = np.array(simulated_vols)
simulated_sharpes = np.array(simulated_sharpes)

# results
opt_return = portfolio_return(optimal_weights, log_returns)
opt_vol = portfolio_std(optimal_weights, cov_matrix)
opt_sharpe = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

print("Pesi ottimali (ordinati):")
for t, w in sorted(zip(tickers, optimal_weights), key=lambda x: x[1], reverse=True):
    print(f"{t}: {w:.4f}")

print(f"\nExpected Annual Return: {opt_return:.4f}")
print(f"Expected Volatility   : {opt_vol:.4f}")
print(f"Sharpe Ratio          : {opt_sharpe:.4f}")

# Graph: Simulated Porfolios + Optimal Portfolio
plt.figure(figsize=(12, 6))
plt.scatter(simulated_vols, simulated_returns, c=simulated_sharpes, cmap='viridis', alpha=0.5)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(opt_vol, opt_return, color='red', marker='*', s=300, label='Optimal Portfolio')
plt.xlabel('Annualized Volatility')
plt.ylabel('Expected Return')
plt.title('Simulated Portfolios and Optimal Portfolio')
plt.legend()
plt.show()

# Graph: Optimal Weights
plt.figure(figsize=(10, 6))
plt.bar(tickers, optimal_weights)
plt.xlabel('Assets')
plt.ylabel('Optimal Weights')
plt.title('Optimal Portfolio Weights')
plt.show()
