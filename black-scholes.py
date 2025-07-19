
import numpy as np
from scipy.stats import norm
import yfinance as yf
import datetime as dt

# Find volatility of the last year of a stock
def get_data(ticker):
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(252)
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    log_returns = np.log(data / data.shift(1)).dropna()
    sigma = log_returns.std() * np.sqrt(252)

    # Get current price
    live_data = yf.Ticker(ticker).history(period='1d', interval='1m')
    current_price = live_data['Close'].iloc[-1]

    return float(sigma), float(current_price)

# Return option price using black-scholes model
def black_scholes(type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = 0
    if type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == 'P':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    if price == 0:
        print('Option type incorrect')
    return price

def get_greeks(type, S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if type == 'C':
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.pdf(d2))
        rho = K * T * np.exp(-r*T) * norm.cdf(d2)

    elif type == 'P':
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r*T) * norm.pdf(-d2))
        rho = -K * T * np.exp(-r*T) * norm.cdf(d2)

    else:
        print('Option type incorrect')

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return {"Delta": delta,
    "Gamma": gamma,
    "Vega": vega,
    "Theta": theta,
    "Rho": rho
     }

# Underlying stock
ticker = input('Enter ticker: ')

# Option type
type = input('Enter option type (C or P): ')

# Variables used in Black-Scholes
K = float(input('Enter strike price: ')) # Exercise price
T = 1 # Time to maturity (years)
r = 0.0406 # Risk-free interest rate (taken from US tresury yeild)
sigma, S = get_data(ticker) # Underlying price and volatiltiy

option_price = black_scholes('C', S, K, T, r, sigma)

print(f'Price: ${option_price:.2f}')

# Get greeks
greeks = get_greeks('C', S, K, T, r, sigma)
for name, value in greeks.items():
    print(f"{name}: {value:.2f}")




