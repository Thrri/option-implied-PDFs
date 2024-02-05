from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline


# Pricing a Call Option using the Integral Approach

# The probability density function of a bimodal distribution
def bimodal_pdf(x, mu1, sigma1, mu2, sigma2, weight):
    # Original bimodal PDF without normalization
    def original_pdf(x, mu1, sigma1, mu2, sigma2, weight):
        return (weight * (1 / (x * sigma1 * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - np.log(S0) - (mu1-0.5*sigma1**2))**2) / (2 * sigma1**2)))+ (1-weight) * (1 / (x * sigma2 * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - np.log(S0) - (mu2-0.5*sigma2**2))**2) / (2 * sigma2**2))

    # Calculate the integral of the original PDF over the stock price range (50 to 150)
    area_under_curve, _ = quad(original_pdf, 50, 150, args=(mu1, sigma1, mu2, sigma2, weight))

    # Return the normalized PDF value for the given x
    return original_pdf(x, mu1, sigma1, mu2, sigma2, weight) / area_under_curve


# The integral function for the call option price
def call_option_integral(K, S0, r, T, mu1, sigma1, mu2, sigma2, weight):
    # The integrand function
    def integrand(S):
        return np.exp(-r * T) * max(S - K, 0) * bimodal_pdf(S, mu1, sigma1, mu2, sigma2, weight)

    # Calculate the integral from K to infinity
    result, _ = quad(integrand, K, np.inf)
    return result


# Get the Black-Scholes implied volatilities from the bimodal call prices
def black_scholes_call(S, K, T, r, sigma):
    """
    Calculate the Black-Scholes price of a call option.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def find_vol(target, S, K, T, r):
    """
    Find the implied volatility of a BS call option
    """
    f = lambda sigma: black_scholes_call(S, K, T, r, sigma) - target

    try:
        return brentq(f, 1e-6, 4)
    except ValueError:
        # Fallback to a wider range if the initial range fails
        try:
            return brentq(f, 1e-8, 10)
        except ValueError:
            # Handle cases where brentq still fails
            print(f"Unable to find root for strike {K} and price {target}")
            return np.nan


def IV_Volatilities():
    simulated_call_prices = [call_option_integral(strike, S0, r, T, mu1, sigma1, mu2, sigma2, weight) for strike in strikes]
    Implied_volatilities = [find_vol(call_price, S0, strike, T, r) for call_price, strike in zip(simulated_call_prices, strikes)]
    return Implied_volatilities


# Given parameters
S0 = 100  # Current stock price
T = 1     # Time to expiration (in years)
r = 0.05  # Risk-free interest rate (annual)
strikes = np.linspace(55, 145, 61)

# Bimodal distribution parameters
mu1 = -0.05  # Mean of the first normal distribution
sigma1 = 0.09 # Standard deviation of the first normal distribution
mu2 = 0.15 # Mean of the second normal distribution
sigma2 = 0.05  # Standard deviation of the second normal distribution
weight = 0.5  # Weight factor for mixing the two distributions


# Now get implied volatilities and interpolate them
implied_vols = IV_Volatilities()

# Cubic spline interpolation
spline_IV = CubicSpline(strikes, implied_vols)

# Generate a finer grid for strikes
fine_strikes = np.linspace(55, 145, 91)

# Evaluate the spline on the finer grid
IV_spline_values = spline_IV(fine_strikes)

# Make BS prices from interpolated values
BS_prices = []
for i in range(len(IV_spline_values)):
    call = black_scholes_call(S0, fine_strikes[i], T, r, IV_spline_values[i])
    BS_prices.append(call)

# Finally apply Breeden-Litzenberger on BS_Prices

# Cubic spline interpolation
spline_BS = CubicSpline(fine_strikes, BS_prices)

# Generate a finer grid for strikes
fine_strikes = np.linspace(55, 145, 1000)

# Evaluate the spline on the finer grid
spline_values = spline_BS(fine_strikes)

# Numerical differentiation
first_derivative = spline_BS.derivative()
second_derivative = first_derivative.derivative()

# Evaluate derivatives on the finer grid
spline_first_derivative = first_derivative(fine_strikes)
spline_second_derivative = second_derivative(fine_strikes)

for i in range(len(spline_second_derivative)):
    spline_second_derivative[i] = spline_second_derivative[i] * np.exp(r*T)


# Plot the true distribution of S at expiration and the implied distribution of S at expiration

# Range of stock prices for plotting
S_range = np.linspace(50, 150, 1000)

# Probability density values for each stock price in the range
pdf_values = [bimodal_pdf(S, mu1, sigma1, mu2, sigma2, weight) for S in S_range]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(S_range, pdf_values, label='True Distribution of S (Bimodal)', linewidth=2)
plt.plot(fine_strikes, spline_second_derivative, linestyle='dashed',label='Implied Distribution of S', linewidth=2)
plt.xlabel('S(T)')
plt.ylabel('Probability')
plt.title('Distribution of Stock Price at Expiration (Derivative Method in IV Space)')
plt.legend()
plt.grid(True)

# Calculate the start and end points for x-axis ticks, rounding to the nearest 10
start = int(np.floor(min(S_range) / 10) * 10)
end = int(np.ceil(max(S_range) / 10) * 10)

# Set x-axis ticks at intervals of 10 using numpy.arange
plt.xticks(np.arange(start, end + 10, 10))

plt.show()