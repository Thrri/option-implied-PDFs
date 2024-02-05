import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def lognormal_pdf(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu)**2 / (2 * sigma**2)))


def call_option(S, K, T, r, sigma, div):
    d1 = (np.log(S / K) + (r - div + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = (S * np.exp(-div * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call


def butterfly_probabilities():

    S0 = 100
    strikes = np.linspace(55, 145, 91)
    expiration = 1
    rf = 0.05
    volatility = 0.2
    delta = 0.0

    simulated_call_prices = []

    for i in range(len(strikes)):
        simulated_call_prices.append(call_option(S0, strikes[i], expiration, rf, volatility, delta))
    
    butterfly_prices = []
    butterfly_payoffs = []
    probabilities = []

    for i in range(len(simulated_call_prices) - 2):
        butterfly_price = simulated_call_prices[i] - 2*simulated_call_prices[i+1] + simulated_call_prices[i+2]
        butterfly_prices.append(butterfly_price)
    
        butterfly_payoff = (strikes[i+2] - strikes[i]) / 2
        butterfly_payoffs.append(butterfly_payoff)

        probability = butterfly_prices[i] / butterfly_payoffs[i] * np.exp(rf*expiration)
        probabilities.append(probability)


    return probabilities


def plot_distributions():
    
    '''Funtion to plot the true distribution of S at expiration and the implied distribution of S at expiration'''

    S0 = 100
    strikes = np.linspace(55, 143, 89)
    expiration = 1
    rf = 0.05
    volatility = 0.2
    

    # Adjusted mean (mu) for the log-normal distribution
    mu = np.log(S0) + (rf - 0.5 * volatility**2) * expiration

    # Range of stock prices for plotting
    S_range = np.linspace(50, 150, 1000)

    # Probability density values for each stock price in the range
    pdf_values = [lognormal_pdf(S, mu, volatility) for S in S_range]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, pdf_values, label='True Distribution of S (Log-Normal)')
    plt.plot(strikes, butterfly_probabilities(), linestyle='dashed',label='Implied Distribution of S')
    plt.xlabel('S(T)')
    plt.ylabel('Probability')
    plt.title('Distribution of Stock Price at Expiration (Butterfly Method)')
    plt.legend()
    plt.grid(True)
    # Calculate the start and end points for x-axis ticks, rounding to the nearest 10
    start = int(np.floor(min(S_range) / 10) * 10)
    end = int(np.ceil(max(S_range) / 10) * 10)

    # Set x-axis ticks at intervals of 10 using numpy.arange
    plt.xticks(np.arange(start, end + 10, 10))
    plt.show()

plot_distributions()