from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

# Pricing a Call Option using the Integral Approach

# The probability density function of a bimodal distribution
def bimodal_pdf(x, mu1, sigma1, mu2, sigma2, weight):
    # Original bimodal PDF without normalization
    def original_pdf(x, mu1, sigma1, mu2, sigma2, weight):
        return (weight * (1 / (x * sigma1 * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - np.log(S0) - (mu1-0.5*sigma1**2))**2) / (2 * sigma1**2))) + (1-weight) * (1 / (x * sigma2 * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - np.log(S0) - (mu2-0.5*sigma2**2))**2) / (2 * sigma2**2))
    
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


# Extract the PDF from the simulated call option prices (butterfly method)

def butterfly_probabilities():

    simulated_call_prices = []

    for strike in strikes:
        call = call_option_integral(strike, S0, r, T, mu1, sigma1, mu2, sigma2, weight)
        simulated_call_prices.append(call)
    
    butterfly_prices = []
    butterfly_payoffs = []
    probabilities = []

    for i in range(len(simulated_call_prices) - 2):
        butterfly_price = simulated_call_prices[i] - 2*simulated_call_prices[i+1] + simulated_call_prices[i+2]
        butterfly_prices.append(butterfly_price)
    
        butterfly_payoff = (strikes[i+2] - strikes[i]) / 2
        butterfly_payoffs.append(butterfly_payoff)

        probability = butterfly_prices[i] / butterfly_payoffs[i] * np.exp(r*T)
        probabilities.append(probability)

    return probabilities


def plot_distributions():
    
    '''Funtion to plot the true distribution of S at expiration and the implied distribution of S at expiration'''

    strikes = np.linspace(55, 143, 89)

    # Range of stock prices for plotting
    S_range = np.linspace(50, 150, 1000)

    # Probability density values for each stock price in the range
    pdf_values = [bimodal_pdf(S, mu1, sigma1, mu2, sigma2, weight) for S in S_range]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, pdf_values, label='True Distribution of S (Bimodal)', linewidth=2)
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



# Given parameters
S0 = 100  # Current stock price
T = 1     # Time to expiration (in years)
r = 0.05  # Risk-free interest rate (annual)
strikes = np.linspace(55, 145, 91)

# Bimodal distribution parameters
mu1 = -0.05  # Mean of the first normal distribution
sigma1 = 0.09  # Standard deviation of the first normal distribution
mu2 = 0.15  # Mean of the second normal distribution
sigma2 = 0.05 # Standard deviation of the second normal distribution
weight = 0.5  # Weight factor for mixing the two distributions

plot_distributions()