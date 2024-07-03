import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(667)

# Given parameters
S0 = 100
v0 = 0.06
r = 0.05
kappa = 1
theta = 0.06
sigma = 0.3
rho = -0.5
lambda_ = 0.01
T = 1
n = 100
N = 1000
K = 100



# See slides CTF 2: stochastic volatility models and Fourier methods April 2, 2024 45 and Euler and Milstein Discretization slide 4
def generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n):
    kappa_tilde = kappa + lambda_
    theta_tilde = (kappa*theta)/(kappa+lambda_)
    dt = T / n
    S = np.zeros(n + 1)
    S[0] = S0
    v = np.zeros(n + 1)
    v[0] = v0
    v_zero_count = 0

    for i in range(1, n + 1):
        Z1 = np.random.normal(0, 1)
        Z2 = np.random.normal(0, 1)
        Zv = Z1 
        Zs = rho * Z1 + np.sqrt(1 - rho**2) * Z2 
        
        dv =  kappa_tilde * (theta_tilde - v[i-1]) * dt + sigma * np.sqrt(v[i-1] * dt) * Zv + 0.25 * sigma**2 * dt * (Zv**2 - 1) 
        v[i] = v[i-1] + dv
        if v[i] < 0:
            v_zero_count += 1
            v[i] = 0 # Applying full truncation

        dS = r * S[i-1] * dt + np.sqrt(max(v[i-1], 0) * dt) * S[i-1] * Zs 
        S[i] = S[i-1] + dS

    return S, v_zero_count

def priceHestonCallViaMilsteinMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K):
    start_time = time.time()
    payoffs = np.zeros(N)
    
    for i in range(N):
        S, v_zero_count = generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
        payoffs[i] = max(S[-1] - K, 0)
        
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_dev = np.std(payoffs) / np.sqrt(N)
    end_time = time.time()
    computing_time = end_time - start_time
    
    return option_price, std_dev, computing_time, v_zero_count, payoffs

option_price, std_dev, computing_time, v_zero_count, payoffs = (priceHestonCallViaMilsteinMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K))
print(f"The option price is (MilsteinMC): {option_price}")
print(f"The standard deviation of the option price is (MilsteinMC): {std_dev}")
print(f"The computing time is (MilsteinMC): {computing_time} seconds")
print(f"The count of zero variance occurences is (MilsteinMC): {v_zero_count}")

num_paths = N
paths = []
for i in range(num_paths):
    S, _ = generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
    paths.append(S)


# Plotting the stock paths
plt.figure(figsize=(14, 6))
for S in paths:
    plt.plot(S)
plt.title('Sample Stock Price Paths under the Heston Model')
plt.xlim(0, 100)
plt.xlabel('Time Steps')
plt.ylabel('Stock Prices')
plt.grid(True)
plt.show()


# Plot the option prices
plt.figure(figsize=(14, 6))
plt.plot(payoffs, marker='o', linestyle='-', color='blue')
plt.title('Payoffs under the Heston Model')
plt.xlabel('Payoff number')
plt.ylabel('Payoffs')
plt.grid(True)
plt.show()