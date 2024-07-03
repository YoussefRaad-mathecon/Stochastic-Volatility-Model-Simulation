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
gamma1 = 0.5
gamma2 = 0.5
C = 1.5



def generateHestonPathEulerDisc(S0, v0, r, kappa, theta, sigma, rho, lambdaa, T, n):
    kappa_tilde = kappa + lambdaa
    theta_tilde = (kappa*theta)/(kappa+lambdaa)
    dt = T / n
    S = np.zeros(n + 1)
    S[0] = S0
    v = np.zeros(n + 1)
    v[0] = v0
    v_zero_count = 0

    for i in range(1, n + 1): # For dynamics: See slides CTF 2: stochastic volatility models and Fourier methods April 2, 2024 45 and Euler and Milstein Discretization slide 4
        Z1 = np.random.normal(0, 1)
        Z2 = np.random.normal(0, 1)
        Zv = Z1
        Zs = rho * Z1 + np.sqrt(1 - rho**2) * Z2
        
        dv = kappa_tilde * (theta_tilde - v[i-1]) * dt + sigma * np.sqrt(v[i-1] * dt) * Zv
        
        v[i] = v[i-1] + dv
        if v[i] <= 0:
            v_zero_count += 1
            v[i] = 0 #full truncation scheme; simpler than using || or max for every iteration; yields same (tested) result

        dS = r * S[i-1] * dt + np.sqrt(v[i-1] * dt) * S[i-1] * Zs 
        S[i] = S[i-1] + dS
    
    return S, v_zero_count

def priceHestonCallViaEulerMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K):
    start_time = time.time()
    total_v_zero_count = 0
    payoffs = np.zeros(N)
    
    for i in range(N):
        S, v_zero_count = generateHestonPathEulerDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
        total_v_zero_count += v_zero_count
        payoffs[i] = max(S[-1] - K, 0)

    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_dev = np.std(payoffs) / np.sqrt(N)
    end_time = time.time()
    computing_time = end_time - start_time
    
    return option_price, std_dev, computing_time, total_v_zero_count, payoffs

option_price, std_dev, computing_time, total_v_zero_count, payoffs = priceHestonCallViaEulerMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K)
S_test = generateHestonPathEulerDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)

print(f"The stock price is (EulerMC): {S_test}")
print(f"The option price is (EulerMC): {option_price}")
print(f"The standard deviation of the option payoff is (EulerMC): {std_dev}")
print(f"The computing time is (EulerMC): {computing_time} seconds")
print(f"The count of zero variance occurrences is (EulerMC): {total_v_zero_count}")

num_paths = N
paths = []
for i in range(num_paths):
    S, _ = generateHestonPathEulerDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
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
