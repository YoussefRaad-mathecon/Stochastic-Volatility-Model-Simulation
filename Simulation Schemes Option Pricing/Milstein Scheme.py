import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(112233)

# Given parameters
S0 = 100
v0 = 0.04
r = 0.05
kappa = 0.5
theta = 0.04
sigma = 1
rho = -0.9
lambda_ = 0.01
T = 10
N = 100
K_values = [100]
gamma1 = 0.5
gamma2 = 0.5
C = 1.5

# Function to generate a Heston path using Milstein discretization with full truncation
def generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n):
    kappa_tilde = kappa + lambda_
    theta_tilde = (kappa * theta) / (kappa + lambda_)
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
        Zs = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2 
        
        dv =  kappa_tilde * (theta_tilde - v[i-1]) * dt + sigma * np.sqrt(v[i-1] * dt) * Zv + 0.25 * sigma**2 * dt * (Zv**2 - 1) 
        v[i] = v[i-1] + dv
        if v[i] < 0:
            v_zero_count += 1
            v[i] = 0  # Applying full truncation

        dS = r * S[i-1] * dt + np.sqrt(max(v[i-1], 0) * dt) * S[i-1] * Zs 
        S[i] = S[i-1] + dS

    return S, v_zero_count

# Function to price a Heston call option using Milstein Monte Carlo simulation
def priceHestonCallViaMilsteinMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K):
    start_time = time.time()
    payoffs = np.zeros(N)
    total_v_zero_count = 0
    
    for i in range(N):
        S, v_zero_count = generateHestonPathMilsteinDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
        print(S[-1])
        payoffs[i] = max(S[-1] - K, 0)
        total_v_zero_count += v_zero_count
        
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_dev = np.std(payoffs) / np.sqrt(N)
    end_time = time.time()
    computing_time = end_time - start_time
    
    return option_price, std_dev, computing_time, total_v_zero_count, payoffs

# Different time steps
time_steps = [10, 20, 40, 80, 160, 320]

# Store results for each time step
results = {K: [] for K in K_values}

for K in K_values:
    for n in time_steps:
        option_price, std_dev, computing_time, total_v_zero_count, payoffs = priceHestonCallViaMilsteinMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K)
        results[K].append((T/n, option_price, std_dev, computing_time, total_v_zero_count, payoffs))

print("Results (Milstein):")
# Print results
total_computing_time = 0
for K in K_values:
    print(f"Results for K = {K}:")
    for result in results[K]:
        dt, option_price, std_dev, computing_time, total_v_zero_count, _ = result
        total_computing_time += computing_time
        print(f"Time step (dt): {dt}")
        print(f"Option price: {option_price}")
        print(f"Standard deviation: {std_dev}")
        print(f"Computing time: {computing_time} seconds")
        print(f"Zero variance occurrences: {total_v_zero_count}\n")
print(f"Total computing time: {total_computing_time} seconds")
