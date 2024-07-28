import numpy as np
import time
import matplotlib.pyplot as plt

np.random.seed(112233)

# Given parameters
S0 = 100
v0 = 0.06
r = 0.05
kappa = 1
theta = 0.06
sigma = 0.3
rho = -0.5
lambdaa = 0.01
T = 1
N = 100
K_values = [100]
print(kappa + lambdaa)
print((kappa * theta) / (kappa + lambdaa))
# Function to generate a Heston path using Euler discretization with full truncation
def generateHestonPathEulerDisc(S0, v0, r, kappa, theta, sigma, rho, lambdaa, T, n):
    kappa_tilde = kappa + lambdaa
    theta_tilde = (kappa * theta) / (kappa + lambdaa)
    dt = T / n
    S = np.zeros(n + 1)
    S[0] = S0
    v = np.zeros(n + 1)
    v[0] = v0
    v_zero_count = 0
    Z1 = np.random.normal(0, 1, n)
    Z2 = np.random.normal(0, 1, n)
    Zv = Z1
    Zs = rho * Z1 + np.sqrt(1 - rho ** 2) * Z2
    for i in range(1, n + 1):
        dv = kappa_tilde * (theta_tilde - v[i - 1]) * dt + sigma * np.sqrt(v[i - 1] * dt) * Zv[i-1]

        v[i] = v[i - 1] + dv
        if v[i] <= 0:
            v_zero_count += 1
            v[i] = 0  # full truncation scheme

        dS = r * S[i - 1] * dt + np.sqrt(v[i - 1] * dt) * S[i - 1] * Zs[i-1]
        S[i] = S[i - 1] + dS

    return S, v_zero_count

# Function to price a Heston call option using Euler Monte Carlo simulation
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

    return option_price, std_dev, computing_time, total_v_zero_count

# Different time steps
time_steps = [1, 2, 4, 8, 16, 32]

# Store results for each time step
results = {K: [] for K in K_values}

for K in K_values:
    for n in time_steps:
        option_price, std_dev, computing_time, total_v_zero_count = priceHestonCallViaEulerMC(S0, v0, r, kappa, theta, sigma, rho, lambdaa, T, n, N, K)
        results[K].append((T/n, option_price, std_dev, computing_time, total_v_zero_count))

print("Results (Euler):")
# Print results
total_computing_time = 0
for K in K_values:
    print(f"Results for K = {K}:")
    for result in results[K]:
        dt, option_price, std_dev, computing_time, total_v_zero_count = result
        total_computing_time += computing_time
        print(f"Time step (dt): {dt}")
        print(f"Option price: {option_price}")
        print(f"Standard deviation: {std_dev}")
        print(f"Computing time: {computing_time} seconds")
        print(f"Zero variance occurrences: {total_v_zero_count}\n")
print(f"Total computing time: {total_computing_time} seconds")
