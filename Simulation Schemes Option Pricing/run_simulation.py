import numpy as np
import matplotlib.pyplot as plt
import time
from  HestonModel import HestonModel

# Given parameters
S0 = 100
v0 = 0.04
r = 0.05
kappa = 0.5
theta = 0.04
sigma = 1
rho = -0.9
lambdaa = 0.00
T = 10
N = 100  # Number of paths for Monte Carlo
K_values = [60, 100, 140]
time_steps = [1, 2, 4, 8, 16, 32]
methods = ["EulerDisc", "MilsteinDisc", "QEDisc", "QEMDisc", "TGMDisc", "TGDisc"]

# Initialize results dictionary
results = {method: {K: [] for K in K_values} for method in methods}

# Run simulations for each method, strike price, and time step
heston_model = HestonModel(S0, v0, r, kappa, theta, sigma, rho, lambdaa, gamma1=1, gamma2=1, alpha=0.5)

for method in methods:
    print(f"Running simulations for method: {method}")
    for K in K_values:
        for n in time_steps:
            dt = T / n
            start_time = time.time()
            option_price, v_zero_counts = heston_model.priceHestonCallViaMC(K, T, n, N, method)
            end_time = time.time()
            computing_time = end_time - start_time
            std_dev = np.std(option_price)
            total_v_zero_count = np.sum(v_zero_counts)
            results[method][K].append((dt, option_price, std_dev, computing_time, total_v_zero_count))
            print(f"Finished K={K}, n={n}, dt={dt}, Price={option_price:.4f}, Time={computing_time:.4f}s")

# Print and plot results
for method in methods:
    print(f"\nResults for method: {method}")
    total_computing_time = 0
    for K in K_values:
        print(f"Strike Price K = {K}:")
        for result in results[method][K]:
            dt, option_price, std_dev, computing_time, total_v_zero_count = result
            total_computing_time += computing_time
            print(f"Time step (dt): {dt:.6f}")
            print(f"Option price: {option_price}")
            print(f"Standard deviation: {std_dev}")
            print(f"Computing time: {computing_time} seconds")
            print(f"Zero variance occurrences: {total_v_zero_count}\n")
    print(f"Total computing time for method {method}: {total_computing_time} seconds")

    # Plotting the results
    plt.figure(figsize=(10, 6))
    for K in K_values:
        dt_values = [result[0] for result in results[method][K]]
        prices = [result[1] for result in results[method][K]]
        plt.plot(dt_values, prices, marker='o', label=f"K={K}")
    
    plt.xscale('log')
    plt.xlabel('Time step (dt)')
    plt.ylabel('Option Price')
    plt.title(f'Option Price vs Time step for {method}')
    plt.legend()
    plt.grid(True)
    plt.show()
