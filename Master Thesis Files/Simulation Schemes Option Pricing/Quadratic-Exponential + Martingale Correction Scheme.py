import numpy as np
from scipy.stats import norm
import time

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

# Function to generate a Heston path using QE discretization with full truncation
def generateHestonPathQEDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n):
    S = np.zeros(n + 1)
    S[0] = S0
    v = np.zeros(n + 1)
    v[0] = v0
    v_zero_count = 0
    psi_greater_than_C_count = 0
    dt = T / n

    kappa_tilde = kappa + lambda_
    theta_tilde = (kappa * theta) / (kappa + lambda_)
    exponent = np.exp(-kappa_tilde * dt)
    
    K1 = gamma1 * dt * (-0.5 + (kappa_tilde * rho) / sigma) - rho / sigma
    K2 = gamma2 * dt * (-0.5 + (kappa_tilde * rho) / sigma) + rho / sigma
    K3 = gamma1 * dt * (1 - rho**2)
    K4 = gamma2 * dt * (1 - rho**2)
    A = (rho / sigma) * (1 + kappa_tilde * gamma2 * dt) - (1 / 2) * gamma2 * dt * rho**2
    
    # Pre-generate random numbers
    Uv_array = np.random.uniform(size=n)
    Zv_array = norm.ppf(Uv_array)
    Zs_array = norm.rvs(size=n)

    for i in range(1, n + 1):
        m = theta_tilde + (v[i - 1] - theta_tilde) * exponent
        s2 = ((v[i - 1] * sigma**2 * exponent * (1 - exponent)) / kappa_tilde +
              (theta_tilde * sigma**2 * (1 - exponent)**2 / (2 * kappa_tilde)))
        psi = s2 / m**2
        Uv = Uv_array[i - 1]

        # Switching rule
        if psi <= C:
            Zv = Zv_array[i - 1]
            b2 = (2 / psi) - 1 + np.sqrt(2 / psi) * np.sqrt((2 / psi) - 1)
            a = m / (1 + b2)
            K0star = -((A * b2 * a) / (1 - 2 * A * a)) + ((1 / 2) * np.log(1 - 2 * A * a)) - (K1 + (1 / 2) * dt * gamma1)

            v_next = a * (np.sqrt(b2) + Zv)**2
        else:  # psi > C
            psi_greater_than_C_count += 1
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m

            K0star = -np.log((beta * (1 - p)) / (beta - A)) - (K1 + (1 / 2) * dt * gamma1)
            if (0 <= Uv <= p):
                v_next = 0
            elif (p < Uv <= 1):
                v_next = (1 / beta) * np.log((1 - p) / (1 - Uv))
        # Zero variance counter and truncation of v if necessary
        if v_next < 0:
            v_zero_count += 1
        v[i] = max(v_next, 0)

        # Calculate log-price (with drift (risk-free rate under Q(lambda_)))
        Zs = Zs_array[i - 1]
        log_Si = np.log(S[i - 1]) + r * dt + K0star + K1 * v[i - 1] + K2 * v[i] + np.sqrt(K3 * v[i - 1] + K4 * v[i]) * Zs
        S[i] = np.exp(log_Si)  
        # Debugging prints
        print(f"Step {i}:")
        print(f"  v[{i}] = {v[i]}")
        print(f"  S[{i}] = {S[i]}")
        print(f"  K0star = {K0star}")
        print(f"  K1 = {K1}, K2 = {K2}, K3 = {K3}, K4 = {K4}")

    return S, v_zero_count, psi_greater_than_C_count

# Function to price a Heston call option using QE Monte Carlo simulation
def priceHestonCallViaQEMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K):
    start_time = time.time()
    payoffs = np.zeros(N)
    total_v_zero_count = 0
    total_psi_greater_than_C_count = 0

    for i in range(N):
        S, v_zero_count, psi_greater_than_C_count = generateHestonPathQEDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
        payoffs[i] = max(S[-1] - K, 0)
        total_v_zero_count += v_zero_count
        total_psi_greater_than_C_count += psi_greater_than_C_count

    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_dev = np.std(payoffs) / np.sqrt(N)
    end_time = time.time()
    computing_time = end_time - start_time

    return option_price, std_dev, computing_time, total_v_zero_count, total_psi_greater_than_C_count, payoffs


# Different time steps
time_steps = [10, 20, 40, 80, 160, 320]

# Store results for each time step
results = {K: [] for K in K_values}

for K in K_values:
    for n in time_steps:
        option_price, std_dev, computing_time, total_v_zero_count, total_psi_greater_than_C_count, payoffs = priceHestonCallViaQEMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K)
        results[K].append((T/n, option_price, std_dev, computing_time, total_v_zero_count, total_psi_greater_than_C_count, payoffs))

print("Results (QE+M):")
# Print results
total_computing_time = 0
for K in K_values:
    print(f"Results for K = {K}:")
    for result in results[K]:
        dt, option_price, std_dev, computing_time, total_v_zero_count, total_psi_greater_than_C_count, _ = result
        total_computing_time += computing_time
        print(f"Time step (dt): {dt}")
        print(f"Option price: {option_price}")
        print(f"Standard deviation: {std_dev}")
        print(f"Computing time: {computing_time} seconds")
        print(f"Zero variance occurrences: {total_v_zero_count}")
        print(f"Occurrences of psi > C: {total_psi_greater_than_C_count}\n")
print(f"Total computing time: {total_computing_time} seconds")
