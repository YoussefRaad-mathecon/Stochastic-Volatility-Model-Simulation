import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

np.random.seed(667)

# Parameters from Andersons paper for the "new parameters" and handin3 finkont2 for the rest
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


# Generate paths
def generateHestonPathQEDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n):
    S = np.zeros(n + 1)
    S[0] = S0
    v = np.zeros(n + 1)
    v[0] = v0
    v_zero_count = 0
    dt = T / n

    kappa_tilde = kappa + lambda_
    theta_tilde = (kappa * theta) / (kappa + lambda_)
    exp = np.exp(-kappa_tilde * dt)


    K0 = -rho * kappa_tilde * theta_tilde * dt / sigma
    K1 = gamma1 * dt * (-0.5 + (kappa_tilde * rho)/ sigma) - rho / sigma
    K2 = gamma2 * dt * (-0.5 + (kappa_tilde * rho) / sigma) + rho / sigma
    K3 = gamma1 * dt * (1 - rho**2)
    K4 = gamma2 * dt * (1 - rho**2)

    for i in range(1, n + 1):

        m = theta_tilde + (v[i - 1] - theta_tilde) * exp
        s2 = ((v[i - 1] * sigma**2 * exp * (1 - exp)) / kappa_tilde +
            (theta_tilde * sigma**2 * (1 - exp)**2 / (2 * kappa_tilde)))
        xi = s2 / m**2
        Uv = np.random.uniform()
        
        # Swithcing rule
        if xi <= C:
            Zv = norm.ppf(Uv)
            b2 = (2 / xi) - 1 + np.sqrt(2 / xi) * np.sqrt((2 / xi) - 1)
            a = m / (1 + b2)
            v_next = a * (np.sqrt(b2) + Zv)**2
        # xi>C
        else:
            p = (xi - 1) / (xi + 1)
            beta = (1 - p) / m
            if (0 <= Uv <= p):
                v_next = 0
            elif (p < Uv <= 1):
                v_next = (1 / beta) * np.log((1 - p) / (1 - Uv))

        #Zero variance counter and truncation of v if necessary          
        if v_next < 0:
            v_zero_count += 1
        v[i] = max(v_next, 0)
    
        #calculate log-price (with drift (risk free rate under Q(lambda_)))
        Zs = norm.rvs()
        
        S[i] = S[i-1] * np.exp(r * dt + K0 + K1 * v[i-1] + K2 * v[i]) * np.exp(np.sqrt(K3 * v[i-1] + K4 * v[i]) * Zs)


    return S, v_zero_count


def priceHestonCallViaQCMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K):
    start_time = time.time()
    payoffs = np.zeros(N)
    v_zero_count = 0
    
    for i in range(N):
        S, v_zero_count = generateHestonPathQEDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
        payoffs[i] = max(S[-1] - K, 0)
        
    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_dev = np.std(payoffs) / np.sqrt(N)
    end_time = time.time()
    computing_time = end_time - start_time
    
    return option_price, std_dev, computing_time, v_zero_count, payoffs

# Generate paths
num_paths = N
paths = []

for i in range(num_paths):
    S, _ = generateHestonPathQEDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
    paths.append(S)

option_price, std_dev, computing_time, v_zero_count, payoffs = priceHestonCallViaQCMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K)
print(f"The option price is (QEMC): {option_price}")
print(f"The standard deviation of the option price is (QEMC): {std_dev}")
print(f"The computing time is (QEMC): {computing_time} seconds")
print(f"The count of zero variance occurrences is (QEMC): {v_zero_count}")

# Plotting
plt.figure(figsize=(14, 6))
for S in paths: 
    plt.plot(S)
plt.title('Sample Stock Price Paths under the Heston Model')
plt.xlim(0, n)
plt.xlabel('Time Steps')
plt.ylabel('Stock Prices')
plt.grid(True)
plt.show()
