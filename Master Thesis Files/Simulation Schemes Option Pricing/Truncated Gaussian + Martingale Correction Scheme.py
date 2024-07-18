import numpy as np
from scipy.stats import norm
from scipy.optimize import root_scalar
import time

np.random.seed(112233)

# Parameters
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
K = 100
gamma1 = 0.5
gamma2 = 0.5
alpha = 4.5

kappa_tilde = kappa + lambda_
theta_tilde = (kappa * theta) / (kappa + lambda_)

# Define the domain for psi
psi_min = 1 / alpha**2
psi_max = sigma**2 / (2 * kappa_tilde * theta_tilde)

# Create an equidistant grid over the domain
psi_grid = np.linspace(psi_min, psi_max, 100)

# Function to find r(psi)
def find_rr(psi, bracket_width=10):
    def phi(x):
        return (2 * np.pi)**(-0.5) * np.exp(-x**2 / 2)

    def Phi(x):
        return norm.cdf(x)

    def equation(rr, psi):
        lhs = rr * phi(rr) + Phi(rr) * (1 + rr**2)
        rhs = (1 + psi) * (phi(rr) + rr * Phi(rr))**2
        return lhs - rhs

    a, b = -bracket_width, bracket_width
    f_a, f_b = equation(a, psi), equation(b, psi)
    while f_a * f_b > 0:
        f_a, f_b = equation(a, psi), equation(b, psi)
        if abs(a) > 1e6 or abs(b) > 1e6:
            raise ValueError("Could not find a valid bracket for root finding.")

    sol = root_scalar(equation, args=(psi,), bracket=[a, b], method='brentq')
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Root finding did not converge")

# Precompute r(psi) over the grid
r_grid = np.array([find_rr(psi) for psi in psi_grid])

# Compute f_mu(psi) and f_sigma(psi)
def f_mu(psi, rr):
    def phi(x):
        return (2 * np.pi)**(-0.5) * np.exp(-x**2 / 2)
    def Phi(x):
        return norm.cdf(x)
    phi_r = phi(rr)
    Phi_r = Phi(rr)
    return rr / (phi_r + rr * Phi_r)

def f_sigma(psi, rr):
    def phi(x):
        return (2 * np.pi)**(-0.5) * np.exp(-x**2 / 2)
    def Phi(x):
        return norm.cdf(x)
    phi_r = phi(rr)
    Phi_r = Phi(rr)
    return (1 / np.sqrt(psi)) / (phi_r + rr * Phi_r)

f_mu_grid = np.array([f_mu(psi, r) for psi, r in zip(psi_grid, r_grid)])
f_sigma_grid = np.array([f_sigma(psi, r) for psi, r in zip(psi_grid, r_grid)])

# Function to find the index in psi_grid that corresponds to the value of psi
def find_nearest_index(array, value):
    return np.abs(array - value).argmin()

def generateHestonPathTGDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n):
    S = np.zeros(n + 1)
    S[0] = S0
    v = np.zeros(n + 1)
    v[0] = v0
    v_zero_count = 0
    dt = T / n

    kappa_tilde = kappa + lambda_
    theta_tilde = (kappa * theta) / (kappa + lambda_)
    exponent = np.exp(-kappa_tilde * dt)
    K0 = -rho * kappa_tilde * theta_tilde * dt / sigma
    K1 = gamma1 * dt * (-0.5 + (kappa_tilde * rho) / sigma) - rho / sigma
    K2 = gamma2 * dt * (-0.5 + (kappa_tilde * rho) / sigma) + rho / sigma
    K3 = gamma1 * dt * (1 - rho**2)
    K4 = gamma2 * dt * (1 - rho**2)
    A = (rho / sigma) * (1 + kappa_tilde * gamma2 * dt) - (1 / 2) * gamma2 * dt * rho**2


    Zs_array = norm.rvs(size=n)
    Uv_array = np.random.uniform(size=n)
    Zv_array = norm.ppf(Uv_array)

    for i in range(1, n + 1):
        m = theta_tilde + (v[i - 1] - theta_tilde) * exponent
        s2 = ((v[i - 1] * sigma**2 * exponent * (1 - exponent)) / kappa_tilde +
              (theta_tilde * sigma**2 * (1 - exponent)**2 / (2 * kappa_tilde)))
        psi = s2 / m**2

        if 1/np.sqrt(psi)>alpha:
            mean = m
            SD = np.sqrt(s2)
            Zv = Zv_array[i - 1]
            d1 = (mean / SD) + A * SD
            d2 = mean / SD
            M = np.exp(A * mean + (1 / 2) * (SD**2) * (A**2)) * norm.cdf(d1) + norm.cdf(-d2) 
            K0star = -np.log(M) - (K1 + (1 / 2) * K3) * v[i-1]
            v_next = max((mean + SD * Zv), 0)
        else:
            nearest_index = find_nearest_index(psi_grid, psi)
            f_mean = f_mu_grid[nearest_index]
            f_SD = f_sigma_grid[nearest_index]
            mean = f_mean * m
            SD = f_SD * np.sqrt(s2)
            Zv = Zv_array[i - 1]
            d1 = (mean / SD) + A * SD
            d2 = mean / SD
            M = np.exp(A * mean + (1 / 2) * (SD**2) * (A**2)) * norm.cdf(d1) + norm.cdf(-d2) 
            K0star = -np.log(M) - (K1 + (1 / 2) * K3) * v[i-1]    
            v_next = max((mean + SD * Zv), 0)

        if v_next < 0:
            v_zero_count += 1
        v[i] = max(v_next, 0)

        Zs = Zs_array[i - 1]
        S[i] = S[i - 1] * np.exp(r * dt + K0star + K1 * v[i - 1] + K2 * v[i]) * np.exp(np.sqrt(K3 * v[i - 1] + K4 * v[i]) * Zs)

    return S, v_zero_count

def priceHestonCallViaTGMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K):
    start_time = time.time()
    payoffs = np.zeros(N)
    total_v_zero_count = 0

    for i in range(N):
        S, v_zero_count = generateHestonPathTGDisc(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n)
        payoffs[i] = max(S[-1] - K, 0)
        total_v_zero_count += v_zero_count

    option_price = np.exp(-r * T) * np.mean(payoffs)
    std_dev = np.std(payoffs) / np.sqrt(N)
    end_time = time.time()
    computing_time = end_time - start_time

    return option_price, std_dev, computing_time, total_v_zero_count, payoffs

# Function to profile
time_steps = [10, 20, 40, 80, 160, 320]
results = []

for n in time_steps:
    option_price, std_dev, computing_time, total_v_zero_count, payoffs = priceHestonCallViaTGMC(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, n, N, K)
    results.append((T / n, option_price, std_dev, computing_time, total_v_zero_count, payoffs))

# Print results
print("Results (TG+M):")
total_computing_time = 0
for result in results:
    dt, option_price, std_dev, computing_time, total_v_zero_count, _ = result
    total_computing_time += computing_time
    print(f"Time step (dt): {dt}")
    print(f"Option price: {option_price}")
    print(f"Standard deviation: {std_dev}")
    print(f"Computing time: {computing_time} seconds")
    print(f"Zero variance occurrences: {total_v_zero_count}\n")
print(f"Total computing time: {total_computing_time} seconds")

