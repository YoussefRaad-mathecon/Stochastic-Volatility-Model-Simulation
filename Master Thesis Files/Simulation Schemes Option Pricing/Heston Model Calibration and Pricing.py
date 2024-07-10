import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.stats import norm

class HestonModel:
    def __init__(self, S0, v0, r, kappa, theta, sigma, rho, lambda_, T, N, gamma1=0.5, gamma2=0.5, C=1.5):
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lambda_ = lambda_
        self.T = T
        self.N = N
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.C = C

    def priceHestonCallMC(self, path_generator, K, n):
        start_time = time.time()
        payoffs = np.zeros(self.N)
        total_v_zero_count = 0

        for i in range(self.N):
            S, v_zero_count = path_generator(self.S0, self.v0, self.r, self.T, n)
            payoffs[i] = max(S[-1] - K, 0)
            total_v_zero_count += v_zero_count

        option_price = np.exp(-self.r * self.T) * np.mean(payoffs)
        std_dev = np.std(payoffs) / np.sqrt(self.N)
        end_time = time.time()
        computing_time = end_time - start_time

        return option_price, std_dev, computing_time, total_v_zero_count

    def generateHestonPathEulerDisc(self, S0, v0, r, T, n):
        kappa_tilde = self.kappa + self.lambda_
        theta_tilde = (self.kappa * self.theta) / (self.kappa + self.lambda_)
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
            Zs = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2
            dv = kappa_tilde * (theta_tilde - v[i - 1]) * dt + self.sigma * np.sqrt(v[i - 1] * dt) * Zv

            v[i] = v[i - 1] + dv
            if v[i] <= 0:
                v_zero_count += 1
                v[i] = 0  # full truncation scheme

            dS = r * S[i - 1] * dt + np.sqrt(v[i - 1] * dt) * S[i - 1] * Zs
            S[i] = S[i - 1] + dS

        return S, v_zero_count

    def generateHestonPathMilsteinDisc(self, S0, v0, r, T, n):
        kappa_tilde = self.kappa + self.lambda_
        theta_tilde = (self.kappa * self.theta) / (self.kappa + self.lambda_)
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
            Zs = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2 
            
            dv =  kappa_tilde * (theta_tilde - v[i-1]) * dt + self.sigma * np.sqrt(v[i-1] * dt) * Zv + 0.25 * self.sigma**2 * dt * (Zv**2 - 1) 
            v[i] = v[i-1] + dv
            if v[i] < 0:
                v_zero_count += 1
                v[i] = 0  # Applying full truncation

            dS = r * S[i-1] * dt + np.sqrt(max(v[i-1], 0) * dt) * S[i-1] * Zs 
            S[i] = S[i-1] + dS

        return S, v_zero_count

    def generateHestonPathQEDisc(self, S0, v0, r, T, n):
        S = np.zeros(n + 1)
        S[0] = S0
        v = np.zeros(n + 1)
        v[0] = v0
        v_zero_count = 0
        dt = T / n

        kappa_tilde = self.kappa + self.lambda_
        theta_tilde = (self.kappa * self.theta) / (self.kappa + self.lambda_)
        exponent = np.exp(-kappa_tilde * dt)

        K0 = -self.rho * kappa_tilde * theta_tilde * dt / self.sigma
        K1 = self.gamma1 * dt * (-0.5 + (kappa_tilde * self.rho) / self.sigma) - self.rho / self.sigma
        K2 = self.gamma2 * dt * (-0.5 + (kappa_tilde * self.rho) / self.sigma) + self.rho / self.sigma
        K3 = self.gamma1 * dt * (1 - self.rho**2)
        K4 = self.gamma2 * dt * (1 - self.rho**2)

        Uv_array = np.random.uniform(size=n)
        Zv_array = norm.ppf(Uv_array)
        Zs_array = norm.rvs(size=n)

        for i in range(1, n + 1):
            m = theta_tilde + (v[i - 1] - theta_tilde) * exponent
            s2 = ((v[i - 1] * self.sigma**2 * exponent * (1 - exponent)) / kappa_tilde +
                  (theta_tilde * self.sigma**2 * (1 - exponent)**2 / (2 * kappa_tilde)))
            psi = s2 / m**2
            Uv = Uv_array[i - 1]

            if psi <= self.C:
                Zv = Zv_array[i - 1]
                b2 = (2 / psi) - 1 + np.sqrt(2 / psi) * np.sqrt((2 / psi) - 1)
                a = m / (1 + b2)
                v_next = a * (np.sqrt(b2) + Zv)**2
            else:  # psi > self.C
                p = (psi- 1) / (psi + 1)
                beta = (1 - p) / m
                if (0 <= Uv <= p):
                    v_next = 0
                elif (p < Uv <= 1):
                    v_next = (1 / beta) * np.log((1 - p) / (1 - Uv))

            if v_next < 0:
                v_zero_count += 1
            v[i] = max(v_next, 0)

            Zs = Zs_array[i - 1]
            S[i] = S[i - 1] * np.exp(r * dt + K0 + K1 * v[i - 1] + K2 * v[i]) * np.exp(np.sqrt(K3 * v[i - 1] + K4 * v[i]) * Zs)

        return S, v_zero_count

    def run_pricing(self, K_values, time_steps):
        results_euler = {K: [] for K in K_values}
        results_milstein = {K: [] for K in K_values}
        results_qe = {K: [] for K in K_values}

        for K in K_values:
            for n in time_steps:
                option_price, std_dev, computing_time, total_v_zero_count = self.priceHestonCallMC(self.generateHestonPathEulerDisc, K, n)
                results_euler[K].append((self.T/n, option_price, std_dev, computing_time, total_v_zero_count))
                
                option_price, std_dev, computing_time, total_v_zero_count = self.priceHestonCallMC(self.generateHestonPathMilsteinDisc, K, n)
                results_milstein[K].append((self.T/n, option_price, std_dev, computing_time, total_v_zero_count))
                
                option_price, std_dev, computing_time, total_v_zero_count = self.priceHestonCallMC(self.generateHestonPathQEDisc, K, n)
                results_qe[K].append((self.T/n, option_price, std_dev, computing_time, total_v_zero_count))

        return results_euler, results_milstein, results_qe

    def print_results(self, results, name):
        print(f"Results ({name}):")
        total_computing_time = 0
        for K in results:
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


# Parameters
S0 = 100
v0 = 0.04
r = 0.03
kappa = 1.5
theta = 0.04
sigma = 0.3
rho = -0.9
lambda_ = 0
T = 1
N = 100

# Instantiate the HestonModel class
heston_model = HestonModel(S0, v0, r, kappa, theta, sigma, rho, lambda_, T, N)

# Define the K values and time steps to test
K_values = [90, 100, 110]
time_steps = [50, 100, 200, 400]

# Run the pricing
results_euler, results_milstein, results_qe = heston_model.run_pricing(K_values, time_steps)

# Print the results
heston_model.print_results(results_euler, "Euler")
heston_model.print_results(results_milstein, "Milstein")
heston_model.print_results(results_qe, "QE")
