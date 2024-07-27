import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar
import time

np.random.seed(112233)

class HestonModel:
    def __init__(self, S0, v0, r, kappa, theta, sigma, rho, lambda_, gamma1, gamma2, alpha):
        self.S0 = S0
        self.v0 = v0
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.lambda_ = lambda_
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.alpha = alpha
        self.kappa_tilde = kappa + lambda_
        self.theta_tilde = (kappa * theta) / (kappa + lambda_)
        self.psi_min = 1 / alpha**2
        self.psi_max = sigma**2 / (2 * self.kappa_tilde * self.theta_tilde)
        self.psi_grid = np.linspace(self.psi_min, self.psi_max, 100)
        self.r_grid = np.array([self.find_rr(psi) for psi in self.psi_grid])
        self.f_mu_grid = np.array([self.f_mu(psi, r) for psi, r in zip(self.psi_grid, self.r_grid)])
        self.f_sigma_grid = np.array([self.f_sigma(psi, r) for psi, r in zip(self.psi_grid, self.r_grid)])

    def generateHestonPathEulerDisc(self, T, n):
        dt = T / n
        S = np.zeros(n + 1)
        S[0] = self.S0
        v = np.zeros(n + 1)
        v[0] = self.v0
        v_zero_count = 0
        Z1 = np.random.normal(0, 1, n)
        Z2 = np.random.normal(0, 1, n)
        Zv = Z1
        Zs = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2

        for i in range(1, n + 1):
            dv = self.kappa_tilde * (self.theta_tilde - v[i - 1]) * dt + self.sigma * np.sqrt(v[i - 1] * dt) * Zv[i-1]
            v[i] = v[i - 1] + dv
            if v[i] <= 0:
                v_zero_count += 1
                v[i] = 0  # full truncation scheme

            dS = self.r * S[i - 1] * dt + np.sqrt(v[i - 1] * dt) * S[i - 1] * Zs[i-1]
            S[i] = S[i - 1] + dS

        return S, v_zero_count

    def generateHestonPathMilsteinDisc(self, T, n):
        dt = T / n
        S = np.zeros(n + 1)
        S[0] = self.S0
        v = np.zeros(n + 1)
        v[0] = self.v0
        v_zero_count = 0

        for i in range(1, n + 1):
            Z1 = np.random.normal(0, 1)
            Z2 = np.random.normal(0, 1)
            Zv = Z1 
            Zs = self.rho * Z1 + np.sqrt(1 - self.rho ** 2) * Z2 
            
            dv = self.kappa_tilde * (self.theta_tilde - v[i-1]) * dt + self.sigma * np.sqrt(v[i-1] * dt) * Zv + 0.25 * self.sigma**2 * dt * (Zv**2 - 1)
            v[i] = v[i-1] + dv
            if v[i] < 0:
                v_zero_count += 1
                v[i] = 0  # Applying full truncation

            dS = self.r * S[i-1] * dt + np.sqrt(max(v[i-1], 0) * dt) * S[i-1] * Zs 
            S[i] = S[i-1] + dS

        return S, v_zero_count
    # Define the method for QED discretization
    def generateHestonPathQEDisc(self, T, n, C):
        dt = T / n
        S = np.zeros(n + 1)
        S[0] = self.S0
        v = np.zeros(n + 1)
        v[0] = self.v0
        v_zero_count = 0
        kappa_tilde = self.kappa_tilde
        theta_tilde = self.theta_tilde
        exponent = np.exp(-kappa_tilde * dt)

        K0 = -self.rho * kappa_tilde * theta_tilde * dt / self.sigma
        K1 = self.gamma1 * dt * (-0.5 + (kappa_tilde * self.rho) / self.sigma) - self.rho / self.sigma
        K2 = self.gamma2 * dt * (-0.5 + (kappa_tilde * self.rho) / self.sigma) + self.rho / self.sigma
        K3 = self.gamma1 * dt * (1 - self.rho**2)
        K4 = self.gamma2 * dt * (1 - self.rho**2)

        # Pre-generate random numbers
        Uv_array = np.random.uniform(size=n)
        Zv_array = norm.ppf(Uv_array)
        Zs_array = norm.rvs(size=n)

        for i in range(1, n + 1):
            m = theta_tilde + (v[i - 1] - theta_tilde) * exponent
            s2 = ((v[i - 1] * self.sigma**2 * exponent * (1 - exponent)) / kappa_tilde +
                  (theta_tilde * self.sigma**2 * (1 - exponent)**2 / (2 * kappa_tilde)))
            psi = s2 / m**2
            Uv = Uv_array[i - 1]

            # Switching rule
            if psi <= C:
                Zv = Zv_array[i - 1]
                b2 = (2 / psi) - 1 + np.sqrt(2 / psi) * np.sqrt((2 / psi) - 1)
                a = m / (1 + b2)
                v_next = a * (np.sqrt(b2) + Zv)**2
            else:  # psi > C
                p = (psi - 1) / (psi + 1)
                beta = (1 - p) / m
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
            S[i] = S[i - 1] * np.exp(self.r * dt + K0 + K1 * v[i - 1] + K2 * v[i]) * np.exp(np.sqrt(K3 * v[i - 1] + K4 * v[i]) * Zs) 

        return S, v_zero_count

    # Define the method for QEM discretization
    def generateHestonPathQEMDisc(self, T, n, C):
        dt = T / n
        S = np.zeros(n + 1)
        S[0] = self.S0
        v = np.zeros(n + 1)
        v[0] = self.v0
        v_zero_count = 0
        kappa_tilde = self.kappa_tilde
        theta_tilde = self.theta_tilde
        exponent = np.exp(-kappa_tilde * dt)

        K1 = self.gamma1 * dt * (-0.5 + (kappa_tilde * self.rho) / self.sigma) - self.rho / self.sigma
        K2 = self.gamma2 * dt * (-0.5 + (kappa_tilde * self.rho) / self.sigma) + self.rho / self.sigma
        K3 = self.gamma1 * dt * (1 - self.rho**2)
        K4 = self.gamma2 * dt * (1 - self.rho**2)
        A = (self.rho / self.sigma) * (1 + kappa_tilde * self.gamma2 * dt) - (1 / 2) * self.gamma2 * dt * self.rho**2

        # Pre-generate random numbers
        Uv_array = np.random.uniform(size=n)
        Zv_array = norm.ppf(Uv_array)
        Zs_array = norm.rvs(size=n)

        for i in range(1, n + 1):
            m = theta_tilde + (v[i - 1] - theta_tilde) * exponent
            s2 = ((v[i - 1] * self.sigma**2 * exponent * (1 - exponent)) / kappa_tilde +
                  (theta_tilde * self.sigma**2 * (1 - exponent)**2 / (2 * kappa_tilde)))
            psi = s2 / m**2
            Uv = Uv_array[i - 1]

            # Switching rule
            if psi <= C:
                Zv = Zv_array[i - 1]
                b2 = (2 / psi) - 1 + np.sqrt(2 / psi) * np.sqrt((2 / psi) - 1)
                a = m / (1 + b2)
                if A >= 1 / (2 * a):
                    raise ValueError(f"Stopping condition met: A >= 1/(2*a) at step {i}")
                K0star = -((A * b2 * a) / (1 - 2 * A * a)) + ((1 / 2) * np.log(1 - 2 * A * a)) - (K1 + (1 / 2) * K3) * v[i-1]  
                v_next = a * (np.sqrt(b2) + Zv)**2
            else:  # psi > C
                p = (psi - 1) / (psi + 1)
                beta = (1 - p) / m
                if A >= beta:
                    raise ValueError(f"Stopping condition met: A >= beta at step {i}")
                K0star = -np.log(p + (beta * (1 - p)) / (beta - A)) - (K1 + (1 / 2) * K3) * v[i-1]  
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
            log_Si = np.log(S[i - 1]) + self.r * dt + K0star + K1 * v[i - 1] + K2 * v[i] + np.sqrt(K3 * v[i - 1] + K4 * v[i]) * Zs
            S[i] = np.exp(log_Si)  

        return S, v_zero_count
    
    def find_rr(self, psi, bracket_width=10):
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
    
    def f_mu(self, psi, rr):
        def phi(x):
            return (2 * np.pi)**(-0.5) * np.exp(-x**2 / 2)
        def Phi(x):
            return norm.cdf(x)
        phi_r = phi(rr)
        Phi_r = Phi(rr)
        return rr / (phi_r + rr * Phi_r)
    
    def f_sigma(self, psi, rr):
        def phi(x):
            return (2 * np.pi)**(-0.5) * np.exp(-x**2 / 2)
        def Phi(x):
            return norm.cdf(x)
        phi_r = phi(rr)
        Phi_r = Phi(rr)
        return (1 / np.sqrt(psi)) / (phi_r + rr * Phi_r)
    
    def find_nearest_index(self, array, value):
        return np.abs(array - value).argmin()
    
    def generateHestonPathTGMDisc(self, T, n):
        S = np.zeros(n + 1)
        S[0] = self.S0
        v = np.zeros(n + 1)
        v[0] = self.v0
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
        A = (self.rho / self.sigma) * (1 + kappa_tilde * self.gamma2 * dt) - (1 / 2) * self.gamma2 * dt * self.rho**2

        Zs_array = norm.rvs(size=n)
        Uv_array = np.random.uniform(size=n)
        Zv_array = norm.ppf(Uv_array)

        for i in range(1, n + 1):
            m = theta_tilde + (v[i - 1] - theta_tilde) * exponent
            s2 = ((v[i - 1] * self.sigma**2 * exponent * (1 - exponent)) / kappa_tilde +
                  (theta_tilde * self.sigma**2 * (1 - exponent)**2 / (2 * kappa_tilde)))
            psi = s2 / m**2

            if 1 / np.sqrt(psi) > self.alpha:
                mean = m
                SD = np.sqrt(s2)
                Zv = Zv_array[i - 1]
                d1 = (mean / SD) + A * SD
                d2 = mean / SD
                M = np.exp(A * mean + (1 / 2) * (SD**2) * (A**2)) * norm.cdf(d1) + norm.cdf(-d2) 
                K0star = -np.log(M) - (K1 + (1 / 2) * K3) * v[i-1]
                v_next = max((mean + SD * Zv), 0)
            else:
                nearest_index = self.find_nearest_index(self.psi_grid, psi)
                f_mean = self.f_mu_grid[nearest_index]
                f_SD = self.f_sigma_grid[nearest_index]
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
            S[i] = S[i - 1] * np.exp(self.r * dt + K0star + K1 * v[i - 1] + K2 * v[i]) * np.exp(np.sqrt(K3 * v[i - 1] + K4 * v[i]) * Zs)

        return S, v_zero_count

    def generateHestonPathTGDisc(self, T, n):
        S = np.zeros(n + 1)
        S[0] = self.S0
        v = np.zeros(n + 1)
        v[0] = self.v0
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

        Zs_array = norm.rvs(size=n)
        Uv_array = np.random.uniform(size=n)

        for i in range(1, n + 1):
            m = theta_tilde + (v[i - 1] - theta_tilde) * exponent
            s2 = ((v[i - 1] * self.sigma**2 * exponent * (1 - exponent)) / kappa_tilde +
                  (theta_tilde * self.sigma**2 * (1 - exponent)**2 / (2 * kappa_tilde)))
            psi = s2 / m**2

            if 1 / np.sqrt(psi) > self.alpha:
                mean = m
                SD = np.sqrt(s2)
                Zv = norm.ppf(Uv_array[i - 1])
                v_next = max(mean + SD * Zv, 0)
            else:
                nearest_index = self.find_nearest_index(self.psi_grid, psi)
                f_mean = self.f_mu_grid[nearest_index]
                f_SD = self.f_sigma_grid[nearest_index]
                mean = f_mean * m
                SD = f_SD * np.sqrt(s2)
                Zv = norm.ppf(Uv_array[i - 1])
                v_next = max(mean + SD * Zv, 0)

            if v_next < 0:
                v_zero_count += 1
            v[i] = max(v_next, 0)

            Zs = Zs_array[i - 1]
            S[i] = S[i - 1] * np.exp(self.r * dt + K0 + K1 * v[i - 1] + K2 * v[i]) * np.exp(np.sqrt(K3 * v[i - 1] + K4 * v[i]) * Zs)

        return S, v_zero_count

    def priceHestonCallViaMC(self, K, T, n, num_paths, method):
        S_paths = np.zeros((num_paths, n + 1))
        v_zero_counts = np.zeros(num_paths)

        for i in range(num_paths):
            if method == "TGMDisc":
                S, v_zero_count = self.generateHestonPathTGMDisc(T, n)
            elif method == "TGDisc":
                S, v_zero_count = self.generateHestonPathTGDisc(T, n)
            elif method == "QEMDisc":
                S, v_zero_count = self.generateHestonPathQEMDisc(T, n, C=1.5)
            elif method == "QEDisc":
                S, v_zero_count = self.generateHestonPathQEDisc(T, n, C=1.5)
            elif method == "MilsteinDisc":
                S, v_zero_count = self.generateHestonPathMilsteinDisc(T, n)
            elif method == "EulerDisc":
                S, v_zero_count = self.generateHestonPathEulerDisc(T, n)
            else:
                raise ValueError("Unknown method")
            
            S_paths[i] = S
            v_zero_counts[i] = v_zero_count

        # Calculate the call option price
        call_payoffs = np.maximum(S_paths[:, -1] - K, 0)
        price = np.exp(-self.r * T) * np.mean(call_payoffs)

        return price, v_zero_counts
