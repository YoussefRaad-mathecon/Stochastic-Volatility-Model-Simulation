import numpy as np
from scipy.stats import chi2, norm, poisson
np.random.seed(112233)

# Given parameters
S_u = 100
V_u = 0.06
r = 0.05
kappa = 0.1
theta = 0.06
sigma_v = 0.5
rho = -0.5
lambdaa = 0.01
t = 1
u = 0
N = 100000
K = 100


def generateHestonPathBDDisc(S_u, V_u, sigma_v, kappa, theta, rho, r, t, K, N):
    d = 4 * theta * kappa / sigma_v**2
    exp_kappa_t = np.exp(-kappa * (t - u))
    payoffs = np.zeros(N)

    for i in range(N):
        lambda_param = 4 * kappa * exp_kappa_t / (sigma_v**2 * (1 - exp_kappa_t)) * V_u
        
        if d > 1:
            chi_d_minus_1 = chi2.rvs(d - 1)
            Z = norm.rvs()
            V_t = sigma_v**2 * (1 - exp_kappa_t) / (4 * kappa) * (Z + np.sqrt(lambda_param))**2 + chi_d_minus_1
        else:
            N_pois = poisson.rvs(0.5 * lambda_param)
            V_t = chi2.rvs(d + 2 * N_pois)

        integrated_variance = 0.5 * (V_u + V_t)
        sqrt_variance_integral = (1 / sigma_v) * (V_t - V_u - kappa * theta * (t - u) + kappa * integrated_variance)
        m = np.log(S_u) + (r * (t - u) - 0.5 * integrated_variance + rho * sqrt_variance_integral)
        sigma = np.sqrt((1 - rho**2) * integrated_variance)
        Z_stock = norm.rvs()
        S_t = np.exp(m + sigma * Z_stock)
        payoffs[i] = np.maximum(S_t - K, 0)

    option_price = np.exp(-r * t) * np.mean(payoffs)
    
    return option_price

