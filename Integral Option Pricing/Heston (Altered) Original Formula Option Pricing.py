import numpy as np
from scipy.integrate import quad
import time

np.random.seed(667)

# Given parameters
S0 = 100
v0 = 0.04
r = 0.05
kappa = 0.5
theta = 0.04
sigma = 1
rho = -0.9
lambda_ = 0.00
K = 100
tau = 10

kappa_tilde = kappa + lambda_
theta_tilde = (kappa * theta) / (kappa + lambda_)
#K=60
#The option price is (OriginalFT): 65.38839015610131
#The computing time is (OriginalFT): 0.02150273323059082 seconds
#The estimated numerical integration error is: 1.2794298577969627e-08


#K=100
#The option price is (OriginalFT): 43.766900951263
#The computing time is (OriginalFT): 0.013590335845947266 seconds
#The estimated numerical integration error is: 7.07720051437187e-09


#K=140
#The option price is (OriginalFT): 23.894996736001076
#The computing time is (OriginalFT): 0.021468400955200195 seconds
#The estimated numerical integration error is: 2.053531654436575e-09

def characteristicFunctionHeston(u, St, vt, r, kappa, theta, sigma, rho, lambda_, tau, j):
    i = complex(0, 1)
    a = kappa * theta
    b1 = kappa + lambda_ - rho * sigma
    b2 = kappa + lambda_
    u1 = 0.5
    u2 = -0.5

    if j == 1:
        b_j = b1
        u_j = u1
    else:
        b_j = b2
        u_j = u2

    d_j = np.sqrt((b_j - rho * sigma * i * u) ** 2 - sigma ** 2 * (2 * u_j * i * u - u ** 2))
    g_j = (b_j - rho * sigma * i * u - d_j) / (b_j - rho * sigma * i * u + d_j)
    C_j = r * i * u * tau + (a / sigma ** 2) * ((b_j - rho * sigma * i * u - d_j) * tau - 2 * np.log((1 - g_j * np.exp(-d_j * tau)) / (1 - g_j)))
    D_j = (b_j - rho * sigma * i * u - d_j) * ((1 - np.exp(-d_j * tau) / sigma ** 2 * (1 - g_j * np.exp(-d_j * tau))))

    return np.exp(C_j + D_j * vt + i * u * np.log(St))

def priceHestonCallViaOriginalFT(St, vt, r, kappa, theta, sigma, rho, lambda_, tau, K):
    integrationlimit = np.inf
    integrandQj = lambda u, j: np.real(np.exp(-complex(0, 1) * u * np.log(K)) * characteristicFunctionHeston(u, St, vt, r, kappa, theta, sigma, rho, lambda_, tau, j) / (complex(0, 1) * u))
    Q1, errorQ1 = quad(integrandQj, 0, integrationlimit, args=(1,))
    Q2, errorQ2 = quad(integrandQj, 0, integrationlimit, args=(2,))
    
    option_price = St * (0.5 + (1 / np.pi) * Q1) - np.exp(-r * tau) * K * (0.5 + (1 / np.pi) * Q2)
    total_error = (1 / np.pi) * (np.abs(errorQ1) + np.abs(errorQ2))
    
    return option_price, total_error

start_time = time.time()
option_price, total_error = priceHestonCallViaOriginalFT(S0, v0, r, kappa, theta, sigma, rho, lambda_, tau, K)
end_time = time.time()
computing_time = end_time - start_time

print(f"The option price is (OriginalFT): {option_price}")
print(f"The computing time is (OriginalFT): {computing_time} seconds")
print(f"The estimated numerical integration error is: {total_error}")
