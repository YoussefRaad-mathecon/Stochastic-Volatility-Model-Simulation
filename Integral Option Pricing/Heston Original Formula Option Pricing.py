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
#The option price is (OriginalFT): 61.91760725197615 K=60

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

    d_j = np.sqrt((rho * sigma * i * u - b_j)**2 - sigma**2 * (2 * u_j * i * u - u**2))
    g_j = (b_j - rho * sigma * i * u + d_j) / (b_j - rho * sigma * i * u - d_j)
    C_j = r * i * u * tau + (a / sigma**2) * ((b_j - rho * sigma * i * u + d_j) * tau - 2 * np.log((1 - g_j * np.exp(d_j * tau)) / (1 - g_j)))
    D_j = ((b_j - rho * sigma * i * u + d_j) / sigma**2 )* ((1 - np.exp(d_j * tau)) / (1 - g_j * np.exp(d_j * tau)))

    return np.exp(C_j + D_j * vt + i * u * np.log(St))


def priceHestonCallViaOriginalFT(St, vt, r, kappa, theta, sigma, rho, lambda_, tau, K):
    integrationlimit = 160
    integrandQj = lambda u, j: np.real(np.exp(-complex(0, 1) * u * np.log(K)) * characteristicFunctionHeston(u, St, vt, r, kappa, theta, sigma, rho, lambda_, tau, j) / (complex(0, 1) * u))
    Q1 = 0.5 + (1 / np.pi) * quad(integrandQj, 0, integrationlimit, args=(1,))[0] #j=1: quad function from scipy.integrate for  numerical integration.
    Q2 = 0.5 + (1 / np.pi) * quad(integrandQj, 0, integrationlimit, args=(2,))[0] #j=2: quad function from scipy.integrate for  numerical integration.
    
    return St * Q1 - np.exp(-r * tau) * K * Q2

start_time = time.time()
option_price = priceHestonCallViaOriginalFT(S0, v0, r, kappa, theta, sigma, rho, lambda_, tau, K)
end_time = time.time()
computing_time = end_time - start_time

print(f"The option price is (OriginalFT): {option_price}")
