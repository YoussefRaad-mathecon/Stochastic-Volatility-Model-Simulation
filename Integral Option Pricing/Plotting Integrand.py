import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# Define the characteristic function for the Heston model
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

# Parameters
kappa = 2
theta = 0.02
rho = -0.5
sigma = 0.3
tau = 0.5
r = 0
v0 = 0.01
S = 100
lambda_ = 0
K = 80

# Integrand functions for Q1 and Q2
def integrand_Q1(u):
    return  np.real(np.exp(-1j * u * np.log(K)) * characteristicFunctionHeston(u, S, v0, r, kappa, theta, sigma, rho, lambda_, tau, 1))

def integrand_Q2(u):
    return np.real(np.exp(-1j * u * np.log(K)) * characteristicFunctionHeston(u, S, v0, r, kappa, theta, sigma, rho, lambda_, tau, 2))

# Define upper bound limits
upper_bounds = np.linspace(0, 100, 500)

# Compute integrals
Q1_values = [quad(integrand_Q1, 0, ub)[0] for ub in upper_bounds]
Q2_values = [quad(integrand_Q2, 0, ub)[0] for ub in upper_bounds]

# Set LaTeX rendering for text with Computer Modern font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

# Plot the results
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(upper_bounds, Q1_values, label='$Q_1$ Integral', color='blue')
ax.plot(upper_bounds, Q2_values, label='$Q_2$ Integral', color='red')
ax.set_xlabel('$M$', fontsize=20)
ax.set_ylabel(r'$\int_{0}^{M} \Re\left\{ \frac{\exp\left\{-iu\log K\right\} \Psi_j(S,v,\tau;u)}{iu} \right\} du$', 
              ha='left', y=1.04, rotation=0, labelpad=20, fontsize=20)  # Increase fontsize here

ax.legend(fontsize=16)
ax.set_ylim(-1, 5)

# Set x-axis ticks to show every 0.2 increment
y_ticks = np.arange(-0.5, 5.5, 0.5)
ax.set_yticks(y_ticks)

x_ticks = np.arange(0, 105, 5)
ax.set_xticks(x_ticks)

# Remove plot borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.show()
