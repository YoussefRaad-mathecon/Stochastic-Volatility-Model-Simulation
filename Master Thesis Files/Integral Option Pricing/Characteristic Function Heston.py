import numpy as np
import matplotlib.pyplot as plt
np.random.seed(666)

plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath,amsfonts}'

params = {
    'St': 100,
    'vt': 0.06,
    'r': 0.05,
    'kappa': 1,
    'theta': 0.06,
    'sigma': 0.3,
    'rho': -0.5,
    'lambda_': 0.01,
    'tau': 1
}

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
    elif j == 2:
        b_j = b2
        u_j = u2
    else:
        pass
    d_j = np.sqrt((rho * sigma * i * u - b_j)**2 - sigma**2 * (2 * u_j * i * u - u**2))
    g_j = (b_j - rho * sigma * i * u + d_j) / (b_j - rho * sigma * i * u - d_j)

    C_j = r * i * u * tau + (a / sigma**2) * ((b_j - rho * sigma * i * u + d_j) * tau - 2 * np.log((1 - g_j * np.exp(d_j * tau)) / (1 - g_j)))
    D_j = ((b_j - rho * sigma * i * u + d_j) / sigma**2) * ((1 - np.exp(d_j * tau)) / (1 - g_j * np.exp(d_j * tau)))

    return np.exp(C_j + D_j * vt + i * u * np.log(St))


u_values = np.linspace(-20, 20, 400)


Psi1_values = [characteristicFunctionHeston(u, j=1, **params) for u in u_values]
Psi2_values = [characteristicFunctionHeston(u, j=2, **params) for u in u_values]

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(r'\textbf{Real and Imaginary Parts of $\Psi_1(u)$ and $\Psi_2(u)$ for $u \in [-20, 20]$}', fontsize=22)

#Real part Psi1
axs[0, 0].plot(u_values, np.real(Psi1_values), label='Real Part of $\Psi_1(u)$', color='blue')
axs[0, 0].legend()
axs[0, 0].set_xlabel('$u$', fontsize=18) 
axs[0, 0].set_ylabel('$\mathfrak{Re}(\Psi_1(u))$', fontsize=18)  

#Imag part Psi1
axs[0, 1].plot(u_values, np.imag(Psi1_values), label='Imaginary Part of $\Psi_1(u)$', color='orange')
axs[0, 1].legend()
axs[0, 1].set_xlabel('$u$', fontsize=18)
axs[0, 1].set_ylabel('$\mathfrak{Im}(\Psi_1(u))$', fontsize=18) 

#Real part Psi2
axs[1, 0].plot(u_values, np.real(Psi2_values), label='Real Part of $\Psi_2(u)$', color='green')
axs[1, 0].legend()
axs[1, 0].set_xlabel('$u$', fontsize=18)  
axs[1, 0].set_ylabel('$\mathfrak{Re}(\Psi_2(u))$', fontsize=18)  

#Imag part Psi2
axs[1, 1].plot(u_values, np.imag(Psi2_values), label='Imaginary Part of $\Psi_2(u)$', color='red')
axs[1, 1].legend()
axs[1, 1].set_xlabel('$u$', fontsize=18) 
axs[1, 1].set_ylabel('$\mathfrak{Im}(\Psi_2(u))$', fontsize=18)  

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
