# Stochastiv Volatility Model Option Pricing
This repository contains my code for simulation of asset prices under the Heston model and OTHER numerical approximations. The code is mostly avaliable in python.

## HestonModel.py class
This class will contain most (if not all) of the implemented methods: Simulation, Monte Carlo Pricing and Exact Pricing.

### Simulations
The implemented Monte Carlo simulation schemes (so far):
- Euler
- Milstein
- Truncated-Gaussian (TG)
- Truncated-Gaussian (TG+M)
- Quadratic-Exponential (QE)
- Quadratic-Exponential (QE-M)
- Failed Broadie and Kaya Drift Interpolation
### Exact Pricing
The implemented exact pricing methods:
- Heston's original
- Heston's stable Original
- Carr-Madan.
