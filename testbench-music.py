import numpy as np

rng = np.random.default_rng()

K_sources = 5
N_antennas = 8
N_samples = 10
SNR_db = 20
sigma2 = np.power(10, -SNR_db/10)
theta = rng.uniform(-np.pi/2, np.pi/2, K_sources)

##################################################
# constructing the incident/detected signals
# s_k,n = x_k,n + jy_k,n with x,y ~ N(0,1)
##################################################
s = np.sqrt(2) * (rng.standard_normal((K_sources, N_samples)) + 1j * rng.standard_normal((K_sources, N_samples)))

##################################################
# steering/mode vectors for "standard ULA"
# hypothesises Theta_i
##################################################
antennas_idx = np.arange(N_antennas)[:, None]
A = np.exp(1j * np.pi * np.sin(theta) * antennas_idx) # broadcast sin with numbers of antennas

##################################################
# sensed signal
##################################################
w = np.sqrt(sigma2 / 2) * (rng.standard_normal((N_antennas, N_samples)) + 1j * rng.standard_normal((N_antennas, N_samples)))
r = A @ s + w