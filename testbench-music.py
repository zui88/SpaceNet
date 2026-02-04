import numpy as np
import scipy as sp

rng = np.random.default_rng()

K = 5
N_antennas = 8
N_samples = 10
d_norm = 0.5 # d / lambda
SNR_db = 20
sigma2 = np.power(10, -SNR_db/10)

s = np.square(0.5) * (rng.standard_normal((K,N_samples)) + 1j * rng.standard_normal((K,N_samples)))
theta = rng.random((K))
noise = np.sqrt(sigma2 * 0.5) * ( rng.standard_normal((N_antennas,N_samples)) + 1j * rng.standard_normal((N_antennas,N_samples)))
r = np.zeros((N_antennas,N_samples), dtype=np.complex128)
n_vec = np.arange(N_antennas)[:,None]
for n in range(N_samples):
    r[:,n:n+1] = np.exp(1j * 2 * np.pi * d_norm * np.sin(theta[:,None].T) * n_vec) @ s[:,n:n+1] + noise[:,n:n+1]
