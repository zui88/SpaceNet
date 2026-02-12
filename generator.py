import numpy as np

class ULASignalGenerator:

    def __init__(self, n_antennas = 8, d_lambda = 1):
        ##################################################
        # d_lambda = d / lambda
        ##################################################
        self.rng = np.random.default_rng()
        self.n_antennas = n_antennas
        self.d_lambda = d_lambda

    def get_steering(self, thetas):
        ##################################################
        # steering/mode vectors for "standard ULA"
        # hypothesises Theta_i
        # A.shape() = (antennas x sources)
        ##################################################
        antennas_idx = np.arange(self.n_antennas)[:, None]
        A = np.exp(1j * np.pi * self.d_lambda * np.sin(thetas) * antennas_idx)  # broadcast sin with numbers of antennas
        return A

    def get_incident_signals(self, k_sources, n_samples):
        ##################################################
        # constructing the incident signals
        # s_k,n = x_k,n + jy_k,n with x,y ~ N(0,1)
        # just random information to be sent (doesn't matter in this case)
        ##################################################
        x = y = self.rng.standard_normal((k_sources, n_samples))
        s = 1/np.sqrt(2) * (x + 1j*y)
        return s

    def generate(self, thetas, n_samples=10, snr_db=20):
        ##################################################
        # sensed signal
        ##################################################
        k_sources = len(thetas)
        sigma2 = np.power(10, -snr_db/10)

        x = y = self.rng.standard_normal((self.n_antennas, n_samples))
        w = np.sqrt(sigma2 / 2) * (x + 1j*y)
        A = self.get_steering(thetas)
        s = self.get_incident_signals(k_sources, n_samples)
        return A @ s + w

