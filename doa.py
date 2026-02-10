import numpy as np
import scipy.stats as stat

class Music(object):
    def __init__(self, steering_provider):
        self.steering_provider = steering_provider

    def compute_unambiguous_sources(self, eigs, n_samples):
        # exploiting the eigenstructure of the matrix
        # P.635; 14.111
        def mdl(eigs, i, n_sensors, n_samples):
            noise = eigs[i:]
            am = np.mean(noise)
            gm = stat.gmean(noise)
            return -n_samples * (n_sensors - 1) * np.log(gm/am) + 1/2 * i * (2 * n_sensors - i) * np.log(n_samples)

        n_sensors = self.steering_provider.get_nsensores()
        return np.argmin([mdl(eigs, i, n_sensors, n_samples) for i in range(n_sensors - 1)])


    def compute_doa(self, r):
        n_samples = r.shape[1]

        # estimate cov matrix
        R_hat = (r @ r.conj().T) / n_samples

        # eigen decomposition
        eigs, vecs = np.linalg.eig(R_hat)

        # sort eigvals and eigvecs with descending order
        idx = np.argsort(eigs)[::-1]
        eigs = eigs[idx]
        vecs = vecs[:,idx]

        # determine number of sources
        k_est = self.compute_unambiguous_sources(eigs, n_samples)

        # noise subspace base
        Un = vecs[:, k_est:]

        # hypothesis: if angles are pointing to sources
        scan_range = np.linspace(-np.pi/2, np.pi/2, 360)
        a = self.steering_provider.get_steering(scan_range)

        # pseudo spectrum: 1 / (a^H(theta) * Un * Un^H * a(theta))
        projection = Un.conj().T @ a
        spectrum = 1 / np.sum(np.abs(projection)**2, axis=0)

        return spectrum, scan_range