import numpy as np
import scipy.stats as stats


def compute_unambiguous_sources(eigs, n_samples, n_sensors):
    # exploiting the eigenstructure of the matrix
    # P.635; 14.111
    def mdl(i):
        noise = eigs[i:]
        am = np.mean(noise)
        gm = stats.gmean(noise)
        return -n_samples * (n_sensors - 1) * np.log(gm/am) + 1/2 * i * (2 * n_sensors - i) * np.log(n_samples)

    return np.argmin([mdl(i) for i in range(n_sensors - 1)])


def compute_noise_subspace(r):
    n_sensors, n_samples = r.shape

    # estimate cov matrix
    R_hat = (r @ r.conj().T) / n_samples

    # eigen decomposition
    eigs, vecs = np.linalg.eig(R_hat)

    # sort eigvals and eigvecs with descending order
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    vecs = vecs[:,idx]

    k_est = compute_unambiguous_sources(eigs, n_samples, n_sensors)

    # noise subspace base
    Un = vecs[:, k_est:]
    return Un


class Music(object):

    def __init__(self, steering_provider):
        self.steering_provider = steering_provider

    def get_doa(self, r):
        Un = compute_noise_subspace(r)
        return self.compute_spectrum(Un)

    def compute_spectrum(self, Un):
        # hypothesis: if angles are pointing to sources
        scan_range = np.linspace(-np.pi/2, np.pi/2, 360)
        a = self.steering_provider.get_steering(scan_range)

        # pseudo spectrum: 1 / (a^H(theta) * Un * Un^H * a(theta))
        projection = Un.conj().T @ a
        spectrum = 1 / np.sum(np.abs(projection)**2, axis=0)
        return spectrum, scan_range


class RootMusic(Music):

    def compute_spectrum(self, Un):
        F = Un @ Un.conj().T
        M = F.shape[0] - 1
        diag_coeffs = [np.sum(np.diag(F, l)) for l in range(-(M-1), M)]
        zeros = np.roots(diag_coeffs[::-1])
        return zeros

