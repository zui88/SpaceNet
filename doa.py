import numpy as np
import scipy.stats as stats


def compute_unambiguous_sources(eigs, n_samples, n_sensors):
    # exploiting the eigenstructure of the matrix
    # P.635; 14.111
    def mdl(i):
        noise = eigs[i:]
        am = np.mean(noise)
        gm = stats.gmean(noise)
        return -n_samples * (n_sensors - 1) * np.log(gm / am) + 1 / 2 * i * (2 * n_sensors - i) * np.log(n_samples)

    return np.argmin([mdl(i) for i in range(n_sensors - 1)])


"""
Rxx wrapper for holding
- cov
- n_samples
- n_sensors
"""
class Rxx():
    None


class Music(object):

    def __init__(self, steering_provider):
        self.steering_provider = steering_provider

    def get_doa(self, r):
        """
        DEEP ROOT MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION
        P. 1
        :param r:
        :return:
        """
        Rxx_hat        = self.compute_estimated_rcov(r)
        U, k_est       = self.compute_evd(Rxx_hat)
        Un             = self.compute_noise_subspace(U, k_est)
        spectrum       = self.compute_spectrum(Un)
        peaks          = self.find_peaks(spectrum, k_est)
        estimated_doas = self.compute_angles(peaks)

        return estimated_doas

    def compute_estimated_rcov(self, r):
        n_sensors, n_samples = r.shape
        Rxx_hat = Rxx()
        Rxx_hat.cov = (r @ r.conj().T) / n_samples
        Rxx_hat.n_samples = n_samples
        Rxx_hat.n_sensors = n_sensors
        return Rxx_hat

    def compute_evd(self, Rxx_hat):
        # eigen decomposition
        eigs, U = np.linalg.eig(Rxx_hat.cov)

        # sort eigvecs with eigvals in descending order
        idx = np.argsort(eigs)[::-1]
        eigs = eigs[idx]
        U = U[:, idx]

        k_est = compute_unambiguous_sources(eigs, Rxx_hat.n_samples, Rxx_hat.n_sensors)
        return U, k_est

    def compute_noise_subspace(self, eigsv, k_est):
        Un = eigsv[:, k_est:]
        return Un

    def compute_spectrum(self, Un):
        None

    def find_peaks(self, spectrum, k_est):
        None

    def compute_angles(self, peaks):
        None


class ClassicMusic(Music):

    def compute_spectrum(self, Un):
        # hypothesis: if angles are pointing to sources
        scan_range = np.linspace(-np.pi/2, np.pi/2, 360)
        a = self.steering_provider.get_steering(scan_range)

        # pseudo spectrum: 1 / (a^H(theta) * Un * Un^H * a(theta))
        projection = Un.conj().T @ a
        spectrum = 1 / np.sum(np.abs(projection)**2, axis=0)
        # cache scan grid for peak-to-angle mapping
        self._scan_range = scan_range
        return spectrum

    def find_peaks(self, spectrum, k_est):
        if k_est <= 0:
            return np.array()

        # local maxima indices
        left_neighbor     = lambda s: s[1:-1] > s[:-2]
        right_neighbor    = lambda s: s[1:-1] > s[2:]
        peaks             = np.where(left_neighbor(spectrum) & right_neighbor(spectrum))[0] # array of indices
        filtered_spectrum = spectrum[peaks]
        sort_fspec_idx    = np.argsort(filtered_spectrum)[::-1]
        top_peaks         = peaks[sort_fspec_idx][:k_est]
        return top_peaks

    def compute_angles(self, peaks):
        if peaks is None or len(peaks) == 0:
            return np.array()
        # peaks are indices into scan grid to retrieve the angeles
        angles = self._scan_range[np.asarray(peaks, dtype=int)]
        return np.rad2deg(angles)


class RootMusic(Music):

    def compute_spectrum(self, Un):
        F           = Un @ Un.conj().T
        M           = F.shape[0] - 1
        diag_coeffs = [np.sum(np.diag(F, l)) for l in range(-(M-1), M)]
        zeros       = np.roots(diag_coeffs[::-1])
        return zeros

    def find_peaks(self, spectrum, k_est):
        inner_roots = spectrum[np.abs(spectrum) < 1]
        delta       = 1 - np.abs(inner_roots)
        idx         = np.argsort(delta)
        return inner_roots[idx[:k_est]]

    def compute_angles(self, peaks):
        root_angles = np.angle(peaks)
        theta_est   = np.arcsin(root_angles / np.pi)
        return np.rad2deg(theta_est)
