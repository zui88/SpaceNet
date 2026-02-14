import numpy as np
import scipy.stats as stats


def mos_penalty(i, eigs, n_samples, n_sensors):
    """
    model order selection penalty term for mdl and aic

    :param i:
    :param eigs:
    :param n_samples:
    :param n_sensors:
    :return:
    """
    noise        = eigs[i:]
    am           = np.mean(noise)
    gm           = stats.gmean(noise)
    penalty_term = -1 * (n_samples * (n_sensors - 1) * np.log(gm / am))
    return penalty_term

def mdl(i, eigs, n_samples, n_sensors):
    """
    P.635; 14.111

    :param i:
    :param eigs:
    :param n_samples:
    :param n_sensors:
    :return:
    """
    penalty = mos_penalty(i, eigs, n_samples, n_sensors)
    return penalty + 0.5 * i * (2 * n_sensors - i) * np.log(n_samples)


def aic(i, eigs, n_samples, n_sensors):
    """
    P.635; 14.110

    :param i:
    :param eigs:
    :param n_samples:
    :param n_sensors:
    :return:
    """
    penalty = mos_penalty(i, eigs, n_samples, n_sensors)
    return penalty + i * (2 * n_sensors - i)


def compute_unambiguous_sources(eigs, n_samples, n_sensors, mos_algo=mdl):
    """
    the number of sources is the minimizing value of the model order selection i el_of {1, ..., P_max},
    where P_max is the number of sensors minus one
    P.635

    :param eigs:
    :param n_samples:
    :param n_sensors:
    :param mos_algo:
    :return:
    """
    k_est = np.argmin([mos_algo(i, eigs, n_samples, n_sensors) for i in range(n_sensors - 1)])
    return k_est


"""
class Rxx_hat: wrapper for covariance matrix internals
cov: covariance matrix
n_samples: number of samples
n_sensors: number of sensors
"""
class Rxx():
    None


class Music(object):

    def __init__(self, steering_provider):
        self.steering_provider = steering_provider

    def compute_doa(self, r):
        """
        DEEP ROOT MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION
        P. 1

        :param r: sensed signal at the antennas
        :return: thetas of the estimated sources in degrees
        """
        Rxx_hat        = self.compute_estimated_rcov(r)
        U, k_est       = self.compute_evd(Rxx_hat)
        Un             = self.compute_noise_subspace(U, k_est)
        spectrum       = self.compute_spectrum(Un)
        peaks = self.identify_signal_sources(spectrum, k_est)
        estimated_doas = self.compute_angles(peaks)

        return estimated_doas

    def compute_estimated_rcov(self, r):
        """

        :param r:
        :return:
        """
        n_sensors, n_samples = r.shape
        Rxx_hat = Rxx()
        Rxx_hat.cov = (r @ r.conj().T) / n_samples
        Rxx_hat.n_samples = n_samples
        Rxx_hat.n_sensors = n_sensors
        return Rxx_hat

    def compute_evd(self, Rxx_hat):
        """

        :param Rxx_hat:
        :return:
        """
        # eigen decomposition
        eigs, U = np.linalg.eig(Rxx_hat.cov)

        # sort eigvecs with eigvals in descending order
        idx = np.argsort(eigs)[::-1]
        eigs = eigs[idx]
        U = U[:, idx]

        k_est = compute_unambiguous_sources(eigs, Rxx_hat.n_samples, Rxx_hat.n_sensors)
        return U, k_est

    def compute_noise_subspace(self, eigsv, k_est):
        """
        exploiting the eigenstructure and the fact that the noise is orthogonal
        to the signal space to segregate the noise space

        :param eigsv: the sorted eigenvectors of the estimated covariance matrix: lambda_1 >= lambda_2 >= ... lambda_m
        :param k_est: the estimated number of sources
        :return: the noise subspace: (Un_1, ..., Un_l)^T
        """
        Un = eigsv[:, k_est:]
        return Un

    def compute_spectrum(self, Un):
        """
        generate the space where the estimated doas can be found by applying hypotheses about the doas of the signals

        :param Un: the estimated noise subspace
        :return: the space of possible doas (root-musik: roots; classic music: spectrum values)
        """
        None

    def identify_signal_sources(self, spectrum, k_est):
        """
        Identifies k_est strongest signal source directions from the spectrum

        :param spectrum: the space of possible signal sources (root-musik: roots; classic music: spectrum values)
        :param k_est: the estimated number of sources
        :return: the signal source information (root-musik: roots; classic music: indices)
        """
        None

    def compute_angles(self, peaks):
        """
        converts the peak information into angles

        :param peaks: meaningful information about the peaks (root-musik: roots; classic music: indices)
        :return: thetas of the estimated signals in degrees
        """
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

    def identify_signal_sources(self, spectrum, k_est):
        if k_est <= 0:
            return np.array()

        # local maxima indices
        gt_left_neighbor  = lambda s: s[1:-1] > s[:-2]
        gt_right_neighbor = lambda s: s[1:-1] > s[2:]
        peaks_idx         = np.where(gt_left_neighbor(spectrum) & gt_right_neighbor(spectrum))[0] # array of indices
        filtered_spectrum = spectrum[peaks_idx]
        sort_fspec_idx    = np.argsort(filtered_spectrum)[::-1]
        top_peaks_idx     = peaks_idx[sort_fspec_idx][:k_est]
        return top_peaks_idx

    def compute_angles(self, peaks):
        if peaks is None or len(peaks) == 0:
            return np.array()
        # peaks are indices into scan grid to retrieve the angeles
        angles = self._scan_range[peaks]
        return np.rad2deg(angles)


class RootMusic(Music):

    def compute_spectrum(self, Un):
        F              = Un @ Un.conj().T
        M              = F.shape[0]
        diag_coeffs    = [np.sum(np.diag(F, l)) for l in range(-(M-1), M)]
        coeffs_ordered = diag_coeffs[::-1] # sort the polys from z^n, z^n-1, ..., z^0
        roots          = np.roots(coeffs_ordered)
        return roots

    def identify_signal_sources(self, spectrum, k_est):
        inner_roots = spectrum[np.abs(spectrum) < 1]
        delta       = 1 - np.abs(inner_roots)
        idx         = np.argsort(delta)
        return inner_roots[idx[:k_est]]

    def compute_angles(self, peaks):
        root_angles = np.angle(peaks)
        theta_est   = np.arcsin(root_angles / np.pi)
        return np.rad2deg(theta_est)
