import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import scipy
import scipy.stats as stats
import tensorflow as tf
from tensorflow import keras

from generator import ObservationContext
from utils import eigh, gradient


def find_roots(coeffs: tf.Tensor) -> tf.Tensor:
    """
    Construct the companion matrix (Begleitmatrix eines Polynoms) to compute the roots.
    The transposed matrix is used (Kardinalform).

    https://de.wikipedia.org/wiki/Begleitmatrix

    PARAMETERS
    ----------
    coeffs : tf.Tensor
        coefficients of the polynomial
    
    RETURNS
    -------
    roots : tf.Tensor
        roots of the polynomial
    """
    coeffs    = tf.cast(coeffs, dtype=tf.dtypes.complex128)  # need to cast, since there are some precision issues in non-eager-mode
    first_row = tf.reshape(-coeffs[1:] / coeffs[0], [1, -1]) # c_n-1 coffs
    # construct a minor diag matrix with ones
    C         = tf.keras.ops.diag(tf.keras.ops.ones(coeffs.shape[0] - 2, dtype=coeffs.dtype), k=-1)
    # cut the last column and append the coff column instead
    C         = tf.concat([first_row, C[1:,:]], axis=0)
    # eigs are the roots because: det(lambda I - C) <=> p(lambda) = 0
    roots     = tf.linalg.eigvals(C)
    return roots


def sample_autocov(X: tf.Tensor, tau: int) -> tf.Tensor:
    """
    Calculates the sample autocovariance matrix R(tau) for a given time lag tau.

        DEEP ROOT MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION; P. 3
        Rx = 1/(T - tau) sum(X(t) X(t+tau)^H) with t = 0, ..., T-tau

    :param X:   Complex-valued array of shape (M,T) containing M antenna signals over T time samples
    :param tau: Time lag value to calculate autocovariance
    :return:    Matrix of shape (M,M) containing the autocovariance matrix; number of samples T
    """
    M, T = X.shape
    X1   = X[:, :T-tau] # x(t)
    X2   = X[:, tau:T]  # x(t+tau)
    R    = tf.matmul(X1, X2, adjoint_b=True)
    return R, T


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
cov_batch: covariance matrix; Size: (batch_size, n_antennas, n_antennas)
n_samples_batch: number of samples; Size: (batch_size,)
n_sensors_batch: number of sensors; Size: (batch_size,)
"""
@dataclass
class Rxx(object):
    cov_batch : tf.Tensor
    n_samples_batch : tf.Tensor
    m_sensors_batch : tf.Tensor


@dataclass
class DoaEstimate:
    angles : np.ndarray
    spectrum : np.ndarray


@dataclass
class DelayDopplerEstimate:
    delays : np.array
    dopplers : np.array
    tau_grid : np.array
    cost_function : np.array


class InterfaceMusic(ABC):

    
    def __init__(self,
                 d_sources : int = None,
                 inference_mode : bool = True,
                 steering_provider = None,
                 signal_provider = None):

        """
        PARAMETERS
        ----------
        d_sources : int = None
            the estimated number of sources; just takes effect if inference_mode is False
        
        inference_mode : bool = True
            if False, the algorithm will not compute the estimated sources, instead d_sources will be used

        steering_provider = None

        signal_provider = None
        """

        self.inference_mode = inference_mode
        self.d_sources      = d_sources

        self.steering_provider = steering_provider
        self.signal_provider   = signal_provider
        self.m_antennas        = None
        self.n_samples         = None

        if self.steering_provider:
            self.m_antennas = self.steering_provider.m_antennas

        if self.signal_provider:
            self.n_samples = self.signal_provider.n_samples


    @abstractmethod
    def estimate(self, r_batched : np.ndarray):
        """
        Parameters
        ----------

        r_batched : np.ndarray
            multidimensional batched array of the sensed signal
        
        """
        pass


class BaseMusicRecipe(InterfaceMusic):

    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def estimate(self, r_batched: np.ndarray):
        """
        DEEP ROOT MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION
        P. 1

        The implementation leverages batch processing.

        :param r_batched: sensed signal at the antennas; Size: (batch_size, n_antennas, n_samples)
        :param inference_mode: if False, the algorithm will not compute the estimated sources
        :return: batched thetas of the estimated sources in degrees; Size: (batch_size, n_hat_incident_sources)
        """
        Rxx_hat        = self.compute_estimated_rcov(r_batched)
        U, eigs, k_est = self.compute_evd(Rxx_hat)
        Un             = self.compute_noise_subspace(U, eigs, k_est)
        spectrum       = self.compute_pseudo_inverse_spectrum(Un)
        peaks          = self.identify_signal_sources(spectrum, k_est)
        estimated_doas = self.compute_parameter_vector(peaks)

        return estimated_doas

    
    def compute_estimated_rcov(self, r_batched):
        """

        :param r_batched: Size: (batch_size, n_antennas, n_samples)
        :return: Rxx_hat: estimated covariance matrix of the signal; Size: (batch_size, n_antennas, n_antennas)
        """
        _, n_sensors, n_samples = r_batched.shape
        cov_batched             = []
        n_samples_batched       = []
        n_sensors_batched       = []

        for r in r_batched:
            cov = tf.matmul(r, r, adjoint_b=True) / n_samples
            cov_batched.append(cov)
            n_samples_batched.append(n_samples)
            n_sensors_batched.append(n_sensors)

        return Rxx(
            tf.stack(cov_batched),
            tf.stack(n_samples_batched),
            tf.stack(n_sensors_batched)
        )

    
    def compute_evd(self, Rxx_hat):
        """

        :param Rxx_hat: Size: (batch_size, n_antennas, n_antennas)
        :return: U_batched: estimated eigenvectors of the estimated covariance matrix; Size: (batch_size, n_antennas, n_antennas)
        :return: k_est_batched: estimated number of sources; Size: (batch_size,)
        """
        U_batched     = []
        eigs_batched  = []
        k_est_batched = []

        for cov, n_samples, m_sensors in zip(Rxx_hat.cov_batch, Rxx_hat.n_samples_batch, Rxx_hat.m_sensors_batch):
            eigs, U = tf.linalg.eigh(cov)
            # sort eigvecs with eigvals in descending order
            idx     = tf.keras.ops.argsort(tf.abs(eigs))[::-1]
            eigs    = tf.gather(eigs, idx)
            U       = tf.gather(U, idx, axis=1)
            k_est   = self.d_sources
            if self.inference_mode:
                # not suitable for keras training procedure
                k_est = compute_unambiguous_sources(eigs.numpy(), n_samples.numpy(), m_sensors.numpy())

            k_est_batched.append(k_est)
            U_batched.append(U)
            eigs_batched.append(eigs)

        return tf.stack(U_batched), tf.stack(eigs_batched), tf.stack(k_est_batched)

    
    def compute_noise_subspace(self, eigsv_batched, eigs_batched, k_est_batched):
        """
        exploiting the eigenstructure and the fact that the noise is orthogonal
        to the signal space to segregate the noise space

        :param eigsv: the sorted eigenvectors of the estimated covariance matrix: lambda_1 >= lambda_2 >= ... lambda_m
        :param k_est: the estimated number of sources
        :return: the noise subspace: (Un_1, ..., Un_l)^T
        """
        Un_batched = []

        for eigsv, k_est in zip(eigsv_batched, k_est_batched):
            Un = eigsv[:, k_est:]
            Un_batched.append(Un)

        return tf.stack(Un_batched)

    
    @abstractmethod
    def compute_pseudo_inverse_spectrum(self, Un_batched: tf.Tensor):
        """
        generate the space where the estimated parameters can be found by applying hypotheses

        :param Un_batched:
        :param Un: the estimated noise subspace
        :return: the space of possible doas (root-musik: roots; classic music: spectrum values)
        """
        None

    
    @abstractmethod
    def identify_signal_sources(self, spectrum_batched: tf.Tensor, k_est_batched: tf.Tensor):
        """
        Identifies k_est strongest signal source directions from the spectrum

        :param k_est_batched:
        :param spectrum_batched:
        :param spectrum: the space of possible signal sources (root-musik: roots; classic music: spectrum values)
        :param k_est: the estimated number of sources
        :return: the signal source information (root-musik: roots; classic music: indices)
        """
        None

    
    def compute_parameter_vector(self, peaks_batched: tf.Tensor):
        """
        Converts the peak information into angles.  The default behaviour is simply pass through the incomming peaks.  That is reasonable in the case of a deep MUSIC implementation, since in such case the ML algorithm converts implicitly the peaks estimation in the right physical entities.  For all other approaches it is recommended to overwrite this method, because it's almost not what you want.

        :param peaks_batched:
        :param peaks: meaningful information about the peaks (root-musik: roots; classic music: indices)
        :return: thetas of the estimated signals in degrees
        """
        angles = peaks_batched
        return angles


class DelayDopplerMusic(BaseMusicRecipe):
    """
    Using a fixed pipeline therefore no evaluation of
    - inference_mode
    """

    def __init__(self, observ_ctx : ObservationContext, *args, **kwargs):
        BaseMusicRecipe.__init__(self, *args, **kwargs)
        """
        Parameters
        ----------

        d_sources : int
            number of incident source from where the delay and doppler will be estimated
        
        """
        self.observ_ctx = observ_ctx


    def estimate(self, r_batched : np.ndarray):
        estimates_batched = []

        for r in r_batched:
            estimates_batched.append(self.compute(r))

        return np.stack(estimates_batched)


    def compute(self, r):
        N = r.shape[0]          # samples
        M = r.shape[1]          # antennas


        # --- transform to frequency space ---
        R = np.fft.fftshift(np.fft.fft(r, axis=0), axes=0)     # (NxM)

        # --- covariance over the sample space N ---
        R_cov = R @ R.conj().T / M

        ################################################################
        # MUSIC
        ################################################################
        # --- eigendecomposition ---
        _, Un = scipy.linalg.eigh(a=R_cov, subset_by_index=[0,(N-self.d_sources-1)])

        # --- MUSIC scan over tau; building the cost function ---
        T_obs         = self.observ_ctx.window_length
        tau_grid      = np.linspace(0, T_obs, N)
        cost_function = np.zeros_like(tau_grid)
        omega_grid    = np.zeros_like(tau_grid)

        # --- signal preparation ---
        fs = self.signal_provider.fs
        f  = np.fft.fftshift(np.fft.fftfreq(N, 1/fs)) # (Nx1)
        w  = f*2*np.pi                                # (Nx1)
        s  = self.signal_provider.generate(n_samples=N)
        S  = np.fft.fftshift(np.fft.fft(s))           # (Nx1)
        dS = gradient(S, w, numpy=True)               # (Nx1)

        # --- compute B because it's not dependant on tau ---
        B_mat = np.real(np.array([[S.conj().T @ S, -S.conj().T @ dS],
                                  [-dS.conj().T @ S, dS.conj().T @ dS]]))

        # --- go over the grid and compute the cost function to be minimized ---
        for i, tau in enumerate(tau_grid):

            # build G(tau)
            v_tau = np.exp(-1j * w * tau)
            G     = np.stack([S * v_tau, -dS * v_tau], axis=1)
            A_mat = np.real(G.conj().T @ Un @ Un.conj().T @ G)

            lambdas, gammas = eigh(a=A_mat, b=B_mat, subset_by_index=[0,1])

            lambda_min       = lambdas[0]
            cost_function[i] = lambda_min
            gamma_min        = gammas[:,0]
            omega_grid[i]    = gamma_min[1] / gamma_min[0]

        # --- results ---
        idx       = np.argsort(cost_function)[:self.d_sources]
        tau_est   = tau_grid[idx]
        omega_est = omega_grid[idx]

        return DelayDopplerEstimate(
            tau_est,
            omega_est,
            tau_grid,
            cost_function,
        )

    
    def compute_pseudo_inverse_spectrum(self):
        pass


    def identify_signal_sources(self):
        pass


def create_permututation_from_estimated_sources(predictions : tf.Tensor) -> tf.Tensor:
    """
    Produce a permutational set of the predictions from k estimated sources.

    PARAMETERS
    ----------
    predictions
        size (k_est x parameters)
    """
    perms = list(itertools.permutations(range(predictions.shape[0])))
    for perm in perms:
        # don't know how many K because of faculty!
        yield tf.gather(predictions, perm)


class DAMusicLoss(keras.losses.Loss):


    @abstractmethod
    def compute_error(self, ground_truth, predictions):
        raise RuntimeError(
            "Deep Augmented MUSIC Loss function not implemented"
        )
    

    def call(self, ground_truth_batched, predictions_batched):
        loss_min_batched = []

        for ground_truth, predictions in zip(ground_truth_batched, predictions_batched):
            loss_permuted = []
            for perm_pred in create_permututation_from_estimated_sources(predictions):
                error = self.compute_error(ground_truth, perm_pred)
                error = tf.math.sqrt(tf.reduce_mean((tf.math.pow(error, 2)), axis=-1))
                loss_permuted.append(error)
            loss_min_batched.append(tf.keras.ops.min(tf.stack(loss_permuted)))

        return tf.stack(loss_min_batched)


class RMSELoss(DAMusicLoss):
    """
    Root Means Square Error
    """

    def __init__(self, gain : float = 1):
        super().__init__()
        self.gain = gain


    # def compute_error(self, ground_truth, predictions):
    #     gt_delay, gt_doppler = ground_truth
    #     delay, doppler       = predictions

    #     delta_delay   = tf.keras.ops.power(gt_delay - delay, 2)
    #     delta_doppler = tf.keras.ops.power(gt_doppler - doppler, 2)

    #     return tf.keras.ops.sqrt(delta_doppler * self.doppler_weight + delta_delay)


    def compute_error(self, ground_truth, predictions):
        gt_delay    = ground_truth[:, 0]
        delay       = predictions
        delta_delay = (gt_delay - delay) * self.gain
        return delta_delay


class DAv1MusicBase(keras.Model, BaseMusicRecipe):

    def __init__(self,
                 eps               : float = 1.0,
                 training          : bool  = False,
                 transfer_learning : bool  = False,
                 model_name        : str   = "unknown",
                 models_to_train   : dict  = {"surrogate":True, "selector":True, "finder":True,},
                 scan_range        : int   = None,
                 *kargs, **kwargs):
        """

        Parameters
        ----------
        eps
        training
        transfer_learning
        model_name
        models_to_train
        kargs
        kwargs
        """

        super().__init__()
        BaseMusicRecipe.__init__(self, *kargs, **kwargs)

        self.scan_range = scan_range

        self.model_base_name        = "modelv1_" + model_name + "_"
        self.model_suffix           = ".keras"
        self.surrogate_network_name = "surrogate"
        self.selector_network_name  = "selector"
        self.finder_network_name    = "finder"

        self.training          = training
        self.transfer_learning = transfer_learning
        self.models_to_train   = models_to_train
        self.eps               = eps

        self.surrogate_network = None
        self.selector_network  = None
        self.finder_network    = None

        if self.training:
            if self.transfer_learning:
                self._preload()
            else:
                self.surrogate_network = self._build_surrogate_network()
                self.selector_network  = self._build_selector_network()
                self.finder_network    = self._build_finder_network()
        else:
            self._preload()


    def save(self):
        self.surrogate_network.save(self.model_base_name + self.surrogate_network_name + self.model_suffix)
        self.selector_network.save(self.model_base_name + self.selector_network_name + self.model_suffix)
        self.finder_network.save(self.model_base_name + self.finder_network_name + self.model_suffix)


    def _preload(self):
        try:
            self.surrogate_network = keras.models.load_model(self.model_base_name + self.surrogate_network_name + self.model_suffix)
            if not self.models_to_train[self.surrogate_network_name]:
                self.surrogate_network.trainalbe = False
        except FileNotFoundError as e:
            print(f"surrogate matrix network -- {self.model_base_name}: {e}!")
            exit()

        try:
            self.selector_network = keras.models.load_model(self.model_base_name + self.selector_network_name + self.model_suffix)
            if not self.models_to_train[self.selector_network_name]:
                self.selector_network.trainalbe = False
        except FileNotFoundError as e:
            print(f"selector network -- {self.model_base_name}: {e}!")
            exit()

        try:
            self.finder_network = keras.models.load_model(self.model_base_name + self.finder_network_name + self.model_suffix)
            if not self.models_to_train[self.finder_network_name]:
                self.finder_network.trainalbe = False
        except FileNotFoundError as e:
            print(f"delay finder network -- {self.model_base_name}: {e}!")
            exit()


    def call(self, inputs):
        r_sensed    = inputs
        predictions = self.estimate(r_sensed)
        return predictions


    @abstractmethod
    def _build_surrogate_network(self):
        pass


    @abstractmethod
    def _build_selector_network(self):
        pass


    @abstractmethod
    def _build_finder_network(self):
        pass


class DeepMusic(DAv1MusicBase):


    def __init__(self, model_name : str = "classic", scan_range : int = 360, **kwargs):
        super().__init__(
            model_name=model_name,
            scan_range=scan_range,
            **kwargs)


    def _build_surrogate_network(self):
        return keras.Sequential([
            keras.layers.Input((2 * self.m_antennas, self.n_samples)),
            keras.layers.BatchNormalization(),
            keras.layers.GRU(2 * self.m_antennas),
            keras.layers.Dense((2 * self.m_antennas * self.m_antennas)),
            keras.layers.Reshape((2 * self.m_antennas, self.m_antennas)),
        ])


    def _build_selector_network(self):
        return keras.Sequential([
            keras.Input((2 * self.m_antennas,)),
            keras.layers.Dense(2 * self.m_antennas, activation='relu'),
            keras.layers.Dense(2 * self.m_antennas, activation='relu'),
            keras.layers.Dense(2 * self.m_antennas, activation='relu'),
            keras.layers.Dense(self.m_antennas),
        ])


    def _build_finder_network(self):
        return keras.Sequential([
            keras.layers.Input((self.scan_range,)),
            keras.layers.Dense(2 * self.m_antennas, activation='relu'),
            keras.layers.Dense(2 * self.m_antennas, activation='relu'),
            keras.layers.Dense(2 * self.m_antennas, activation='relu'),
            keras.layers.Dense(self.d_sources),
        ])


    def compute_estimated_rcov(self, r_batched):
        """

        Parameters
        ----------
        r_batched

        Returns
        -------

        """
        real       = tf.math.real(r_batched)
        imag       = tf.math.imag(r_batched)
        surrogate  = self.surrogate_network(tf.concat([real, imag], axis=1))
        batch_size = surrogate.shape[0]

        M     = surrogate.shape[-1]
        split = surrogate.shape[-1]
        real  = surrogate[:, split :, :]
        imag  = surrogate[:, : split, :]

        # enforce hermitian symmetry: R = R^H
        cov = tf.complex(real, imag)
        cov = tf.matmul(cov, cov, adjoint_b=True)
        tf.vectorized_map(lambda c: c + tf.eye(M, dtype=tf.complex64) * self.eps, cov)
        return Rxx(
            cov_batch=cov,
            n_samples_batch=tf.constant([0 for _ in tf.range(batch_size)]),
            m_sensors_batch=tf.constant([0 for _ in tf.range(batch_size)]),
        )


    def compute_pseudo_inverse_spectrum(self, Un_batched):
        spectrum_batched = []
        self._scan_range = []

        for Un in Un_batched:
            # hypothesis: if angles are pointing to sources
            scan_range = np.linspace(-np.pi / 2, np.pi / 2, self.scan_range)
            a = self.steering_provider.get_steering(scan_range)

            # pseudo spectrum: 1 / (a^H(theta) * Un * Un^H * a(theta))
            projection = tf.matmul(Un, a, adjoint_a=True)
            spectrum = 1 / tf.keras.ops.sum(np.abs(projection)**2, axis=0)
            # cache scan grid for peak-to-angle mapping
            self._scan_range.append(scan_range)
            spectrum_batched.append(spectrum)

        return tf.stack(spectrum_batched)


    def compute_noise_subspace(self, eigsv_batched, eigs_batched, k_est_batched):
        """
        PARAMETERS
        ----------
        eigsv_batched
            batched eigenvectors

        eigs_batched
            batched eigenvalues

        k_est_batched
            batched number of sources

        RETURNS
        -------
            noise_subspace
        """
        real       = tf.math.real(eigs_batched)
        imag       = tf.math.imag(eigs_batched) # is aloways zero because of hermitian structure of the r cov
        q_batched  = self.selector_network(tf.keras.ops.append(real, imag, axis=1))
        q_batched  = tf.expand_dims(q_batched, axis=-1)
        Un_batched = tf.complex(
            q_batched * tf.math.real(eigsv_batched),
            q_batched * tf.math.imag(eigsv_batched),
            )
        return Un_batched


    def identify_signal_sources(self, spectrum_batched : tf.Tensor, k_est_batched : tf.Tensor):
        """
        since the pipeline is compiled / trained for a fixed number of signal sources it states true that all estimated ks holds the same value henceforth k_1 = k_2 = ... = k_n

        RETURN
        ------
            batched indicies of the tensor given from the peak finder network
        """
        # these are all real values
        delays_batched = self.finder_network(spectrum_batched)
        return delays_batched


    def compute_parameter_vector(self, delays_batched : tf.Tensor):
        """
        just a pass through
        """
        return delays_batched


class DeepDopplerMusic(DAv1MusicBase):
    """
    This class implements the abstract MUSIC base class and exploiting subspace method for estimating time delay and doppler shifts.
    In contrast to classic MUSIC where the array mode vector is used to estimate direction of arivals here signam mode vector is
    deployed for estimating signal parameters.  That means the roles are kind of switched compared to the classic MUSIC approach.
    """
    
    
    def __init__(self, observ_ctx : ObservationContext, scan_range : int = 400, m_antennas : int = None, *args, **kwargs):
        """
        The modelparameters (hyperparameters) will be passed on which the model is trained to.  As a consequence for each distinct set of parameters a whole new training and model fit procedure must be executed (for complex scenarios one single training session could be very expensive).

        PARAMETERS
        ----------
        scan_range : int
            Defines the range where the hypothesis is applied, also the points in the pseudo spectrum.  The literature ofthen denotes the dimensionality as R samples (constructing the pseudo spectrum).
        """
        super().__init__(
            scan_range=scan_range,
            *args, **kwargs)

        self.observ_ctx                 = observ_ctx
        self.m_antennas                 = m_antennas
        self.fs                         = self.signal_provider.fs
        self.T                          = self.observ_ctx.window_length
        self.n_samples                  = int(self.fs * self.T)
        self._cached_spectrum_constants = None
        

    def _build_surrogate_network(self):
        return keras.Sequential([
            keras.Input((2 * self.n_samples, self.m_antennas)),
            keras.layers.BatchNormalization(),
            keras.layers.GRU(2 * self.n_samples),
            keras.layers.Dense((2 * self.n_samples * self.n_samples)),
            keras.layers.Reshape((2 * self.n_samples, self.n_samples)),
        ])


    def _build_selector_network(self):
        return keras.Sequential([
            keras.Input((2 * self.n_samples,)),
            keras.layers.Dense(2 * self.n_samples, activation='relu'),
            keras.layers.Dense(2 * self.n_samples, activation='relu'),
            keras.layers.Dense(2 * self.n_samples, activation='relu'),
            keras.layers.Dense(self.n_samples),
        ])


    def _build_finder_network(self):
        return keras.Sequential([
            keras.layers.Input((self.scan_range,)),
            keras.layers.Dense(2 * self.n_samples, activation='relu'),
            keras.layers.Dense(2 * self.n_samples, activation='relu'),
            keras.layers.Dense(2 * self.n_samples, activation='relu'),
            keras.layers.Dense(self.d_sources),
        ])




    def _get_spectrum_constants(self) -> tuple:
        """
        G is unique per run, so G remains the same for a whole batch.
        B is not dependent on tau.  Therefore, it also remains stable for the whole batch.
        The same holds for the computation of the tau grid.

        RETURNS
        -------
        spectrum_constants : tuple

            Tau_grid
            G
            B
        """

        if self._cached_spectrum_constants is not None:
            return self._cached_spectrum_constants

        dt       = 1.0 / self.fs
        w        = tf.signal.fftshift(tf.cast(np.fft.fftfreq(self.n_samples, dt), dtype=tf.float32)) * (2.0 * np.pi)
        s        = tf.convert_to_tensor(self.signal_provider.generate(n_samples=self.n_samples), dtype=tf.complex64,)
        S        = tf.signal.fftshift(tf.signal.fft(s))
        dS       = gradient(S, tf.cast(w, dtype=tf.complex64))

        tau_grid = tf.linspace(
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(self.T, dtype=tf.float32),
            self.scan_range,
        )

        v_tau = tf.math.exp(
            tf.complex(
                tf.zeros((self.scan_range, self.n_samples), dtype=tf.float32),
                -tau_grid[:, None] * w[None, :],
            )
        )
        G = tf.stack(
            [
                v_tau * S[None, :],
                -v_tau * dS[None, :],
            ],
            axis=-1,
        )

        b11 = tf.reduce_sum(tf.math.conj(S) * S)
        b12 = -tf.reduce_sum(tf.math.conj(S) * dS)
        b21 = -tf.reduce_sum(tf.math.conj(dS) * S)
        b22 = tf.reduce_sum(tf.math.conj(dS) * dS)
        B   = tf.math.real(
            tf.stack(
                [
                    tf.stack([b11, b12]),
                    tf.stack([b21, b22]),
                ]
            )
        )

        self._cached_spectrum_constants = (tau_grid, G, B)

        return self._cached_spectrum_constants


    def compute_estimated_rcov(self, r_batched):
        """
        Consumes the whole batch and returns a surrogate
        representative of the covariance matrix.  The contract is that
        r_batched is of shape (batch, 2N, M).  Internelly the output is
        to be enforced to match the hermitian structure so that the EVD
        remains stable and meaningful.

        PARAMETERS
        ----------

        r_batched
            a tensor that represents a batch of complex signals

        """
        real         = tf.math.real(r_batched)
        imag         = tf.math.imag(r_batched)
        surrogate    = self.surrogate_network(tf.concat([real, imag], axis=1))
        batch_size = surrogate.shape[0]

        # make it complex (NxN)
        N     = surrogate.shape[-1]
        split = surrogate.shape[-1]
        real  = surrogate[:, split :, :]
        imag  = surrogate[:, : split, :]

        # enforce hermitian symmetry: R = R^H
        cov = tf.complex(real, imag)
        cov = tf.matmul(cov, cov, adjoint_b=True)
        tf.vectorized_map(lambda c: c + tf.eye(N, dtype=tf.complex64) * self.eps, cov)
        return Rxx(
            cov_batch=cov,
            n_samples_batch=tf.constant([0 for _ in tf.range(batch_size)]),
            m_sensors_batch=tf.constant([0 for _ in tf.range(batch_size)]),
        )

    
    def compute_noise_subspace(self, eigsv_batched, eigs_batched, k_est_batched):
        """
        PARAMETERS
        ----------
        eigsv_batched
            batched eigenvectors

        eigs_batched
            batched eigenvalues

        k_est_batched
            batched number of sources

        RETURNS
        -------
            noise_subspace
        """
        real      = tf.math.real(eigs_batched)
        imag      = tf.math.imag(eigs_batched) # is aloways zero because of hermitian structure of the r cov
        q_batched = self.selector_network(tf.keras.ops.append(real, imag, axis=1))
        # expand dims (batch_size, n_samples) -> (batch_size, n_samples, 1) for broadcasting
        # goal is eventually to apply: Q_diag @ eigsv_batched -> gaining a multiplication
        # of a diagonal matrix here broadcasting q_batches with the batched eigenvectors
        # (batch size, n samples, 1) -> (batch size, n samples, n samples)
        q_batched  = tf.expand_dims(q_batched, axis=-1)
        Un_batched = tf.complex(
            q_batched * tf.math.real(eigsv_batched),
            q_batched * tf.math.imag(eigsv_batched),
        )
        return Un_batched


    def compute_pseudo_inverse_spectrum(self, Un_batched):
        """
        Calculates the MUSIC spectrum according to P = 1 / (a^H En En^H a)

        PARAMETERS
        ----------
        Un
            The estimated noise space vectors

        RETURNS
        -------
        DelayDopplerEstimate

            The estimated delays with the corresponding doppler and the spectrum (see DelayDopplerEstimate definition)
            (batch size, scan range)
        """
        tau_grid, G, B = self._get_spectrum_constants()
        self.cost_function_batched, self.omega_grid_batched = self._compute_spectrum_graph(
            Un_batched,
            G,
            B,
        )

        # (batch size, scan range)
        self.tau_grid_batched = tf.broadcast_to(
            tau_grid[None, :],
            [tf.shape(Un_batched)[0], self.scan_range],
        )

        return self.cost_function_batched


    @tf.function(reduce_retracing=True)
    def _compute_spectrum_graph(self, Un_batched, G, B) -> tuple:
        """
        Compute G(tau)^H U_n for the whole batch and tau grid at once because of expensive training duration.

        PARAMETERS
        ----------
        Un_batched (batch, n samples, n samples)
        G (scan range, n samples, 2)
        B (2, 2)

        Returns
        -------
            inverse spectrum grid
            omega grid

        """

        GU = tf.einsum("tij,bik->btjk", tf.math.conj(G), Un_batched)
        A_mat = tf.math.real(
            tf.einsum("btjk,btck->btjc", GU, tf.math.conj(GU))
        )
        # -1 that the total size remains constant -> (batch size * scan rang, 2, 2)
        # because for to solve the general eigenvalue problem (this function can just compute one element at once)
        A_flat = tf.reshape(A_mat, [-1, 2, 2])

        def solve_generalized_eigh(a_mat):
            lambdas, gammas = eigh(a=a_mat, b=B, subset_by_index=[0, 1])
            gamma_min = gammas[:, 0]
            omega = tf.math.real(gamma_min[1] / gamma_min[0])
            return lambdas[0], tf.cast(omega, dtype=tf.float32)

        lambda_min_flat, omega_flat = tf.vectorized_map(
            solve_generalized_eigh,
            A_flat,
        )

        # recover the above-mentioned dimensions
        batch_size     = tf.shape(Un_batched)[0]
        spectrum_shape = [batch_size, self.scan_range]
        return (
            tf.reshape(lambda_min_flat, spectrum_shape),
            tf.reshape(omega_flat     , spectrum_shape),
        )


    def identify_signal_sources(self, spectrum_batched : tf.Tensor, k_est_batched : tf.Tensor):
        """
        since the pipeline is compiled / trained for a fixed number of signal sources it states true that all estimated ks holds the same value henceforth k_1 = k_2 = ... = k_n

        RETURN
        ------
            batched indicies of the tensor given from the peak finder network
        """
        # these are all real values
        delays_batched = self.finder_network(spectrum_batched)
        return delays_batched


    def compute_parameter_vector(self, delays_batched : tf.Tensor):
        """
        just a pass through
        """
        return delays_batched


class ClassicMusic(BaseMusicRecipe):


    def __init__(self, scan_range : int = 360, *args, **kwargs):
        """

        Parameters
        ----------
        scan_range
            steps that are going to go through the hypothesis space
        """
        super().__init__(*args, **kwargs)

        self.scan_range = scan_range

    
    def compute_pseudo_inverse_spectrum(self, Un_batched):
        spectrum_batched = []
        self._scan_range = []

        for Un in Un_batched:
            # hypothesis: if angles are pointing to sources
            scan_range = np.linspace(-np.pi / 2, np.pi / 2, self.scan_range)
            a = self.steering_provider.get_steering(scan_range)

            # pseudo spectrum: 1 / (a^H(theta) * Un * Un^H * a(theta))
            projection = tf.matmul(Un, a, adjoint_a=True)
            spectrum = 1 / tf.keras.ops.sum(np.abs(projection)**2, axis=0)
            # cache scan grid for peak-to-angle mapping
            self._scan_range.append(scan_range)
            spectrum_batched.append(spectrum)

        return tf.stack(spectrum_batched)

    
    def identify_signal_sources(self, spectrum_batched, k_est_batched):
        top_peaks_batched = []

        for spectrum, k_est in zip(spectrum_batched, k_est_batched):
            if k_est <= 0:
                top_peaks_batched.append(np.array([]))

            # local maxima indices
            gt_left_neighbor  = lambda s: s[1:-1] > s[:-2]
            gt_right_neighbor = lambda s: s[1:-1] > s[2:]
            peaks_idx         = tf.keras.ops.where(gt_left_neighbor(spectrum) & gt_right_neighbor(spectrum))[0] # array of indices
            filtered_spectrum = tf.gather(spectrum, peaks_idx)
            sort_fspec_idx    = tf.keras.ops.argsort(filtered_spectrum)[::-1] # descending order
            top_peaks_idx     = tf.gather(peaks_idx, sort_fspec_idx)[:k_est]
            top_peaks_batched.append(top_peaks_idx)

        return tf.stack(top_peaks_batched)

    
    def compute_parameter_vector(self, peaks_batched):
        angles_batched = []

        for peaks, scan_range in zip(peaks_batched, self._scan_range):
            if peaks is None or len(peaks) == 0:
                return np.array([])
            # peaks are indices into scan grid to retrieve the angeles
            angles = tf.gather(scan_range, peaks)
            angles_batched.append(angles)

        return tf.stack(angles_batched)


class RootMusic(BaseMusicRecipe):

    
    def __init__(self, eps_roots: float=1e-5, *args, **kwargs):
        """

        :param eps_roots: fuzziness parameter for estimating the inner roots
        """
        super().__init__(*args, **kwargs)
        self.eps_roots = eps_roots

    
    def compute_pseudo_inverse_spectrum(self, Un_batched):
        roots_batched = []

        for Un in Un_batched:
            F              = tf.matmul(Un ,Un, adjoint_b=True)
            M              = F.shape[0]
            diag_coeffs    = [tf.keras.ops.sum(tf.keras.ops.diag(F, l)) for l in range(-(M-1), M)]
            coeffs_ordered = diag_coeffs[::-1] # sort the polys from z^n, z^n-1, ..., z^0
            roots          = find_roots(coeffs_ordered)
            roots_batched.append(roots)

        return tf.stack(roots_batched)


    def identify_signal_sources(self, spectrum_batched, k_est_batched):
        found_roots_batched = []

        def empty():
            return spectrum[:0]

        for spectrum, k_est in zip(spectrum_batched, k_est_batched):
            sort_idx             = tf.keras.ops.argsort(tf.abs(tf.abs(spectrum) - 1))
            roots_sort_ascending = tf.gather(spectrum, sort_idx)
            mask                 = (tf.abs(roots_sort_ascending) - 1) < self.eps_roots
            roots                = roots_sort_ascending[mask][:k_est] # in eager mode False; no roots less than the unit circle can be found!!

            found_roots = tf.cond(
                k_est <= 0,
                empty,
                lambda: roots)

            found_roots_batched.append(found_roots)

        return tf.stack(found_roots_batched)


    def compute_parameter_vector(self, peaks_batched):
        thetas_batched = []

        for peaks in peaks_batched:
            root_angles = tf.keras.ops.angle(peaks)
            thetas_est   = tf.keras.ops.arcsin(root_angles / np.pi)
            thetas_batched.append(thetas_est)

        return tf.stack(thetas_batched)

    
class RMSPELoss(DAMusicLoss):
    """
    Root Means Square Phase Error
    """


    def compute_error(self, ground_truth, predictions):
        doa       = ground_truth
        perm_pred = predictions

        error = (((doa - perm_pred) + (np.pi / 2)) % np.pi) - np.pi / 2

        return error


class DeepRootMusic(keras.Model, RootMusic):


    def __init__(self, tau: int=8, activation_value: float=0.3, training: bool=False, model_load_name: str='model.h5', *args, **kwargs):
        """
        PARAMETERS
        ----------
        tau : int
            aims as a parameter for the empirical autocorrelation matrix from 0 .. tau_max
        """
        super().__init__()
        RootMusic.__init__(self, *args, **kwargs)

        self.n_samples = None
        self.training  = training
        self.tau       = tau
        m_antennas     = self.steering_provider.m_antennas
        self.model     = keras.Sequential([
            # Input block #0
            keras.Input(shape=(2*m_antennas, m_antennas, tau)),
            # CNN block #1
            keras.layers.Conv2D(16, kernel_size=2, name='conv1'),
            keras.layers.LeakyReLU(activation_value),
            # CNN block #2
            keras.layers.Conv2D(32, kernel_size=2, name='conv2'),
            keras.layers.LeakyReLU(activation_value),
            # CNN block #3
            keras.layers.Conv2D(64, kernel_size=2, name='conv3'),
            keras.layers.LeakyReLU(activation_value),
            # DCNN block #1
            keras.layers.Conv2DTranspose(32, kernel_size=2, name='deconv1'),
            keras.layers.LeakyReLU(activation_value),
            # dcnn block #2
            keras.layers.Conv2DTranspose(16, kernel_size=2, name='deconv2'),
            keras.layers.LeakyReLU(activation_value),
            # dcnn block #3
            keras.layers.Dropout(0.2),
            keras.layers.Conv2DTranspose(1, kernel_size=2, name='deconv3'),
        ])
        if not self.training:
            self.model = keras.models.load_model(model_load_name)

    
    def call(self, inputs):
        r_sensed = inputs
        doa_pred = self.estimate(r_sensed)
        return doa_pred

    
    def compute_estimated_rcov(self, r_batched):
        Rxx_tau      = self.compute_empirical_autocorrelation(r_batched)
        Rxx_predict  = self.predict(Rxx_tau)
        Kxx          = self.compose_real_imag(Rxx_predict)
        Rzz          = self.compute_hermit_psd(Kxx, eps=1)
        n_sensors    = Rzz.shape[2]
        n_batch_size = Rzz.shape[0]
        return Rxx(
            cov_batch=Rzz,
            m_sensors_batch=tf.constant([n_sensors for _ in range(n_batch_size)]),
            n_samples_batch=tf.constant([self.n_samples for _ in range(n_batch_size)]),
        )

    
    # private functions
    def compute_empirical_autocorrelation(self, r_batched: tf.Tensor) -> tf.Tensor:
        """
        computes a set of empirical autocovariance matrices with tau = 0, ..., tau_max

        the matrix will be decomposed into real and imaginary parts, hence the shape of the output is (tau_max, 2*M, M)

        :param r: Size: (batch_size, n_antennas, n_samples)
        :return: a set of autocovariance matrices R(tau); shape: (tau_max, 2*M, M)
        """
        autocov_batched = []

        for r in r_batched:
            autocov_set = []
            for i in range(self.tau):
                Rxx, n  = sample_autocov(r, tau=i)
                imag    = tf.math.imag(Rxx)
                real    = tf.math.real(Rxx)
                autocov_set.append(tf.keras.ops.append(imag, real, axis=0))

            self.n_samples = n
            autocov_batched.append(tf.stack(autocov_set))

        return tf.stack(autocov_batched)


    def predict(self, Rxx_tau: tf.Tensor) -> tf.Tensor:
        """

        :param Rxx_tau: shape: (batch_size, tau, 2*M, M)
        :return: Eager Tensor; shape: (batch_size, 2*M, M)
        """
        # shape: (batch_size, tau, 2*M, M) -> NCHW format
        batch_size, _, M2, M = Rxx_tau.shape
        # tf.transpose: Permutes the dimensions according to the value of perm.
        Rxx_NHWC             = tf.transpose(Rxx_tau, perm=(0, 2, 3, 1)) # shape: (batch_size, 2*M, M, tau) -> NHWC format
        Rxx_predicted        = self.model(Rxx_NHWC, training=self.training)
        # flatten the output
        Rxx_complex_split    = tf.reshape(Rxx_predicted, shape=(batch_size, M2, M))
        return Rxx_complex_split

    
    def compose_real_imag(self, Rxx: tf.Tensor) -> tf.Tensor:
        """
        Combine the real and imaginary parts of the autocovariance matrices into a single matrix.  Per definition,
        the first M elements are the imaginary part and the last M elements are the real part.

        :param Rxx: shape: (batch_size, 2*M, M)
        :return: shape: (batch_size, M, M)
        """
        M        = Rxx.shape[-1]
        Rxx_real = Rxx[:, M :, :]
        Rxx_imag = Rxx[:, : M, :]
        Rxx_tag  = tf.complex(Rxx_real, Rxx_imag)
        return Rxx_tag

    
    def compute_hermit_psd(self, Kxx: tf.Tensor, eps: int=1):
        """
        Ensures to compute a PSD (Positive Semi-Definite) matrix with Hermitian symmetry,
        i.e. Rzz = Rzz^H, by adding eps to the diagonal of Kxx

        :param Kxx: Size: (batch_size, M, M)
        :param eps: noise variance
        :return: Size: (batch_size, M, M)
        """
        batch_size, _, M = Kxx.shape
        Rzz              = []
        for batch in range(batch_size):
            Kxx_Hermit = tf.matmul(Kxx[batch], Kxx[batch], adjoint_b=True)
            eps_add    = tf.eye(M, dtype=tf.complex64) * eps
            Rzz.append(Kxx_Hermit + eps_add)

        return tf.stack(Rzz)
