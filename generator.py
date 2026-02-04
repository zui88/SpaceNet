import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod


class SignalGenerator(ABC):

    
    def __init__(self, n_samples : int = None, T : float = 1, fs : float = 50):
        """
        PARAMETER
        ---------
        n_samples : int
            when n_samples is explicitly specified fs and T are ignored when the signal is produced

        T : float
            pulse lenght

        fs : float
            frequency to which the signal is sampled
        """
        self.T          = T
        self.fs         = fs
        self.n_samples_ = n_samples
        self.t          = np.arange(0, self.T, 1/self.fs)


    @property
    def n_samples(self):
        if self.n_samples_ is None:
            n =self.T * self.fs
        else:
            n = self.n_samples_
        return n

    
    @abstractmethod
    def function(self, x):
        pass

    
    def generate(self, T_pad : float = None, n_samples : int = None, array : int = 1, axis : int = 0):
        """
        PARAMETER
        ---------
        T_pad : float
            when a pad value is given the generated signal is filled up with zeros to this value

        array : int
            How many arrays the function returns.  When array > 1 the function returns array clones of the signal; f.e. (SamplesxArray), where Samples = T * fs
        """
        s = self.function(self.t)
        s = self.conditional_pad(s, T_pad, n_samples)

        if array > 1:
            s = np.vstack(np.array([self.function(self.t) for _ in range(array)]))

        if axis == 1:
            s = s.T

        return s

    
    def conditional_pad(self, signal, T_pad : float | None, n_samples : int | None):
        s = signal

        if T_pad is not None:
            length = T_pad*self.fs
            s = tf.keras.ops.pad(s, (0, length))[:length]

        if n_samples is not None:
            length = n_samples
            s = tf.keras.ops.pad(s, (0, length))[:length]

        return s


    def gradient(self, T_pad : float = None, n_samples : int = None):
        x = tf.Variable(self.t)
        x = tf.cast(x, tf.complex64)
        with tf.GradientTape() as tape:
            tape.watch(x)
            y = self.function(x)

        first_derivate = tape.gradient(y, x)
        first_derivate = self.conditional_pad(first_derivate, T_pad, n_samples)

        return first_derivate


class ChirpSignal(SignalGenerator):

    
    def __init__(self, alpha : float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
    
    
    def function(self, x):
        j         = tf.complex(0.0, 1.0)
        alpha     = tf.cast(self.alpha, tf.complex64)
        x_complex = tf.cast(x, tf.complex64)
        s         = tf.math.exp(j * alpha * x_complex**2)
        return s
    

class SincTSignal(SignalGenerator):
    
    
    def function(self, x):
        s = np.sinc(x/self.T) / (1 - (x/self.T)**2)
        return s


class RandomSignal(SignalGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng()


    def function(self, x):
        y = self.rng.standard_normal((self.n_samples,))
        x = self.rng.standard_normal((self.n_samples,))
        s = 1/np.sqrt(2) * (x + 1j*y)
        return s


class ArrayGeometry(ABC):
    """
    It defines an array that consist of number of array sensors and incident signals sources.

    f.e. (Arrays x Signals)
    """

    def __init__(self, antennas = 6):
        self.antennas = antennas


    @abstractmethod
    def get_steering(self, thetas : np.ndarray, axis: int = 0) -> np.ndarray:
        pass


    @property
    def m_antennas(self):
        m = self.antennas
        return m

    
class ULAArray(ArrayGeometry):


    def __init__(self, d_lambda : float = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_lambda = d_lambda


    def get_steering(self, thetas : np.ndarray, axis: int = 0) -> np.ndarray:
        """
        steering/mode vectors for "standard ULA"

        A.shape() = (antennas x sources) for axis == 0
        A.shape() = (sources x antennas) for axis == 1
        """
        if len(thetas) < 1:
            raise RuntimeError("thetas must be at least has the size of 1")
        
        antennas_idx = np.arange(self.m_antennas)[:, None]
        A            = np.exp(1j * np.pi * self.d_lambda * np.sin(thetas) * antennas_idx)  # broadcast sin with numbers of antennas

        if axis == 1:
            A = A.T

        return A


class RandomArray(ArrayGeometry):
    """
    It defines an array that consist of number of array sensors and incident signals sources.

    f.e. (Arrays x Signals)
    """
    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng()

    def get_steering(self, thetas : np.ndarray, axis : int = 0):
        if len(thetas) < 1:
            raise RuntimeError("thetas must be at least has the size of 1")
        
        M = self.m_antennas
        D = len(thetas)

        x = self.rng.standard_normal((D, M))
        y = self.rng.standard_normal((D, M))
        s = 1/np.sqrt(2) * (x + 1j*y)
        return s


class SignalSynthesizer(ABC):


    def __init__(self, snr_db : int = 30):
        self.sigma2 = np.power(10, -snr_db/10)
        self.rng    = np.random.default_rng()


    @abstractmethod
    def generate(self, *args, **kwargs):
        pass


    def get_rand(self, N : int, M : int = None):
        return self.rng.standard_normal(N) if M is None else self.rng.standard_normal((N, M))


    def get_noise_matrix(self, dimension : int | tuple):
        """
        dimension : int | tuple
            either scalar or tuple of two that defines the matrix or vector

            (tuple(0) x tuple(M))
            (NxM)
        """
        w = None
        
        if isinstance(dimension, tuple):
            N = dimension[0]
            M = dimension[1]
            w = np.sqrt(self.sigma2/2) * (self.get_rand(N, M) + 1j * self.get_rand(N, M)) # (NxM)

        if isinstance(dimension, int):
            N = dimension
            w = np.sqrt(self.sigma2/2) * (self.get_rand(N) + 1j * self.get_rand(N)) # (N)

        if w is None:
            raise RuntimeError("wront dimension set!")
        
        return w


class DOASignalSynthesizer(SignalSynthesizer):

    def __init__(self, array_geometry, signal_generator, *args, **kwargs):
        """
        Parameters
        ----------
        
        """
        super().__init__(*args, **kwargs)
        self.array_geometry   = array_geometry
        self.signal_generator = signal_generator

    def generate(self, thetas : np.ndarray):
        """
        generate the sensed signal

        Parameters
        ----------

        thetas : list
            direction of arrival angles
        """
        D = len(thetas)
        A = self.array_geometry.get_steering(thetas)
        Q = self.signal_generator.generate(array=D)
        w = self.get_noise_matrix((A.shape[0], Q.shape[1]))
        R = A @ Q + w
        return R


class ObservationContext():

    def __init__(self, T : float = 8):
        self.T = T

    @property
    def window_length(self):
        return self.T


class DelayDopplerSignalSynthesizer(SignalSynthesizer):

    def __init__(self, array_geometry, signal_generator, observ_ctx = ObservationContext(), *args, **kwargs):
        """
        Parameters
        ---------- 
        T : ObservationContext
            Holds the duration of the entire observation window
        """
        super().__init__(*args, **kwargs)
        self.array_geometry   = array_geometry
        self.signal_generator = signal_generator
        self.T                = observ_ctx.window_length
        self.fs               = signal_generator.fs

    @property
    def n_samples(self):
        n = self.T * self.fs
        return n

    def generate(self, taus : list = [0.5, 3], omegas : list = [0.01, -0.03], thetas : list = [0, 30]):
        M = self.array_geometry.m_antennas
        N = self.n_samples

        Q = self.generate_Q(np.array(taus), np.array(omegas))
        A = self.array_geometry.get_steering(np.array(thetas))
        w = self.get_noise_matrix((N, M))
        R = Q @ A + w           # (NxM)
        return R

    def generate_Q(self, taus, omegas):
        """
        RETURN
        ------
            signal matrix with shape(NxD)
        """
        if not len(taus) == len(omegas):
            raise RuntimeError("taus and omega must be the same length!")
        
        t            = np.arange(0, self.T, 1/self.fs)
        s            = self.signal_generator.generate(self.T)
        omegas_shift = np.exp(1j*omegas[None, :]*t[:, None]) # (NxD)
        n_omegas     = omegas_shift.shape[1]
        signals      = []
        for k in range(n_omegas):
            phase_shifted_signal = s * omegas_shift[:,k]
            pad                  = int(taus[k] * self.fs)
            time_shifted_signal  = np.pad(phase_shifted_signal, (pad, 0))
            transformed_signal   = time_shifted_signal[0:self.n_samples]
            signals.append(transformed_signal)

        Q = np.stack(signals, axis=1)
        return Q
        
