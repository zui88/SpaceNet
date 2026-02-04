from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class SignalGenerator(ABC):

    
    def __init__(self, n_samples : int = None, T : float = 1, fs : int = 50):
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
            # because in general T_pad is a float
            length = int(T_pad * self.fs)
            s = tf.keras.ops.pad(s, (0, length))[:tf.constant(length)]

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


class ObservationContext:


    def __init__(self, T: float = 8):
        self.T = T


    @property
    def window_length(self):
        return self.T
