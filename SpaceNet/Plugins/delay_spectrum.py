from SpaceNet.Synthesizer.signal import ObservationContext, SignalGenerator
from SpaceNet.Plugins.plugin import Ports, Link, Plugin
from SpaceNet.Utils.tf_np_ports import gradient, eigh

from dataclasses import dataclass
import functools

import tensorflow as tf
import numpy as np


@dataclass(frozen=True)
class Spectrum:
    """
    for printing the spectrum
    """
    spectrum: tf.Tensor
    scan_range: tf.Tensor


class DelaySpectrum(Plugin):


    def __init__(self,
                 scan_range: int,
                 signal_provider: SignalGenerator,
                 observation_context: ObservationContext,
                 ):
        """

        Parameters
        ----------
        scan_range
            The grid where the spectrum will be plotted.
            The higher the range, the more precise the possible predictions are;
            f.e. the highest reasonable range is N samples from the sample space, but this leads to
            very high computational time.

        signal_provider
            the signal provider is responsible for generating the distinct signal and
            also holds the properties of the signal, f.e. the sample frequency 'fs'.

        n_samples
            number of samples, this is also called the 'sample space'

        observation_context
            The context where the observation will be recorded.
        """
        self.input_ports: Ports  = {"Un": Link()}
        self.output_ports: Ports = {
            "spectrum": Link(),
            "spectrum_obj": Link(),
            "taus_grid": Link(),
            "omegas_grid": Link(),
        }

        self.scan_range          = scan_range
        self.signal_provider     = signal_provider
        self.observation_context = observation_context

        self.tau_grid, self.G, self.B = self._get_constants()


    def execute(self):
        Un: tf.Tensor = self.input_ports["Un"].value

        cost_function, doppler_grid = self._compute_spectrum_graph(Un, self.G, self.B)
        tau_grid                    = tf.broadcast_to(
            self.tau_grid[None, :],
            [tf.shape(Un)[0], self.scan_range],
        )

        self.output_ports["spectrum"].value     = cost_function
        self.output_ports["spectrum_obj"].value = Spectrum(cost_function, tau_grid)
        self.output_ports["taus_grid"].value    = tau_grid
        self.output_ports["omegas_grid"].value  = doppler_grid


    @tf.function(reduce_retracing=True)
    def _compute_spectrum_graph(self, Un: tf.Tensor, G: tf.Tensor, B: tf.Tensor) -> tuple:
        """
        Compute G(tau)^H U_n for the whole batch and tau grid
        at once because of expensive training duration.

        PARAMETERS
        ----------
        Un_batched (batch, n samples, n samples)
        G (scan range, n samples, 2)
        B (2, 2)

        Returns
        -------
            inverse spectrum grid: relates to the estimated delays
            omega grid: relates to the estimated doppler shifts

        """

        GU    = tf.einsum("tij,bik->btjk", tf.math.conj(G), Un)
        A_mat = tf.math.real(tf.einsum("btjk,btck->btjc", GU, tf.math.conj(GU)))
        # -1 that the total size remains constant -> (batch size * scan rang, 2, 2)
        # because for to solve the general eigenvalue problem (this function can just compute one element at once)
        A_flat = tf.reshape(A_mat, [-1, 2, 2])

        def solve_generalized_eigh(a_mat):
            lambdas, gammas = eigh(a=a_mat, b=B, subset_by_index=[0, 1])
            gamma_min       = gammas[:, 0]
            omega           = tf.math.real(gamma_min[1] / gamma_min[0])
            return lambdas[0], tf.cast(omega, dtype=tf.float32)


        lambda_min_flat, omega_flat = tf.vectorized_map(
            solve_generalized_eigh,
            A_flat,
        )

        # recover the above-mentioned dimensions
        batch_size     = tf.shape(Un)[0]
        spectrum_shape = [batch_size, self.scan_range]
        return (
            tf.reshape(lambda_min_flat, spectrum_shape),
            tf.reshape(omega_flat     , spectrum_shape),
        )


    @functools.cache
    def _get_constants(self):
        """This function computes all data that stays constant during the whole inference step through the
        estimation pipeline, e.g. when one calls 'engine.estimate(.)'.

        The returned data is named to match the nomenclature in "Subspace-Based Estimation of Time Delays and Doppler Shifts".

        Returns
        -------
            tau_grid, G, B
        """
        n_samples = int(self.signal_provider.fs * self.observation_context.T) # samples from sample space

        dt = 1.0 / self.signal_provider.fs
        w  = tf.signal.fftshift(tf.cast(np.fft.fftfreq(n_samples, dt), dtype=tf.float32)) * (2.0 * np.pi)
        s  = tf.convert_to_tensor(self.signal_provider.generate(n_samples=n_samples), dtype=tf.complex64,)
        S  = tf.signal.fftshift(tf.signal.fft(s))
        dS = gradient(S, tf.cast(w, dtype=tf.complex64))

        tau_grid = tf.linspace(
            tf.constant(0.0, dtype=tf.float32),
            tf.constant(self.observation_context.T, dtype=tf.float32),
            self.scan_range,
        )

        v_tau = tf.math.exp(
            tf.complex(
                tf.zeros((self.scan_range, n_samples), dtype=tf.float32),
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

        return tau_grid, G, B
