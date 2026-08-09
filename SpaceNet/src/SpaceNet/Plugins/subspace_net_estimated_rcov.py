"""
This plugin implements one component of the approach from Guy Revach of his proposed
framework SubspaceNet.
"""

from SpaceNet.Plugins.plugin import Link, Plugin, Ports

from dataclasses import dataclass

import tensorflow as tf
import keras


@dataclass(frozen=True, slots=True)
class Rxx:
    cov_batch: tf.Tensor
    n_samples_batch: tf.Tensor
    m_sensors_batch: tf.Tensor


class SubspaceNetEstimateRcov(Plugin):
    def __init__(
        self,
        rcov_network: keras.models.Model,
        tau: int = 8,
        eps: float = 1,
    ) -> None:
        self.input_ports: Ports = {"r_sensed": Link()}
        self.output_ports: Ports = {"surrogate_rcov": Link()}

        self.rcov_network = rcov_network
        self.tau = tau
        self.eps = eps

    def execute(self) -> None:
        r_batched = tf.convert_to_tensor(self.input_ports["r_sensed"].value)

        Rxx_tau = self._compute_empirical_autocorrelation(r_batched)
        Rxx_predict = self._predict(Rxx_tau)
        Kxx = self._compose_real_imag(Rxx_predict)
        Rzz = self._compute_hermit_psd(Kxx, eps=self.eps)
        n_sensors = Rzz.shape[2]
        batch_size, m_antennas, n_samples = Rzz.shape

        self.output_ports["surrogate_rcov"].value = Rxx(
            cov_batch=Rzz,
            # pyrefly: ignore [bad-argument-type]
            m_sensors_batch=tf.constant([n_sensors for _ in range(batch_size)]),
            # pyrefly: ignore [bad-argument-type]
            n_samples_batch=tf.constant([n_samples for _ in range(batch_size)]),
        )

    @staticmethod
    def _sample_autocov(X: tf.Tensor, tau: int) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Calculates the sample autocovariance matrix R(tau) for a given time lag tau.

            Guy Revach; DEEP ROOT MUSIC ALGORITHM FOR DATA-DRIVEN DOA ESTIMATION; P. 3
            Rx = 1/(T - tau) sum(X(t) X(t+tau)^H) with t = 0, ..., T-tau

        :param X:   Complex-valued array of shape (M,T) containing M antenna signals over T time samples
        :param tau: Time lag value to calculate autocovariance
        :return:    Matrix of shape (M,M) containing the autocovariance matrix; number of samples T
        """
        M, T = X.shape
        # pyrefly: ignore [unsupported-operation]
        X1 = X[:, : T - tau]  # x(t)
        X2 = X[:, tau:T]  # x(t+tau)
        R = tf.matmul(X1, X2, adjoint_b=True)
        # pyrefly: ignore [bad-return]
        return R, T

    def _compute_empirical_autocorrelation(self, r_batched: tf.Tensor) -> tf.Tensor:
        """
        computes a set of empirical autocovariance matrices with tau = 0, ..., tau_max

        the matrix will be decomposed into real and imaginary parts, hence the shape of the output is (tau_max, 2*M, M)

        :param r: Size: (batch_size, n_antennas, n_samples)
        :return: a set of autocovariance matrices R(tau); shape: (tau_max, 2*M, M)
        """

        def compute_tau(r, tau):
            Rxx, _ = self._sample_autocov(r, tau=tau)

            imag = tf.math.imag(Rxx)
            real = tf.math.real(Rxx)

            return tf.concat([imag, real], axis=0)

        def compute_sample(r):
            taus = tf.range(self.tau)

            return tf.map_fn(
                lambda tau: compute_tau(r, tau),
                taus,
                fn_output_signature=tf.TensorSpec(
                    shape=(None, None),
                    dtype=r.dtype.real_dtype,
                ),
            )

        return tf.map_fn(
            compute_sample,
            r_batched,
            fn_output_signature=tf.TensorSpec(
                shape=(None, None, None),
                dtype=r_batched.dtype.real_dtype,
            ),
        )

    def _compose_real_imag(self, Rxx):
        """
        Combine the real and imaginary parts of the autocovariance matrices into a single matrix.  Per definition,
        the first M elements are the imaginary part and the last M elements are the real part.

        :param Rxx: shape: (batch_size, 2*M, M)
        :return: shape: (batch_size, M, M)
        """
        M = Rxx.shape[-1]
        Rxx_real = Rxx[:, M:, :]
        Rxx_imag = Rxx[:, :M, :]
        Rxx_tag = tf.complex(Rxx_real, Rxx_imag)
        return Rxx_tag

    def _compute_hermit_psd(self, Kxx, eps: float) -> tf.Tensor:
        """
        Ensures to compute a PSD (Positive Semi-Definite) matrix with Hermitian symmetry,
        i.e. Rzz = Rzz^H, by adding eps to the diagonal of Kxx

        :param Kxx: Size: (batch_size, M, M)
        :param eps: noise variance
        :return: Size: (batch_size, M, M)
        """
        batch_size, _, M = Kxx.shape
        Rzz = []
        for batch in range(batch_size):
            Kxx_Hermit = tf.matmul(Kxx[batch], Kxx[batch], adjoint_b=True)
            eps_add = tf.eye(M, dtype=tf.complex64) * eps
            Rzz.append(Kxx_Hermit + eps_add)

        return tf.stack(Rzz)

    def _predict(self, Rxx_tau):
        """

        :param Rxx_tau: shape: (batch_size, tau, 2*M, M)
        :return: Eager Tensor; shape: (batch_size, 2*M, M)
        """
        # shape: (batch_size, tau, 2*M, M) -> NCHW format
        batch_size, _, M2, M = Rxx_tau.shape
        # tf.transpose: Permutes the dimensions according to the value of perm.
        Rxx_NHWC = tf.transpose(
            Rxx_tau, perm=(0, 2, 3, 1)
        )  # shape: (batch_size, 2*M, M, tau) -> NHWC format
        Rxx_predicted = self.rcov_network(Rxx_NHWC)
        # flatten the output
        Rxx_complex_split = tf.reshape(Rxx_predicted, shape=(batch_size, M2, M))
        return Rxx_complex_split
