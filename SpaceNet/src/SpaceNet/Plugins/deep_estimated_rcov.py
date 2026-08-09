from SpaceNet.Plugins.plugin import Link, Plugin, Ports
from SpaceNet.Plugins.estimated_rcov import Rxx

from tensorflow import keras
import tensorflow as tf


class DeepEstimateRcov(Plugin):
    def __init__(self, surrogate_network: keras.models.Model, eps: float = 1.0):

        self.rcov_network = surrogate_network
        self.eps_rcov = eps
        self.input_ports: Ports = {"r_sensed": Link()}
        self.output_ports: Ports = {"surrogate_rcov": Link()}

    def execute(self) -> None:
        r_batched = tf.convert_to_tensor(self.input_ports["r_sensed"].value)
        real = tf.math.real(r_batched)
        imag = tf.math.imag(r_batched)
        X = tf.concat([real, imag], axis=1)
        X = tf.transpose(X, perm=[0, 2, 1])
        surrogate = self.rcov_network(X)

        batch_size = tf.shape(surrogate)[0]
        n_sensors = tf.shape(surrogate)[-1]
        split = n_sensors
        cov_real = surrogate[:, :split, :]
        cov_imag = surrogate[:, split:, :]

        # alternativ approach: oberes dreieck -> spiegeln auf unten
        # ################################################################
        # ensures that every matrix B is hermitian and positive
        # semidefinit even if B hasn't full rank though (I*eps)
        #
        # S = B x B^H + I * eps
        #
        # S is hermitian and positiv semidefinit
        B = tf.complex(cov_real, cov_imag)
        B = tf.matmul(B, B, adjoint_b=True)
        S = B + tf.eye(n_sensors, dtype=B.dtype) * self.eps_rcov

        self.output_ports["surrogate_rcov"].value = Rxx(
            cov_batch=S,
            n_samples_batch=tf.zeros((batch_size,), dtype=tf.int32),
            m_sensors_batch=tf.fill((batch_size,), n_sensors),
        )
