from dataclasses import dataclass

import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


@dataclass(frozen=True)
class Rxx:
    cov_batch: tf.Tensor
    n_samples_batch: tf.Tensor
    m_sensors_batch: tf.Tensor


class EstimateRcov(Plugin):


    def __init__(self):
        self.input_ports: Ports  = {"r_sensed": Link()}
        self.output_ports: Ports = {"r_cov": Link()}


    def execute(self) -> None:
        r_batched               = tf.convert_to_tensor(self.input_ports["r_sensed"].value)
        _, n_sensors, n_samples = r_batched.shape

        cov_batched     = []
        n_samples_batch = []
        n_sensors_batch = []

        for r in r_batched:
            cov = tf.matmul(r, r, adjoint_b=True) / n_samples
            cov_batched.append(cov)
            n_samples_batch.append(n_samples)
            n_sensors_batch.append(n_sensors)

        self.output_ports["r_cov"].value = Rxx(
            tf.stack(cov_batched),
            tf.convert_to_tensor(n_samples_batch),
            tf.convert_to_tensor(n_sensors_batch),
        )
