from SpaceNet.Plugins.plugin import Link, Plugin, Ports

from dataclasses import dataclass

import tensorflow as tf
import numpy as np


#todo: remove Rxx, using ports instead for dataflow
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
        r_batched                = tf.convert_to_tensor(self.input_ports["r_sensed"].value)
        _, m_antennas, n_samples = r_batched.shape

        cov_batched     = []
        n_samples_batch = []
        n_sensors_batch = []

        for r in r_batched:
            cov = tf.matmul(r, r, adjoint_b=True) / n_samples
            cov_batched.append(cov)
            n_samples_batch.append(n_samples)
            n_sensors_batch.append(m_antennas)

        self.output_ports["r_cov"].value = Rxx(
            tf.stack(cov_batched),
            tf.convert_to_tensor(n_samples_batch),
            tf.convert_to_tensor(n_sensors_batch),
        )


class EstimateRcovFFT(Plugin):


    def __init__(self):
        self.input_ports: Ports  = {"r_sensed": Link()}
        self.output_ports: Ports = {"r_cov": Link()}


    def execute(self) -> None:
        r: tf.Tensor = self.input_ports["r_sensed"].value

        m_antennas = r.shape[-1]
        n_samples  = r.shape[-2]
        R          = np.fft.fftshift(
            np.fft.fft(r, axis=1),
            axes=1,
        )  # (batch,N,M)

        R_cov = R @ R.conj().swapaxes(-1, -2) / m_antennas # (N,N)

        self.output_ports["r_cov"].value = Rxx(
            tf.stack(R_cov),
            tf.convert_to_tensor([n_samples for _ in range(r.shape[0])]),
            tf.convert_to_tensor([m_antennas for _ in range(r.shape[0])]),
        )
