from SpaceNet.Plugins.plugin import Link, Plugin, Ports

from dataclasses import dataclass

import tensorflow as tf


#todo: remove and use Ports instead
@dataclass(frozen=True, slots=True)
class Eigs:
    eigs_batch: tf.Tensor
    n_samples_batch: tf.Tensor
    m_sensors_batch: tf.Tensor


class EVD(Plugin):


    def __init__(self):

        self.input_ports: Ports  = {"r_cov": Link()}
        self.output_ports: Ports = {
            "eigs_v": Link(),
            "eigs": Link(),
        }


    def execute(self) -> None:
        r_cov = self.input_ports["r_cov"].value

        eigs, eigs_v = tf.linalg.eigh(r_cov.cov_batch)
        idx          = tf.argsort(tf.abs(eigs), axis=1, direction="DESCENDING")
        eigs         = tf.gather(eigs, idx, batch_dims=1)
        eigs_v       = tf.gather(eigs_v, idx, axis=2, batch_dims=1)

        self.output_ports["eigs_v"].value = tf.cast(eigs_v, tf.complex64)
        self.output_ports["eigs"].value   = Eigs(
            tf.cast(eigs, tf.complex64),
            r_cov.n_samples_batch,
            r_cov.m_sensors_batch,
        )
