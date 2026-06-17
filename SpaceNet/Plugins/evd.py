from SpaceNet.Plugins.plugin import Link, Plugin, Ports

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True, slots=True)
class Eigs:
    eigs_batch: tf.Tensor
    n_samples_batch: tf.Tensor
    m_sensors_batch: tf.Tensor


class EVD(Plugin):


    def __init__(self):

        self.input_ports: Ports  = {"rcov": Link()}
        self.output_ports: Ports = {
            "eigsv": Link(),
            "eigs": Link(),
        }


    def execute(self) -> None:
        r_hat = self.input_ports["rcov"].value

        eigs, eigsv = tf.linalg.eigh(r_hat.cov_batch)
        idx         = tf.argsort(tf.abs(eigs), axis=1, direction="DESCENDING")
        eigs        = tf.gather(eigs, idx, batch_dims=1)
        eigsv       = tf.gather(eigsv, idx, axis=2, batch_dims=1)

        self.output_ports["eigsv"].value = eigsv
        self.output_ports["eigs"].value  = Eigs(
            eigs,
            r_hat.n_samples_batch,
            r_hat.m_sensors_batch,
        )
