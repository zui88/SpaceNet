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
        r_hat         = self.input_ports["rcov"].value
        eigsv_batched = []
        eigs_batched  = []

        for cov in r_hat.cov_batch:
            # if there is a guaranty that cov is hermitian then 'eigh' could be applied
            eigs, eigsv = tf.linalg.eigh(cov)
            idx         = tf.argsort(tf.abs(eigs), direction="DESCENDING")
            eigs_batched.append(tf.gather(eigs, idx))
            eigsv_batched.append(tf.gather(eigsv, idx, axis=1))

        self.output_ports["eigsv"].value = tf.stack(eigsv_batched)
        self.output_ports["eigs"].value  = Eigs(
            tf.stack(eigs_batched),
            r_hat.n_samples_batch,
            r_hat.m_sensors_batch,
        )
