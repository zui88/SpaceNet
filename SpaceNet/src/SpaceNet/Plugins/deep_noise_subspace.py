from SpaceNet.Plugins.plugin import Link, Plugin, Ports

import tensorflow as tf
from tensorflow import keras


class DeepNoiseSubspace(Plugin):
    def __init__(
        self,
        selector_network: keras.models.Model,
    ):
        self.selector_network = selector_network
        self.input_ports: Ports = {
            "eigs_v": Link(),
            "eigs": Link(),
        }
        self.output_ports: Ports = {"Un": Link()}

    def execute(self) -> None:
        eigs_v = self.input_ports["eigs_v"].value
        eigs = self.input_ports["eigs"].value.eigs_batch

        real = tf.math.real(eigs)
        imag = tf.math.imag(eigs)
        roh = self.selector_network(tf.keras.ops.append(real, imag, axis=1))
        roh_diag = tf.linalg.diag(roh)
        Un = tf.complex(
            roh_diag * tf.math.real(eigs_v),
            roh_diag * tf.math.imag(eigs_v),
        )

        self.output_ports["Un"].value = Un
