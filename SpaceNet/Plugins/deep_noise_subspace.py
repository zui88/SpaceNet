from SpaceNet.Plugins.plugin import Link, Plugin, Ports

import tensorflow as tf
from tensorflow import keras


class DeepNoiseSubspace(Plugin):


    def __init__(self, selector_network: keras.models.Model, ):
        self.selector_network = selector_network
        self.input_ports: Ports = {
            "eigsv": Link(),
            "eigs": Link(),
        }
        self.output_ports: Ports = {"Un": Link()}


    def execute(self) -> None:
        eigsv_batched = self.input_ports["eigsv"].value
        eigs = self.input_ports["eigs"].value.eigs_batch

        real = tf.math.real(eigs)
        imag = tf.math.imag(eigs)
        q_batched = self.selector_network(tf.keras.ops.append(real, imag, axis=1))
        q_batched = tf.expand_dims(q_batched, axis=-1)

        self.output_ports["Un"].value = tf.complex(
            q_batched * tf.math.real(eigsv_batched),
            q_batched * tf.math.imag(eigsv_batched),
        )
