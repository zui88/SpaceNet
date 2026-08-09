import numpy as np
import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class ComputeRootDOA(Plugin):
    def __init__(self):
        self.input_ports: Ports = {"roots": Link()}
        self.output_ports: Ports = {"doa": Link()}

    def execute(self) -> None:
        roots_batched = self.input_ports["roots"].value

        thetas_batched = tf.math.asin(
            tf.math.angle(roots_batched)
            / tf.constant(
                np.pi,
                dtype=roots_batched.dtype.real_dtype,
            )
        )

        self.output_ports["doa"].value = thetas_batched
