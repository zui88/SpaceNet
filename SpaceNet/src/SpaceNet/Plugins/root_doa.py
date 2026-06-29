import numpy as np
import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class ComputeRootDOA(Plugin):
    def __init__(self):
        self.input_ports: Ports = {"roots": Link()}
        self.output_ports: Ports = {"doa": Link()}

    def execute(self) -> None:
        roots_batched = self.input_ports["roots"].value
        thetas_batched = []

        for roots in roots_batched:
            root_angles = tf.keras.ops.angle(roots)
            thetas_batched.append(tf.keras.ops.arcsin(root_angles / np.pi))

        self.output_ports["doa"].value = tf.stack(thetas_batched)
