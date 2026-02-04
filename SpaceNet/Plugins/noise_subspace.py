import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class NoiseSubspace(Plugin):


    def __init__(self):
        self.input_ports: Ports = {
            "eigsv": Link(),
            "k_est": Link(),
        }
        self.output_ports: Ports = {"Un": Link()}


    def execute(self) -> None:
        eigsv_batched = self.input_ports["eigsv"].value
        k_est_batched = self.input_ports["k_est"].value
        Un_batched    = []

        for eigsv, k_est in zip(eigsv_batched, k_est_batched):
            Un_batched.append(eigsv[:, int(k_est.numpy()):])

        self.output_ports["Un"].value = tf.stack(Un_batched)
