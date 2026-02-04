import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class RootSelector(Plugin):


    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.input_ports: Ports = {
            "roots": Link(),
            "k_est": Link(),
        }
        self.output_ports: Ports = {"roots": Link()}


    def execute(self) -> None:
        roots_batched = self.input_ports["roots"].value
        k_est_batched = self.input_ports["k_est"].value
        selected_roots_batched = []

        for roots, k_est in zip(roots_batched, k_est_batched):
            k_est = int(k_est.numpy())
            if k_est <= 0:
                selected_roots_batched.append(roots[:0])
                continue

            sort_idx = tf.keras.ops.argsort(tf.abs(tf.abs(roots) - 1))
            sorted_roots = tf.gather(roots, sort_idx)
            roots_near_unit = sorted_roots[(tf.abs(sorted_roots) - 1) < self.eps]

            if len(roots_near_unit) < k_est:
                roots_near_unit = sorted_roots

            selected_roots_batched.append(roots_near_unit[:k_est])

        self.output_ports["roots"].value = tf.stack(selected_roots_batched)
