import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


def find_roots(coeffs: list[tf.Tensor]) -> tf.Tensor:
    coeffs = tf.cast(tf.stack(coeffs), dtype=tf.complex128)
    first_row = tf.reshape(-coeffs[1:] / coeffs[0], [1, -1])
    companion = tf.keras.ops.diag(
        tf.keras.ops.ones(coeffs.shape[0] - 2, dtype=coeffs.dtype),
        k=-1,
    )
    companion = tf.concat([first_row, companion[1:, :]], axis=0)
    return tf.linalg.eigvals(companion)


class ComputeRootSpectrum(Plugin):
    def __init__(self):
        self.input_ports: Ports = {"Un": Link()}
        self.output_ports: Ports = {"roots": Link()}

    def execute(self) -> None:
        un_batched = self.input_ports["Un"].value
        roots_batched = []

        for un in un_batched:
            projector = tf.matmul(un, un, adjoint_b=True)
            n_sensors = projector.shape[0]
            diag_coeffs = [
                tf.keras.ops.sum(tf.keras.ops.diag(projector, offset))
                for offset in range(-(n_sensors - 1), n_sensors)
            ]
            roots_batched.append(find_roots(diag_coeffs[::-1]))

        self.output_ports["roots"].value = tf.stack(roots_batched)
