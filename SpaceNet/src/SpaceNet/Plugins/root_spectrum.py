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
    """
    Resolves the root polynomial J(z) = v^H * Un *Un^H * v.

    Warning!!! Prerequisites: works just for standard ULA configuration
    """

    def __init__(self):
        self.input_ports: Ports = {"Un": Link()}
        self.output_ports: Ports = {"roots": Link()}

    def execute(self) -> None:
        un_batched = self.input_ports["Un"].value

        def compute_roots(un):
            projector = tf.matmul(un, un, adjoint_b=True)

            m_sensors = tf.shape(projector)[0]

            offsets = tf.range(
                -(m_sensors - 1),
                m_sensors,
            )

            def get_diagonal_sum(offset):
                return tf.reduce_sum(
                    tf.linalg.diag_part(
                        projector,
                        k=offset,
                    )
                )

            diag_coeffs = tf.map_fn(
                get_diagonal_sum,
                offsets,
                fn_output_signature=projector.dtype,
            )

            return find_roots(tf.reverse(diag_coeffs, axis=[0]))

        roots_batched = tf.map_fn(
            compute_roots,
            un_batched,
            fn_output_signature=tf.TensorSpec(
                shape=(None,),
                dtype=tf.complex128,
            ),
        )

        self.output_ports["roots"].value = roots_batched

    # def execute(self) -> None:
    #     un_batched = self.input_ports["Un"].value
    #     roots_batched = []

    #     for un in un_batched:
    #         projector = tf.matmul(un, un, adjoint_b=True)
    #         m_sensors = projector.shape[0]
    #         diag_coeffs = [
    #             # just works for std ula configuration
    #             tf.keras.ops.sum(tf.keras.ops.diag(projector, offset))
    #             for offset in range(-(m_sensors - 1), m_sensors)
    #         ]
    #         roots_batched.append(find_roots(diag_coeffs[::-1]))

    #     self.output_ports["roots"].value = tf.stack(roots_batched)
