import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class RootSelector(Plugin):
    def __init__(self, eps: float = 1e-5):
        self.eps = eps
        self.input_ports: Ports = {
            "roots": Link(),
            "d_est": Link(),
        }
        self.output_ports: Ports = {"roots": Link()}

    def execute(self) -> None:
        roots_batched = self.input_ports["roots"].value
        d_est_batched = self.input_ports["d_est"].value

        def select_roots(inputs):
            roots, k_est = inputs

            k_est = tf.cast(k_est, tf.int32)

            sort_idx = tf.argsort(tf.abs(tf.abs(roots) - 1))
            sorted_roots = tf.gather(roots, sort_idx)

            roots_near_unit = tf.boolean_mask(
                sorted_roots,
                (tf.abs(sorted_roots) - 1) < self.eps,
            )

            n_near_unit = tf.shape(roots_near_unit)[0]

            # If there are fewer near-unit roots than requested,
            # use all sorted roots.
            candidates = tf.cond(
                n_near_unit < k_est,
                lambda: sorted_roots,
                lambda: roots_near_unit,
            )

            # k_est <= 0 -> empty tensor
            return tf.cond(
                k_est <= 0,
                lambda: tf.zeros(
                    [0],
                    dtype=roots.dtype,
                ),
                lambda: candidates[:k_est],
            )

        selected_roots_batched = tf.map_fn(
            select_roots,
            (roots_batched, d_est_batched),
            fn_output_signature=tf.TensorSpec(
                shape=(None,),
                dtype=roots_batched.dtype,
            ),
        )

        self.output_ports["roots"].value = selected_roots_batched
