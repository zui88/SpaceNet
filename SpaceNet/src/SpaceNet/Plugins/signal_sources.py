import numpy as np
import scipy.stats as stats
import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


def _mos_penalty(i, eigs, n_samples, n_sensors):
    noise = eigs[i:]
    am = np.mean(noise)
    gm = stats.gmean(noise)
    return -1 * (n_samples * (n_sensors - 1) * np.log(gm / am))


def _mdl(i, eigs, n_samples, n_sensors):
    penalty = _mos_penalty(i, eigs, n_samples, n_sensors)
    return penalty + 0.5 * i * (2 * n_sensors - i) * np.log(n_samples)


def _compute_unambiguous_sources(eigs, n_samples, n_sensors):
    return np.argmin(
        [_mdl(i, eigs, n_samples, n_sensors) for i in range(n_sensors - 1)]
    )


class SignalSources(Plugin):
    def __init__(self, d_sources: int | None = None, inference_mode: bool = True):
        """lambda_min multiplicity: estimate the number of impinging signals traversing the array configuration."""
        self.d_sources = d_sources
        self.inference_mode = inference_mode
        self.input_ports: Ports = {"eigs": Link()}
        self.output_ports: Ports = {"d_est": Link()}

    def execute(self) -> None:
        eigs = self.input_ports["eigs"].value

        if not self.inference_mode:
            batch_size = eigs.eigs_batch.shape[0]
            self.output_ports["d_est"].value = tf.fill(
                [batch_size],
                tf.cast(self.d_sources, tf.int32),
            )
        else:
            d_est_batched = []

            for eig_values, n_samples, m_sensors in zip(
                eigs.eigs_batch,
                eigs.n_samples_batch,
                eigs.m_sensors_batch,
            ):
                d_est = _compute_unambiguous_sources(
                    tf.math.real(eig_values).numpy(),
                    int(n_samples.numpy()),
                    int(m_sensors.numpy()),
                )
                d_est_batched.append(d_est)

            self.output_ports["d_est"].value = tf.convert_to_tensor(d_est_batched)
