from SpaceNet.Plugins.plugin import Link, Plugin, Ports

import tensorflow as tf
import numpy as np


class ComputeDOA(Plugin):
    def __init__(self, scan_range: int):
        self.input_ports: Ports = {"peaks": Link()}
        self.output_ports: Ports = {"doa": Link()}

        self.scan_range = scan_range

    def execute(self) -> None:
        peaks_batched = self.input_ports["peaks"].value
        scan_range = np.linspace(-np.pi / 2, np.pi / 2, self.scan_range)
        angles_batched = []

        for peaks in peaks_batched:
            if peaks is None or len(peaks) == 0:
                angles_batched.append(tf.convert_to_tensor([], dtype=tf.float64))
                continue
            angles_batched.append(tf.gather(scan_range, peaks))

        try:
            self.output_ports["doa"].value = tf.stack(angles_batched)
        except ValueError:
            self.output_ports["doa"].value = angles_batched


class ComputeDelayDoppler(Plugin):
    def __init__(self):
        self.input_ports: Ports = {
            "peaks": Link(),
            "taus_grid": Link(),
            "omegas_grid": Link(),
        }
        self.output_ports: Ports = {
            "taus": Link(),
            "omegas": Link(),
        }

    def execute(self) -> None:
        idx = self.input_ports["peaks"].value
        taus_grid = self.input_ports["taus_grid"].value
        omegas_grid = self.input_ports["omegas_grid"].value

        # batch_dims=1: use the first dimension as the batch dimension
        taus_est = tf.gather(taus_grid, idx, axis=1, batch_dims=1)
        omegas_est = tf.gather(omegas_grid, idx, axis=1, batch_dims=1)

        self.output_ports["taus"].value = taus_est
        self.output_ports["omegas"].value = omegas_est
