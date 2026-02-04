import numpy as np
import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class ComputeDOA(Plugin):


    def __init__(self, scan_range: int):
        self.scan_range = scan_range
        self.input_ports: Ports = {"peaks": Link()}
        self.output_ports: Ports = {"doa": Link()}


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
