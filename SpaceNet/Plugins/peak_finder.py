import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class PeakFinder(Plugin):


    def __init__(self):
        self.input_ports: Ports = {
            "spectrum": Link(),
            "k_est": Link(),
        }
        self.output_ports: Ports = {"peaks": Link()}


    def execute(self) -> None:
        spectrum_batched  = self.input_ports["spectrum"].value
        k_est_batched     = self.input_ports["k_est"].value
        top_peaks_batched = []

        for spectrum, k_est in zip(spectrum_batched, k_est_batched):
            k_est     = int(k_est.numpy())
            peaks_idx = tf.keras.ops.where(
                (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
            )[0] + 1

            if len(peaks_idx) == 0:
                peaks_idx = tf.keras.ops.argsort(spectrum)[::-1][:k_est]
            else:
                filtered_spectrum = tf.gather(spectrum, peaks_idx)
                sort_fspec_idx    = tf.keras.ops.argsort(filtered_spectrum)[::-1]
                peaks_idx         = tf.gather(peaks_idx, sort_fspec_idx)[:k_est]

            top_peaks_batched.append(peaks_idx)

        try:
            self.output_ports["peaks"].value = tf.stack(top_peaks_batched)
        except ValueError:
            self.output_ports["peaks"].value = top_peaks_batched
