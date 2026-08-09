from SpaceNet.Plugins.plugin import Link, Plugin, Ports

from abc import ABC, abstractmethod

import tensorflow as tf
import numpy as np


class Finder(ABC):
    @abstractmethod
    def find(self, spectrum: tf.Tensor, d_est: tf.Tensor) -> tf.Tensor: ...


class LocalPeaks(Finder):
    def find(self, spectrum_batched: tf.Tensor, d_est_batched: tf.Tensor) -> tf.Tensor:
        peaks = []
        for spectrum, d_est in zip(spectrum_batched, d_est_batched):
            d_est = int(d_est.numpy())
            peaks_idx = (
                tf.keras.ops.where(
                    (spectrum[1:-1] > spectrum[:-2]) & (spectrum[1:-1] > spectrum[2:])
                )[0]
                + 1
            )

            if len(peaks_idx) == 0:
                peaks_idx = tf.keras.ops.argsort(spectrum)[::-1][:d_est]
            else:
                filtered_spectrum = tf.gather(spectrum, peaks_idx)
                sort_fspec_idx = tf.keras.ops.argsort(filtered_spectrum)[::-1]
                peaks_idx = tf.gather(peaks_idx, sort_fspec_idx)[:d_est]

            peaks.append(peaks_idx)

        return tf.convert_to_tensor(peaks, dtype=tf.int32)


class DelayDopplerIdx(Finder):
    def find(self, spectrum: tf.Tensor, d_est: tf.Tensor) -> tf.Tensor:
        sorted_idx = tf.argsort(spectrum, axis=-1)
        # todo for dynamic d estimation
        # d_est always the same
        d = d_est[0]
        idx = sorted_idx[:, :d]
        return idx


class PeakFinder(Plugin):
    def __init__(self, mode: str = "local-peaks"):
        """

        Parameters
        ----------
        mode: str (default 'local-peaks')
            local-peaks | dd-idx
        """
        self.input_ports: Ports = {
            "spectrum": Link(),
            "d_est": Link(),
        }
        self.output_ports: Ports = {"peaks": Link()}

        self._strategy: Finder | None
        self.mode = mode

        match mode:
            case "local-peaks":
                self._strategy = LocalPeaks()
            case "dd-idx":
                self._strategy = DelayDopplerIdx()
            case _:
                self._strategy = None

    def execute(self) -> None:
        if self._strategy is None:
            raise NotImplementedError(f"no strategy available: {self.mode}")

        spectrum = self.input_ports["spectrum"].value
        d_est = self.input_ports["d_est"].value
        peaks = self._strategy.find(spectrum, d_est)

        try:
            self.output_ports["peaks"].value = tf.stack(peaks)
        except ValueError:
            self.output_ports["peaks"].value = peaks
