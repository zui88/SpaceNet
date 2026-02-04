from SpaceNet.Plugins.plugin import Ports, Link, Plugin
from SpaceNet.Configs.base import SteeringType

from dataclasses import dataclass

import tensorflow as tf
import numpy as np


@dataclass(frozen=True)
class Spectrum:
    """
    for printing the spectrum
    """
    spectrum: tf.Tensor
    scan_range: tf.Tensor


class ComputePseudoInverseSpectrum(Plugin):


    def __init__(self, scan_range: int, steering: SteeringType):
        """

        Parameters
        ----------
        scan_range
        steering
        """
        self.scan_range = scan_range
        self.steering   = steering

        self.input_ports: Ports  = {"Un": Link()}
        self.output_ports: Ports = {"spectrum": Link(),
                                    "SpectrumObj": Link()}


    def execute(self):
        Un_batched = self.input_ports["Un"].value

        spectrum_batched: list[tf.Tensor] = []
        spectrum_obj: list[Spectrum]      = []

        for Un in Un_batched:
            # hypothesis: if angles are pointing to sources
            scan_range = tf.keras.ops.linspace(-1 * np.pi / 2, np.pi / 2, self.scan_range, endpoint=False)
            # pyrefly: ignore [bad-argument-count]
            a          = self.steering(scan_range)

            a          = tf.cast(a, dtype=tf.complex64)
            Un         = tf.cast(Un, dtype=tf.complex64)

            # pseudo spectrum: 1 / (a^H(theta) * Un * Un^H * a(theta))
            projection = tf.matmul(Un, a, adjoint_a=True)
            spectrum   = 1 / tf.keras.ops.sum(tf.keras.ops.abs(projection)**2, axis=0)
            spectrum_batched.append(spectrum)
            spectrum_obj.append(Spectrum(spectrum, scan_range))

        self.output_ports["spectrum"].value    = spectrum_batched
        self.output_ports["SpectrumObj"].value = spectrum_obj
