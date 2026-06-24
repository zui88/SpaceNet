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
                                    "spectrum_obj": Link()}


    def execute(self):
        Un = self.input_ports["Un"].value

        # hypothesis: if angles are pointing to sources
        scan_range = tf.range(self.scan_range, dtype=tf.float32)
        scan_range = -np.pi/2 + np.pi * scan_range/self.scan_range

        # pyrefly: ignore [bad-argument-count]
        a = self.steering(scan_range)
        a = tf.cast(a, dtype=tf.complex64)

        Un = tf.cast(Un, dtype=tf.complex64)

        # pseudo spectrum: 1 / (a^H(theta) * Un * Un^H * a(theta))
        projection = tf.matmul(Un, a, adjoint_a=True)
        spectrum   = 1 / tf.reduce_sum(tf.abs(projection)**2, axis=1)

        self.output_ports["spectrum"].value     = spectrum
        self.output_ports["spectrum_obj"].value = Spectrum(spectrum, scan_range)
