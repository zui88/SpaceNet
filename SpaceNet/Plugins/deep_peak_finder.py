from SpaceNet.Plugins.plugin import Link, Plugin, Ports

from tensorflow import keras
import tensorflow as tf
import numpy as np


class DeepPeakConverter(Plugin):
    def __init__(self, finder_network: tf.keras.Model) -> None:
        self.finder_network = finder_network
        self.input_ports: Ports = {"spectrum": Link()}
        self.output_ports: Ports = {"value": Link()}

    def execute(self) -> None:
        spectrum = self.input_ports["spectrum"].value

        if isinstance(spectrum, np.ndarray):
            spectrum = tf.stack(spectrum)

        value = self.finder_network(spectrum)
        self.output_ports["value"].value = value
