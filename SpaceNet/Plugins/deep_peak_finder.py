import tensorflow as tf
from tensorflow import keras

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class DeepPeakFinder(Plugin):


    def __init__(self, finder_network: tf.keras.Model) -> None:
        self.finder_network      = finder_network
        self.input_ports: Ports  = {"spectrum": Link()}
        self.output_ports: Ports = {"doa": Link()}


    def execute(self) -> None:
        spectrum_batched = self.input_ports["spectrum"].value

        if isinstance(spectrum_batched, list):
            spectrum_batched = tf.stack(spectrum_batched)

        self.output_ports["doa"].value = self.finder_network(spectrum_batched)
