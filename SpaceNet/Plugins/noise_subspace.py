import tensorflow as tf

from SpaceNet.Plugins.plugin import Link, Plugin, Ports


class NoiseSubspace(Plugin):


    def __init__(self, d_sources: int | None = None) -> None:
        self.input_ports: Ports = {
            "eigsv": Link(),
            "k_est": Link(),
        }
        self.output_ports: Ports = {"Un": Link()}
        self.d_sources           = d_sources


    def execute(self) -> None:
        eigsv = self.input_ports["eigsv"].value
        k_est = self.input_ports["k_est"].value[0] if self.d_sources is None else self.d_sources

        # todo varying signal sources aren't supported right now
        Un = eigsv[:, :, k_est:]

        self.output_ports["Un"].value = Un
