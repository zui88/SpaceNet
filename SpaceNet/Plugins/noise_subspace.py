from SpaceNet.Plugins.plugin import Link, Plugin, Ports

import tensorflow as tf


class NoiseSubspace(Plugin):


    def __init__(self, d_sources: int | None = None) -> None:
        """Segregating noise from signal space.  If 'd_sources' is given, the input port 'd_est' is ignored but will be calculated internally.

        Interface Ports
        ---------------
        Input
            eigs_v: Link
            d_est: Link

        Output
            Un: Link

        Parameters
        ----------
        d_sources
            Fixed dimension of the signal space.  Internally, the counterpart 'd_est' is statically set.
        """
        self.input_ports: Ports = {
            "eigs_v": Link(),
            "d_est": Link(),
        }
        self.output_ports: Ports = {"Un": Link()}
        self.d_sources           = d_sources


    def execute(self) -> None:
        eigs_v: tf.Tensor = self.input_ports["eigs_v"].value
        d_est: int        = self.input_ports["d_est"].value[0] if self.d_sources is None else self.d_sources

        # todo: varying signal sources aren't supported right now
        Un = eigs_v[:, :, d_est:]

        self.output_ports["Un"].value = Un
