from SpaceNet.Plugins.deep_estimated_rcov import DeepEstimateRcov
from SpaceNet.Plugins.deep_peak_finder import DeepPeakConverter
from SpaceNet.Plugins.noise_subspace import NoiseSubspace
from SpaceNet.Plugins.delay_spectrum import DelaySpectrum
from SpaceNet.Synthesizer.signal import SignalGenerator, ObservationContext
from SpaceNet.Recipes.recipe import Recipe
from SpaceNet.Plugins.evd import EVD

from tensorflow import keras


class DeepDelayDoppler(Recipe):
    def __init__(
        self,
        d_sources: int,
        scan_range: int,
        signal_provider: SignalGenerator,
        observation_context: ObservationContext,
        surrogate_network: keras.models.Model,
        finder_network: keras.models.Model,
        eps: float = 1.0,
        **_,
    ):
        """

        Parameters
        ----------
        d_sources
        steering
        scan_range
        surrogate_network
        finder_network
        eps
            epsilon parameter for the surrogate network
        _
            dummy object
        """

        super().__init__()

        #############################################
        # register the plugins
        #############################################
        self.register_node(
            "estimated_rcov",
            DeepEstimateRcov(
                surrogate_network,
                eps,
            ),
        )
        self.register_node("evd", EVD())
        self.register_node("noise_subspace", NoiseSubspace(d_sources))
        self.register_node(
            "inv_spec",
            DelaySpectrum(
                scan_range,
                signal_provider,
                observation_context,
            ),
        )
        self.register_node(
            "peak_finder",
            DeepPeakConverter(
                finder_network,
            ),
        )

        #############################################
        # connect the plugins
        #############################################
        self.connect_node("input", "r_sensed", "estimated_rcov", "r_sensed")

        self.connect_node("estimated_rcov", "surrogate_rcov", "evd", "r_cov")

        self.connect_node("evd", "eigs_v", "noise_subspace", "eigs_v")

        self.connect_node("noise_subspace", "Un", "inv_spec", "Un")

        self.connect_node("inv_spec", "spectrum", "peak_finder", "spectrum")

        self.connect_node("peak_finder", "value", "output", "tau_est")
        self.connect_node("inv_spec", "spectrum_obj", "output", "spectrum_obj")
