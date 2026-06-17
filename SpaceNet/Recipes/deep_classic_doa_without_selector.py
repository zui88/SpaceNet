from SpaceNet.Plugins.deep_estimated_rcov import DeepEstimateRcov
from SpaceNet.Plugins.deep_peak_finder import DeepPeakFinder
from SpaceNet.Plugins.evd import EVD
from SpaceNet.Plugins.noise_subspace import NoiseSubspace
from SpaceNet.Plugins.pseudo_inverse_spectrum import ComputePseudoInverseSpectrum
from SpaceNet.Plugins.signal_sources import SignalSources
from SpaceNet.Recipes.recipe import Recipe
from SpaceNet.Configs.base import SteeringType

import keras


class DeepClassicDOA(Recipe):


    def __init__(
            self,
            d_sources: int,
            steering: SteeringType,
            scan_range: int,
            surrogate_network: keras.models.Model,
            finder_network: keras.models.Model,
            eps: float = 1.0,
            **_):

        super().__init__()

        self.steering       = steering
        self.d_sources      = d_sources
        self.scan_range     = scan_range
        self.rcov_network   = surrogate_network
        self.finder_network = finder_network
        self.eps_rcov       = eps

        self.register_node("estimated_rcov", DeepEstimateRcov(
            self.rcov_network,
            self.eps_rcov,
        ))
        self.register_node("evd", EVD())
        self.register_node("noise_subspace", NoiseSubspace(d_sources))
        self.register_node("inv_spec", ComputePseudoInverseSpectrum(
            self.scan_range,
            self.steering,
        ))
        self.register_node("peak_finder", DeepPeakFinder(
            self.finder_network,
        ))

        self.connect_node("input", "r_sensed",
                          "estimated_rcov", "r_sensed")

        self.connect_node("estimated_rcov", "surrogate_rcov",
                          "evd", "rcov")

        self.connect_node("evd", "eigsv",
                          "noise_subspace", "eigsv")

        self.connect_node("noise_subspace", "Un",
                          "inv_spec", "Un")

        self.connect_node("inv_spec", "spectrum",
                          "peak_finder", "spectrum")

        self.connect_node("peak_finder", "doa",
                          "output", "doa")
        self.connect_node("inv_spec", "SpectrumObj",
                          "output", "SpectrumObj")
