from SpaceNet.Configs.base import SteeringType
from SpaceNet.Plugins.doa import ComputeDOA
from SpaceNet.Plugins.estimated_rcov import EstimateRcov
from SpaceNet.Plugins.evd import EVD
from SpaceNet.Plugins.noise_subspace import NoiseSubspace
from SpaceNet.Plugins.peak_finder import PeakFinder
from SpaceNet.Plugins.pseudo_inverse_spectrum import ComputePseudoInverseSpectrum
from SpaceNet.Plugins.signal_sources import SignalSources
from SpaceNet.Recipes.recipe import Recipe


class ClassicDOA(Recipe):


    def __init__(
            self,
            steering: SteeringType,
            scan_range: int,
            d_sources: int | None = None,
            inference_mode: bool = True):
        super().__init__()

        self.steering: SteeringType = steering
        self.scan_range: int = scan_range
        self.d_sources = d_sources
        self.inference_mode = inference_mode

        #############################################
        # register the plugins
        #############################################
        self.register_node("estimated_rcov", EstimateRcov())
        self.register_node("evd", EVD())
        self.register_node("signal_sources", SignalSources(self.d_sources, self.inference_mode))
        self.register_node("noise_subspace", NoiseSubspace())
        self.register_node("inv_spec", ComputePseudoInverseSpectrum(self.scan_range, self.steering))
        self.register_node("peak_finder", PeakFinder())
        self.register_node("doa", ComputeDOA(self.scan_range))

        #############################################
        # connect the plugins
        #############################################
        self.connect_node("input", "r_sensed",
                          "estimated_rcov", "r_sensed")

        self.connect_node("estimated_rcov", "rcov",
                          "evd", "rcov")
        self.connect_node("evd", "eigs",
                          "signal_sources", "eigs")
        self.connect_node("evd", "eigsv",
                          "noise_subspace", "eigsv")
        self.connect_node("signal_sources", "k_est",
                          "noise_subspace", "k_est")
        self.connect_node("noise_subspace", "Un",
                          "inv_spec", "Un")
        self.connect_node("inv_spec", "spectrum",
                          "peak_finder", "spectrum")
        self.connect_node("signal_sources", "k_est",
                          "peak_finder", "k_est")
        self.connect_node("peak_finder", "peaks",
                          "doa", "peaks")

        self.connect_node("doa", "doa",
                          "output", "doa")
        self.connect_node("inv_spec", "SpectrumObj",
                          "output", "SpectrumObj")
