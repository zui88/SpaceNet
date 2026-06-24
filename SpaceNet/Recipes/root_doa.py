from SpaceNet.Plugins.estimated_rcov import EstimateRcov
from SpaceNet.Plugins.evd import EVD
from SpaceNet.Plugins.noise_subspace import NoiseSubspace
from SpaceNet.Plugins.root_doa import ComputeRootDOA
from SpaceNet.Plugins.root_selector import RootSelector
from SpaceNet.Plugins.root_spectrum import ComputeRootSpectrum
from SpaceNet.Plugins.signal_sources import SignalSources
from SpaceNet.Recipes.recipe import Recipe


class RootDOA(Recipe):


    def __init__(
            self,
            d_sources: int | None = None,
            inference_mode: bool = True,
            eps_roots: float = 1e-5
        ):

        super().__init__()

        self.d_sources = d_sources
        self.inference_mode = inference_mode
        self.eps = eps_roots

        self.register_node("estimated_rcov", EstimateRcov())
        self.register_node("evd", EVD())
        self.register_node("signal_sources", SignalSources(self.d_sources, self.inference_mode))
        self.register_node("noise_subspace", NoiseSubspace())
        self.register_node("root_spec", ComputeRootSpectrum())
        self.register_node("root_selector", RootSelector(self.eps))
        self.register_node("doa", ComputeRootDOA())

        self.connect_node("input", "r_sensed",
                          "estimated_rcov", "r_sensed")
        self.connect_node("estimated_rcov", "r_cov",
                          "evd", "r_cov")
        self.connect_node("evd", "eigs",
                          "signal_sources", "eigs")
        self.connect_node("evd", "eigs_v",
                          "noise_subspace", "eigs_v")
        self.connect_node("signal_sources", "d_est",
                          "noise_subspace", "d_est")
        self.connect_node("noise_subspace", "Un",
                          "root_spec", "Un")
        self.connect_node("root_spec", "roots",
                          "root_selector", "roots")
        self.connect_node("signal_sources", "d_est",
                          "root_selector", "d_est")
        self.connect_node("root_selector", "roots",
                          "doa", "roots")

        self.connect_node("doa", "doa",
                          "output", "doa")
        self.connect_node("root_spec", "roots",
                          "output", "roots")
