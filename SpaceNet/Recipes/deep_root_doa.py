from SpaceNet.Plugins.root_doa import ComputeRootDOA
from SpaceNet.Plugins.root_selector import RootSelector
from SpaceNet.Plugins.root_spectrum import ComputeRootSpectrum
from SpaceNet.Plugins.noise_subspace import NoiseSubspace
from SpaceNet.Plugins.signal_sources import SignalSources
from SpaceNet.Plugins.estimated_rcov import EstimateRcov
from SpaceNet.Plugins.evd import EVD
from SpaceNet.Plugins.subspace_net_estimated_rcov import SubspaceNetEstimateRcov
from SpaceNet.Recipes.recipe import Recipe

import keras


class DeepRootDOA(Recipe):
    def __init__(
        self,
        rcov_network: keras.models.Model,
        d_sources: int | None = None,
        inference_mode: bool = False,
        tau: int = 8,
        eps_roots: float = 1e-5,
        eps_rcov: float = 1,
    ):

        super().__init__()

        self.d_sources = d_sources
        self.inference_mode = inference_mode
        self.rcov_network = rcov_network
        self.eps_roots = eps_roots
        self.eps_rcov = eps_rcov
        self.tau = tau

        self.register_node(
            "estimated_rcov",
            SubspaceNetEstimateRcov(
                self.rcov_network,
                eps=self.eps_rcov,
                tau=self.tau,
            ),
        )
        self.register_node("evd", EVD())
        self.register_node(
            "signal_sources", SignalSources(self.d_sources, self.inference_mode)
        )
        self.register_node("noise_subspace", NoiseSubspace())
        self.register_node("root_spec", ComputeRootSpectrum())
        self.register_node("root_selector", RootSelector(self.eps_roots))
        self.register_node("doa", ComputeRootDOA())

        self.connect_node("input", "r_sensed", "estimated_rcov", "r_sensed")
        self.connect_node("estimated_rcov", "surrogate_rcov", "evd", "rcov")
        self.connect_node("evd", "eigs", "signal_sources", "eigs")
        self.connect_node("evd", "eigsv", "noise_subspace", "eigsv")
        self.connect_node("signal_sources", "k_est", "noise_subspace", "k_est")
        self.connect_node("noise_subspace", "Un", "root_spec", "Un")
        self.connect_node("root_spec", "roots", "root_selector", "roots")
        self.connect_node("signal_sources", "k_est", "root_selector", "k_est")
        self.connect_node("root_selector", "roots", "doa", "roots")

        self.connect_node("doa", "doa", "output", "doa")
        self.connect_node("root_spec", "roots", "output", "roots")
