from SpaceNet.Plugins.delay_spectrum import DelaySpectrum
from SpaceNet.Plugins.doa import ComputeDelayDoppler
from SpaceNet.Plugins.noise_subspace import NoiseSubspace
from SpaceNet.Plugins.evd import EVD
from SpaceNet.Plugins.peak_finder import PeakFinder
from SpaceNet.Plugins.estimated_rcov import EstimateRcovFFT as EstimateRcov
from SpaceNet.Plugins.signal_sources import SignalSources
from SpaceNet.Plugins.classic_delay_doppler import (
    ClassicDelayDoppler as DelayDopplerPlugin,
)
from SpaceNet.Recipes.recipe import Recipe
from SpaceNet.Synthesizer.signal import SignalGenerator, ObservationContext


class ClassicDelayDopplerFast(Recipe):
    def __init__(
        self,
        d_sources: int,
        observ_ctx,
        signal_provider,
    ):
        super().__init__()

        #############################################
        # register the plugins
        #############################################
        self.register_node(
            "classic_delay_doppler",
            DelayDopplerPlugin(d_sources, observ_ctx, signal_provider),
        )

        #############################################
        # connect the plugins
        #############################################
        self.connect_node("input", "r_sensed", "classic_delay_doppler", "r_sensed")

        self.connect_node("classic_delay_doppler", "tau_est", "output", "tau_est")
        self.connect_node("classic_delay_doppler", "omega_est", "output", "omega_est")

        self.connect_node("classic_delay_doppler", "tau_grid", "output", "tau_grid")
        self.connect_node(
            "classic_delay_doppler", "cost_function", "output", "cost_function"
        )


class ClassicDelayDoppler(Recipe):
    def __init__(
        self,
        d_sources: int,
        scan_range: int,
        observation_context: ObservationContext,
        signal_provider: SignalGenerator,
    ):
        super().__init__()

        #############################################
        # register the plugins
        #############################################
        self.register_node("estimated_rcov", EstimateRcov())
        self.register_node("evd", EVD())
        self.register_node(
            "signal_sources",
            SignalSources(
                d_sources,
                False,
            ),
        )
        self.register_node("noise_subspace", NoiseSubspace())
        self.register_node(
            "inv_spec",
            DelaySpectrum(
                scan_range,
                signal_provider,
                observation_context,
            ),
        )
        self.register_node("peak_finder", PeakFinder(mode="dd-idx"))
        self.register_node("delay_doppler", ComputeDelayDoppler())

        #############################################
        # connect the plugins
        #############################################
        self.connect_node("input", "r_sensed", "estimated_rcov", "r_sensed")

        self.connect_node("estimated_rcov", "r_cov", "evd", "r_cov")
        self.connect_node("evd", "eigs", "signal_sources", "eigs")
        self.connect_node("evd", "eigs_v", "noise_subspace", "eigs_v")
        self.connect_node("signal_sources", "d_est", "noise_subspace", "d_est")
        self.connect_node("noise_subspace", "Un", "inv_spec", "Un")

        self.connect_node("inv_spec", "spectrum", "peak_finder", "spectrum")
        self.connect_node("inv_spec", "taus_grid", "delay_doppler", "taus_grid")
        self.connect_node("inv_spec", "omegas_grid", "delay_doppler", "omegas_grid")

        self.connect_node("signal_sources", "d_est", "peak_finder", "d_est")

        self.connect_node("peak_finder", "peaks", "delay_doppler", "peaks")

        self.connect_node("delay_doppler", "taus", "output", "tau_est")
        self.connect_node("delay_doppler", "omegas", "output", "omega_est")
        self.connect_node("inv_spec", "spectrum_obj", "output", "spectrum_obj")
