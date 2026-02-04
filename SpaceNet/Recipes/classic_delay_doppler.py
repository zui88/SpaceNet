from SpaceNet.Plugins.classic_delay_doppler import ClassicDelayDoppler as DelayDopplerPlugin
from SpaceNet.Recipes.recipe import Recipe


class ClassicDelayDoppler(Recipe):


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
        self.register_node("classic_delay_doppler", DelayDopplerPlugin(d_sources, observ_ctx, signal_provider))

        #############################################
        # connect the plugins
        #############################################
        self.connect_node("input", "r_sensed",
                          "classic_delay_doppler", "r_sensed")

        self.connect_node("classic_delay_doppler", "tau_est",
                          "output", "tau_est")
        self.connect_node("classic_delay_doppler", "omega_est",
                          "output", "omega_est")
        self.connect_node("classic_delay_doppler", "tau_grid",
                          "output", "tau_grid")
        self.connect_node("classic_delay_doppler", "cost_function",
                          "output", "cost_function")
