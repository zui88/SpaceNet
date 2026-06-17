from SpaceNet.Recipes.classic_delay_doppler import ClassicDelayDoppler
from SpaceNet.Utils.decorators import static_vars
from SpaceNet.Engines.engine import CapabilityRegistryType, DelayDoppler
from SpaceNet.Configs.DelayDoppler.config import Config as DDConfig
from SpaceNet.Engines.engine import RetDD, Engine
from SpaceNet.Recipes.recipe import Recipe

from typing import Callable, Optional, Type, Dict


class DelayDopperEngine(Engine[RetDD]):


    def __init__(self, config: DDConfig, recipe_cls: Type[Recipe]):
        self.recipe: Optional[Recipe] = None
        self._capability_registry: CapabilityRegistryType = {}
        self.recipe_cls: Type[Recipe] = recipe_cls
        self.config = config

        self.dispatcher: Dict[Type[Recipe], Callable[[], Recipe]] = {
            ClassicDelayDoppler: self._construct_classic_dd,
        }


    @property
    def capability_registry(self) -> CapabilityRegistryType:
        """Property to access the capability registry"""
        return self._capability_registry


    def init(self):
        """
        The init function shall be called after capabilities has been registered.

        Returns
        -------

        """
        try:
            self.recipe = self.dispatcher[self.recipe_cls]()
        except KeyError as e:
            raise NotImplementedError(f"Recipe not registered for dispatching: {e}")


    def _construct_classic_dd(self) -> Recipe:
        d_sources = self.config.base.d_sources

        if d_sources is not None:
            return ClassicDelayDoppler(
                d_sources=d_sources,
                observ_ctx=self.config.observ_ctx,
                signal_provider=self.config.base.signal_provider,
            )
        else:
            raise NotImplementedError("d_sources not specified, inference not supported")


    @static_vars(tries=0)
    def estimate(self, **inputs) -> RetDD:
        if self.recipe is not None:
            result = self.recipe.run(r_sensed=inputs["r_sensed"])
            return DelayDoppler(result["tau_est"], result["omega_est"]), result
        else:
            if DelayDopperEngine.estimate.tries < 5:
                DelayDopperEngine.estimate.tries += 1
                self.init()
                return self.estimate(**inputs)
            raise NotImplementedError("Engine deferred initialization not implemented")
