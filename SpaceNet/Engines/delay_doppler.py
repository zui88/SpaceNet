import numpy as np

from SpaceNet.Capabilities.deep_augment import DeepAugment, ModelName
from SpaceNet.Recipes.deep_delay_doppler import DeepDelayDoppler
from SpaceNet.Recipes.classic_delay_doppler import ClassicDelayDopplerFast, ClassicDelayDoppler
from SpaceNet.Utils.decorators import static_vars
from SpaceNet.Engines.engine import CapabilityRegistryType, DelayDoppler
from SpaceNet.Configs.DelayDoppler.config import Config as DDConfig
from SpaceNet.Engines.engine import RetDD, Engine
from SpaceNet.Recipes.recipe import Recipe

from typing import Callable

from tensorflow import keras


class DelayDopperEngine(Engine[RetDD]):


    def __init__(self, config: DDConfig, recipe_cls: type[Recipe]):
        self.recipe: Recipe | None = None
        self.capability_registry: CapabilityRegistryType = {}
        self.recipe_cls: type[Recipe] = recipe_cls
        self.config: DDConfig = config

        self.dispatcher: dict[type[Recipe], Callable[[], Recipe]] = {
            ClassicDelayDopplerFast: self._construct_classic_dd_fast,
            ClassicDelayDoppler: self._construct_classic_dd,
            DeepDelayDoppler: self._construct_da_dd,
        }


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


    def _construct_da_dd(self) -> Recipe:
        da_capability                                      = self.capability_registry[DeepAugment]
        models: dict[ModelName, keras.models.Model] | None = da_capability.get_models()
        if models is None:
            raise NotImplementedError("no models registered yet")

        d_sources = self.config.base.d_sources
        if d_sources is not None:
            return DeepDelayDoppler(
                d_sources,
                self.config.base.scan_range,
                self.config.base.signal_provider,
                self.config.observ_ctx,
                models["surrogate"],
                models["finder"],
                self.config.deep_augmented.eps_rcov,
            )
        else:
            raise NotImplementedError("d_sources not specified, inference not supported")


    def _construct_classic_dd_fast(self) -> Recipe:
        d_sources = self.config.base.d_sources

        if d_sources is not None:
            return ClassicDelayDopplerFast(
                d_sources=d_sources,
                observ_ctx=self.config.observ_ctx,
                signal_provider=self.config.base.signal_provider,
            )
        else:
            raise NotImplementedError("d_sources not specified, inference not supported")


    def _construct_classic_dd(self) -> Recipe:
        d_sources = self.config.base.d_sources

        if d_sources is not None:
            return ClassicDelayDoppler(
                d_sources=d_sources,
                scan_range=self.config.base.scan_range,
                observation_context=self.config.observ_ctx,
                signal_provider=self.config.base.signal_provider,
            )
        else:
            raise NotImplementedError("d_sources not specified, inference not supported")


    @static_vars(tries=0)
    def estimate(self, **inputs) -> RetDD:
        if self.recipe is not None:
            result = self.recipe.run(r_sensed=inputs["r_sensed"])
            # todo
            try:
                return DelayDoppler(result["tau_est"], result["omega_est"]), result
            except KeyError:
                return DelayDoppler(result["tau_est"], np.ndarray([])), result
        else:
            if DelayDopperEngine.estimate.tries < 5:
                DelayDopperEngine.estimate.tries += 1
                self.init()
                return self.estimate(**inputs)
            raise NotImplementedError("Engine deferred initialization not implemented")
