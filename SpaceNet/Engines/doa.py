from SpaceNet.Recipes.deep_root_doa import DeepRootDOA
from SpaceNet.Utils.decorators import static_vars
from SpaceNet.Engines.engine import CapabilityRegistryType
from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Configs.Doa import config as DoaConfig
from SpaceNet.Recipes.classic_doa import ClassicDOA
from SpaceNet.Recipes.deep_classic_doa_without_selector import DeepClassicDOA as DeepClassicDOA_WS
from SpaceNet.Recipes.deep_classic_doa import DeepClassicDOA
from SpaceNet.Engines.engine import RetDoa, Doa, Engine
from SpaceNet.Recipes.root_doa import RootDOA
from SpaceNet.Recipes.recipe import Recipe

from typing import Callable, Optional, Type, Dict

from tensorflow.python.trackable.data_structures import NoDependency


class DoaEngine(Engine[RetDoa]):


    def __init__(self, configs: DoaConfig.Config, recipe_cls: Type[Recipe]):
        self.recipe: Optional[Recipe] = None
        self._capability_registry: CapabilityRegistryType = NoDependency({})
        self.recipe_cls: Type[Recipe] = recipe_cls
        self.configs = configs
        self.base_config = configs.base
        self.deep_config = configs.deep_augmented


    @property
    def capability_registry(self) -> CapabilityRegistryType:
        return self._capability_registry


    def init(self):
        """
        The init function shall be called after capabilities has been registered.

        Returns
        -------

        """
        dispatcher: Dict[Type[Recipe], Callable[[], Recipe]] = {
            DeepClassicDOA: self._construct_deep_classic_doa,
            DeepClassicDOA_WS: self._construct_deep_classic_doa_ws,
            ClassicDOA: self._construct_classic,
            RootDOA: self._construct_root_doa,
            DeepRootDOA: self._construct_deep_root_doa
        }

        try:
            self.recipe = dispatcher[self.recipe_cls]()
        except KeyError as e:
            raise NotImplementedError(f"Recipe not registered for dispatching: {e}")



    def _construct_deep_classic_doa_ws(self) -> Recipe:
        models = self._capability_registry[DeepAugment].get_models()
        if models is None:
            raise NotImplementedError("models not registered")

        if self.base_config.d_sources is not None:
            return DeepClassicDOA_WS(
                steering=self.base_config.steering,
                d_sources=self.base_config.d_sources,
                scan_range=self.base_config.scan_range,
                eps=self.deep_config.eps_rcov,
                surrogate_network=models["surrogate"],
                finder_network=models["finder"],
            )
        else:
            raise NotImplementedError("d_sources not configured")


    def _construct_deep_classic_doa(self) -> Recipe:
        models = self._capability_registry[DeepAugment].get_models()
        if models is None:
            raise NotImplementedError("models not registered")

        return DeepClassicDOA(
            steering=self.base_config.steering,
            scan_range=self.base_config.scan_range,
            eps=self.deep_config.eps_rcov,
            surrogate_network=models["surrogate"],
            finder_network=models["finder"],
            selector_network=models["selector"],
        )


    def _construct_root_doa(self) -> Recipe:
        return RootDOA(
            d_sources=self.base_config.d_sources,
            inference_mode=self.base_config.inference_mode,
            eps_roots=self.deep_config.eps_roots,
        )


    def _construct_deep_root_doa(self) -> Recipe:
        models = self._capability_registry[DeepAugment].get_models()
        if models is None:
            raise NotImplementedError("models not registered")

        return DeepRootDOA(
            rcov_network=models["rcov"],
            d_sources=self.base_config.d_sources,
            inference_mode=self.base_config.inference_mode,
            eps_rcov=self.deep_config.eps_rcov,
            eps_roots=self.deep_config.eps_roots,
        )



    def _construct_classic(self) -> Recipe:
        return ClassicDOA(
            steering=self.base_config.steering,
            scan_range=self.base_config.scan_range,
            d_sources=self.base_config.d_sources,
            inference_mode=self.base_config.inference_mode,
        )


    @static_vars(tries=0)
    def estimate(self, **inputs) -> RetDoa:
        if self.recipe is not None:
            result = self.recipe.run(r_sensed=inputs["r_sensed"])
            return Doa(result["doa"]), result
        else:
            if DoaEngine.estimate.tries < 5:
                DoaEngine.estimate.tries += 1
                self.init()
                return self.estimate(**inputs)
            raise NotImplementedError()
