from SpaceNet.Capabilities.configuration import Configuration
from SpaceNet.Capabilities.configuration_json import JsonEncoderDecoder
from SpaceNet.Recipes.deep_root_doa import DeepRootDOA
from SpaceNet.Recipes.recipe import Recipe
from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Capabilities.deep_augment_v1 import DeepAugmentV1
from SpaceNet.Engines.engine import RetDoa
from SpaceNet.Configs.Doa import config as DoaConfig
from SpaceNet.Engines.doa import DoaEngine
from SpaceNet.Engines.engine import Engine
from SpaceNet.Recipes.classic_doa import ClassicDOA
from SpaceNet.Recipes.deep_classic_doa_without_selector import DeepClassicDOA as DeepClassicDOA_WS
from SpaceNet.Recipes.deep_classic_doa import DeepClassicDOA
from SpaceNet.Recipes.root_doa import RootDOA
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.finder import build_network as build_finder
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.selector import build_network as build_selector
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.surrogate import build_network as build_surrogate
from SpaceNet.Utils.DeepAugmented.Networks.DeepRoot.rcov import build_network as build_rcov

from typing import Callable, Any, Optional, Type, Dict, Tuple, List

import numpy as np
import keras


def create_classic_music(configs: DoaConfig.Config) -> Engine[RetDoa]:
    """

    Parameters
    ----------
    configs
        'doa_base'

    Returns
    -------

    """
    return DoaEngine(configs, ClassicDOA)


def create_root_music(configs: DoaConfig.Config) -> Engine[RetDoa]:
    return DoaEngine(configs, RootDOA)


def _make_trainable(cls: Type[DoaEngine]) -> Type[DoaEngine]:
    """
    Extend the given class to be a keras model that supports training.  The 'trainable' class
    takes care of passing the arguments to the right subclass and also initializes the keras
    model.

    Parameters
    ----------
    cls

    Returns
    -------
    DoaEngine
        The returned engine is 'silently' trainable, though the user can also implicitly issue
        keras methods/functions even if those are hidden.

    """
    class TrainableDoaEngine(keras.Model, cls):


        def __init__(self, *args, **inputs) -> None:
            super().__init__()
            cls.__init__(self, *args, **inputs)
            self.transformer = None


        def call(self, inputs):
            r_sensed    = inputs
            predictions = self.estimate(r_sensed=r_sensed)
            return predictions[0]._thetas


        def estimate(self, **inputs) -> RetDoa:
            return cls.estimate(self, **self.transform(**inputs))


        def transform(self, r_sensed):
            if self.transformer is None:
                self.transformer = self._capability_registry[DeepAugment].transformer
                if self.transformer is None: raise NotImplementedError("No transformer registered")

            return {"r_sensed":self.transformer(r_sensed)}


    return TrainableDoaEngine


def build_networks(names: Tuple[str, ...], creator_funs: Tuple[Callable[..., keras.Model], ...], args_funs: Tuple[Dict[str, Any], ...]) -> Dict[str, keras.Model]:
    return {name: create_model(**args) for name, create_model, args in zip(names, creator_funs, args_funs)}


def create_deep_classic_music(configs: Optional[DoaConfig.Config] = None, define_models: bool = True, kind: str = "ws") -> Engine[RetDoa]:
    """

    Parameters
    ----------
    configs
    define_models
        either when fitting the model it means do not load a preexisting model and refine that or
        for the estimation case it means just loading already pretrained models.

    kind
        sl: with selector network
        ws: without selector network

    Returns
    -------

    """

    defined_models: Optional[Dict[str, keras.Model]] = None
    model_names: List[str] = []

    # Replace match/case with if/elif/else
    if kind == "sl":
        model_names = ["surrogate", "selector", "finder"]

        if not define_models:
            defined_models = build_networks(("surrogate", "finder", "selector"),
                                            (build_surrogate, build_finder, build_selector),
                                            ({"m_antennas":configs.base.array.antennas, "n_samples":configs.base.signal.n_samples},
                                             {"scan_range":configs.base.scan_range, "m_antennas":configs.base.array.antennas, "d_sources":configs.base.d_sources},
                                             {"m_antennas":configs.base.array.antennas},
                                             ))
        engine = _create_deep_music(configs, DeepClassicDOA, model_names, defined_models)

    else:  # default case (ws)
        model_names = ["surrogate", "finder"]

        if not define_models:
            defined_models = build_networks(("surrogate", "finder"),
                                            (build_surrogate, build_finder),
                                            ({"m_antennas":configs.base.array.antennas, "n_samples":configs.base.signal.n_samples},
                                             {"scan_range":configs.base.scan_range, "m_antennas":configs.base.array.antennas, "d_sources":configs.base.d_sources},
                                             ))
        engine = _create_deep_music(configs, DeepClassicDOA_WS, model_names, defined_models)


    class StandardScaler:


        def __init__(self):
            self.mean = None
            self.std  = None


        def fit(self, X):
            self.mean = np.mean(X, axis=0, keepdims=True)
            self.std  = np.std(X, axis=0, keepdims=True)


        def transform(self, X):
            scaled = (X - self.mean) / self.std
            return scaled


        def save(self, path_name):
            np.savez(path_name, mean=self.mean, std=self.std)


        def is_initialized(self) -> bool:
            return self.mean is not None


        @classmethod
        def load(cls, path_name):
            data_dict   = np.load(path_name)
            scaler      = cls()
            scaler.mean = data_dict["mean"]
            scaler.std  = data_dict["std"]

            return scaler


    engine.capability_registry[DeepAugment].register_transformer(StandardScaler)

    return engine


def create_deep_root_music(configs: Optional[DoaConfig.Config] = None, define_models: bool = False, activation_value: float = 0.3) -> Engine[RetDoa]:
    """

    Parameters
    ----------
    configs
    define_models
    activation_value
        Activation value for the RELU layer

    Returns
    -------

    """

    defined_models: Optional[Dict[str, keras.Model]] = None
    if define_models:
        defined_models = build_networks(("rcov",), (build_rcov,), ({"m_antennas":configs.base.array.antennas, "activation_value":activation_value},))

    return _create_deep_music(configs, DeepRootDOA,["rcov"], defined_models)


def _create_deep_music(configs: Optional[DoaConfig.Config], cls: Type[Recipe], model_names: List[str], defined_models: Optional[Dict[str, keras.Model]]) -> Engine[RetDoa]:
    configs_load_save_path = 'configs'
    if configs is not None:
        configDecoder = JsonEncoderDecoder(configs)
        configDecoder.save(configs_load_save_path)
    else:
        configDecoder = JsonEncoderDecoder(DoaConfig.Config)
        configs       = configDecoder.load(configs_load_save_path)

    DeepEngine                                = _make_trainable(DoaEngine)
    engine                                    = DeepEngine(configs, cls)
    engine._capability_registry[Configuration] = configDecoder
    base_path                                 = configs.deep_augmented.model_dir
    base_name                                 = configs.deep_augmented.base_name

    capability = DeepAugmentV1(engine, model_names, base_name, base_path)

    if defined_models is None:
        capability.load_models()
    else:
        capability.register_models(defined_models)

    engine._capability_registry[DeepAugment] = capability

    engine.init()

    return engine
