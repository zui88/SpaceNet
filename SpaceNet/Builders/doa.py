from SpaceNet.Configs.Doa import config as DoaConfig
from SpaceNet.Engines.make_trainable import make_trainable
from SpaceNet.Capabilities.configuration import Configuration
from SpaceNet.Capabilities.configuration_json import JsonEncoderDecoder
from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Capabilities.deep_augment_v1 import DeepAugmentV1
from SpaceNet.Capabilities.standard_scaler import StandardScaler
from SpaceNet.Capabilities.dummy_scaler import DummyScaler
from SpaceNet.Recipes.deep_root_doa import DeepRootDOA
from SpaceNet.Recipes.recipe import Recipe
from SpaceNet.Recipes.classic_doa import ClassicDOA
from SpaceNet.Recipes.deep_classic_doa_without_selector import DeepClassicDOA as DeepClassicDOA_WS
from SpaceNet.Recipes.deep_classic_doa import DeepClassicDOA
from SpaceNet.Recipes.root_doa import RootDOA
from SpaceNet.Engines.engine import RetDoa
from SpaceNet.Engines.doa import DoaEngine
from SpaceNet.Engines.engine import Engine
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.finder import build_network as build_finder
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.selector import build_network as build_selector
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.surrogate import build_network as build_surrogate
from SpaceNet.Utils.DeepAugmented.Networks.DeepRoot.rcov import build_network as build_rcov

from typing import Callable, Any

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


def build_networks(names: tuple[str, ...], creator_funs: tuple[Callable[[...], keras.Model], ...], args_funs: tuple[dict[str, Any], ...]) -> dict[str, keras.Model]:
    return {name: create_model(**args) for name, create_model, args in zip(names, creator_funs, args_funs)}


def create_deep_classic_music(configs: DoaConfig.Config | None = None, define_models: bool = False, kind: str = "ws") -> Engine[RetDoa]:
    """

    Parameters
    ----------
    configs
    define_models
        Either when fitting the model it means do not load a preexisting model and refine that, but the configuration must be given in advance.
        For the estimation case it means just loading already pretrained models.

    kind
        sl: with selector network
        ws: without selector network

    Returns
    -------

    """

    defined_models = None

    match kind:

        case "sl":
            model_names = ["surrogate", "selector", "finder"]

            if define_models:
                defined_models = build_networks(("surrogate", "finder", "selector"),
                                                (build_surrogate, build_finder, build_selector),
                                                ({"m_antennas":configs.base.array.antennas, "n_samples":configs.base.signal.n_samples},
                                                 {"scan_range":configs.base.scan_range, "m_antennas":configs.base.array.antennas, "d_sources":configs.base.d_sources},
                                                 {"m_antennas":configs.base.array.antennas},
                                                 ))
            engine = _create_deep_music(configs, DeepClassicDOA, model_names, defined_models)

        case _:
            model_names = ["surrogate", "finder"]

            if define_models:
                defined_models = build_networks(("surrogate", "finder"),
                                                (build_surrogate, build_finder),
                                                ({"m_antennas":configs.base.array.antennas, "n_samples":configs.base.signal.n_samples},
                                                 {"scan_range":configs.base.scan_range, "m_antennas":configs.base.array.antennas, "d_sources":configs.base.d_sources},
                                                 ))
            engine = _create_deep_music(configs, DeepClassicDOA_WS, model_names, defined_models)

    # todo
    #engine.capability_registry[DeepAugment].register_transformer(StandardScaler)
    engine.capability_registry[DeepAugment].register_transformer(DummyScaler)

    return engine


def create_deep_root_music(configs: DoaConfig.Config | None = None, define_models: bool = False, activation_value: float = 0.3) -> Engine[RetDoa]:
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

    defined_models = None
    if define_models:
        defined_models = build_networks(("rcov",), (build_rcov,), ({"m_antennas":configs.base.array.antennas, "activation_value":activation_value},))

    return _create_deep_music(configs, DeepRootDOA,["rcov"], defined_models)


def _create_deep_music(configs: DoaConfig.Config | None, cls: type[Recipe], model_names: list[str], defined_models: dict[str, keras.Model] | None) -> Engine[RetDoa]:
    configs_load_save_path = 'configs'
    if configs is not None:
        configDecoder = JsonEncoderDecoder(configs)
        configDecoder.save(configs_load_save_path)
    else:
        configDecoder = JsonEncoderDecoder(DoaConfig.Config)
        configs       = configDecoder.load(configs_load_save_path)

    DeepEngine                                = make_trainable(DoaEngine)
    engine                                    = DeepEngine(configs, cls)
    engine.capability_registry[Configuration] = configDecoder
    base_path                                 = configs.deep_augmented.model_dir
    base_name                                 = configs.deep_augmented.base_name

    capability = DeepAugmentV1(engine, model_names, base_name, base_path)

    if defined_models is None:
        capability.load_models()
    else:
        capability.register_models(defined_models)

    engine.capability_registry[DeepAugment] = capability

    engine.init()

    return engine
