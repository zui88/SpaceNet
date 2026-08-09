from SpaceNet.Capabilities.configuration import Configuration
from SpaceNet.Capabilities.configuration_json import JsonEncoderDecoder
from SpaceNet.Capabilities.deep_augment_v1 import DeepAugmentV1
from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Engines.utils import make_trainable
from SpaceNet.Engines.doa import DoaEngine
from SpaceNet.Engines.delay_doppler import DelayDopperEngine
from SpaceNet.Engines.engine import Engine, RetDoa, RetDD
from SpaceNet.Recipes.recipe import Recipe
from SpaceNet.Configs.Doa.config import Config as DoaConfig
from SpaceNet.Configs.DelayDoppler.config import Config as DDConfig

from tensorflow import keras

from typing import Callable, Any

from pathlib import Path


def build_networks(
    names: tuple[str, ...],
    creator_functions: tuple[Callable[[...], keras.Model], ...],
    args_functions: tuple[dict[str, Any], ...],
) -> dict[str, keras.Model]:
    return {
        name: create_model(**args)
        for name, create_model, args in zip(names, creator_functions, args_functions)
    }


def create_deep_music_engine(
    configs: DoaConfig | DDConfig | None,
    engine_type: type[Engine],
    cls: type[Recipe],
    model_names: list[str],
    defined_models: dict[str, keras.Model] | None,
    config_dir: Path,
) -> Engine[RetDoa] | Engine[RetDD]:
    configs_load_save_path = config_dir / "configs"
    if configs is not None:
        configDecoder = JsonEncoderDecoder(configs)
        configDecoder.save(configs_load_save_path)
    else:
        configDecoder = (
            JsonEncoderDecoder(DoaConfig)
            if engine_type is DoaEngine
            else JsonEncoderDecoder(DDConfig)
        )
        configs = configDecoder.load(configs_load_save_path)

    DeepEngine = (
        make_trainable(DoaEngine)
        if engine_type is DoaEngine
        else make_trainable(DelayDopperEngine)
    )
    engine = DeepEngine(configs, cls)
    engine.capability_registry[Configuration] = configDecoder
    # base_path = configs.deep_augmented.model_dir
    # todo make deep v1 using Path instead of strings -> Windows, Lunux, Unix
    base_path = str(config_dir)
    base_name = configs.deep_augmented.base_name

    capability = DeepAugmentV1(engine, model_names, base_name, base_path)

    if defined_models is None:
        capability.load_models()
    else:
        capability.register_models(defined_models)

    engine.capability_registry[DeepAugment] = capability

    engine.init()

    return engine
