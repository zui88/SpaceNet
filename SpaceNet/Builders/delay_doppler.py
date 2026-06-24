from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Capabilities.dummy_scaler import DummyScaler
from SpaceNet.Builders.utils import create_deep_music_engine, build_networks
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.surrogate import build_network as build_surrogate
from SpaceNet.Utils.DeepAugmented.Networks.DeepClassic.finder import build_network as build_finder
from SpaceNet.Configs.DelayDoppler.config import Config
from SpaceNet.Engines.engine import Engine
from SpaceNet.Engines.delay_doppler import DelayDopperEngine
from SpaceNet.Engines.engine import RetDD
from SpaceNet.Recipes.classic_delay_doppler import ClassicDelayDoppler
from SpaceNet.Recipes.deep_delay_doppler import DeepDelayDoppler


def create_delay_doppler_music(configs: Config) -> Engine[RetDD]:
    return DelayDopperEngine(configs, ClassicDelayDoppler)


def create_deep_doppler_music(configs: Config, define_models: bool = False) -> Engine[RetDD]:

    defined_models = None
    model_names    = ["surrogate", "finder"]

    if define_models:
        defined_models = build_networks(tuple(model_names),
                                        (build_surrogate, build_finder),
                                        (
                                            {"m_antennas": configs.base.array.antennas, "n_samples": configs.n_sample_space, "mode":"n-space"},
                                            {"scan_range": configs.base.scan_range, "n_samples": configs.n_sample_space, "d_sources": configs.base.d_sources},
                                         ))

    engine = create_deep_music_engine(configs, DelayDopperEngine, DeepDelayDoppler, model_names, defined_models)

    engine.capability_registry[DeepAugment].register_transformer(DummyScaler)

    return engine