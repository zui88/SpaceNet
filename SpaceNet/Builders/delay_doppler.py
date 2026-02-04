from SpaceNet.Engines.delay_doppler import DelayDopperEngine
from SpaceNet.Configs.DelayDoppler.config import Config
from SpaceNet.Engines.engine import Engine
from SpaceNet.Engines.engine import RetDD
from SpaceNet.Recipes.classic_delay_doppler import ClassicDelayDoppler


def create_delay_doppler_music(configs: Config) -> Engine[RetDD]:
    return DelayDopperEngine(configs, ClassicDelayDoppler)
