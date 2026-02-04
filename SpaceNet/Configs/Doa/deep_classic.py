from SpaceNet.Configs.base import ArrayKind, SignalKind
from SpaceNet.Configs.Doa.config import Config


config = Config()

config.base.d_sources = 4
config.base.inference_mode = False

config.base.array.antennas = 10
config.base.array.kind = ArrayKind.ULA_ARRAY

config.base.signal.n_samples = 50
config.base.signal.kind = SignalKind.RANDOM_SIGNAL
