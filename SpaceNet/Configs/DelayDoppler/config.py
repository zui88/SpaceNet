from SpaceNet.Synthesizer.signal import ObservationContext
from SpaceNet.Configs.base import Config as BaseConfig

from msgspec import Struct, field


class Observation(Struct):
    T: float = 8.0


class Config(Struct):
    base: BaseConfig = field(
        default_factory=BaseConfig
    )
    observation: Observation = field(default_factory=Observation)


    @property
    def observ_ctx(self) -> ObservationContext:
        return ObservationContext(T=self.observation.T)
