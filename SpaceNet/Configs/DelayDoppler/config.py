from Synthesizer.signal import ObservationContext
from ..deep_augmented import Config as DeepAugmentedConfig
from ..base import Config as BaseConfig

from msgspec import Struct, field


class Observation(Struct):
    T: float = 8.0


class Config(Struct):
    base: BaseConfig = field(
        default_factory=BaseConfig
    )
    observation: Observation = field(
        default_factory=Observation
    )
    deep_augmented: DeepAugmentedConfig = field(
        default_factory=DeepAugmentedConfig
    )


    @property
    def n_sample_space(self) -> int:
        n_space = int(self.observation.T * self.base.signal.fs)
        return n_space


    @property
    def observ_ctx(self) -> ObservationContext:
        return ObservationContext(T=self.observation.T)
