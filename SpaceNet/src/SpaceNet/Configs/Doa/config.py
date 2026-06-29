from .. import deep_augmented
from .. import base

from msgspec import Struct, field


type BaseConfig = base.Config
type DeepAugmentedConfig = deep_augmented.Config


class Config(Struct, frozen=True):
    base: BaseConfig = field(default_factory=base.Config)
    deep_augmented: DeepAugmentedConfig = field(default_factory=deep_augmented.Config)
