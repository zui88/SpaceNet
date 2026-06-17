from SpaceNet.Configs import deep_augmented
from SpaceNet.Configs import base

from typing import TypeAlias

from msgspec import Struct, field


BaseConfig: TypeAlias = base.Config
DeepAugmentedConfig: TypeAlias = deep_augmented.Config


class Config(Struct, frozen=True):
    base: BaseConfig = field(
        default_factory=base.Config
    )
    deep_augmented: DeepAugmentedConfig = field(
        default_factory=deep_augmented.Config
    )
