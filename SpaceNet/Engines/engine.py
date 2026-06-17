from SpaceNet.Capabilities.configuration import Configuration
from SpaceNet.Capabilities.deep_augment import DeepAugment

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, TypeAlias, Union, Optional, Dict, Type

import numpy as np


@dataclass(slots=True)
class Doa:
    _thetas: np.ndarray


    @property
    def thetas(self) -> np.ndarray:
        def f(x):
            if x > np.pi / 2:
                return (x % np.pi) - np.pi
            if x < -np.pi / 2:
                return x % np.pi
            return x


        vf = np.vectorize(f)
        return vf(self._thetas)


    @thetas.setter
    def thetas(self, thetas: np.ndarray):
        self._thetas = thetas


@dataclass(frozen=True, slots=True)
class DelayDoppler:
    delay: np.ndarray
    doppler: np.ndarray


RetDoa: TypeAlias = tuple[Doa, Any]
RetDD: TypeAlias = tuple[DelayDoppler, Any]

Capability: TypeAlias = Union[Any, DeepAugment]
CapabilityRegistryType: TypeAlias = Dict[Type[Capability], Capability]

T = TypeVar('T')

class Engine(Protocol[T]):
    """
    Parameters
    ----------
    configs
        The correct config for a distinct type of engine.  The engine builder is responsible for
        registering the config.
    """


    capability_registry: CapabilityRegistryType
    configs: Any


    def estimate(self, **inputs) -> T:
        ...
