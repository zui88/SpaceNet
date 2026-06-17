from SpaceNet.Capabilities.deep_augment import DeepAugment

from typing import Any, Protocol, TypeVar, TypeAlias, Union, Dict, Type
from dataclasses import dataclass

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

    # Use private attribute to prevent Keras serialization issues with Type keys
    _capability_registry: CapabilityRegistryType
    configs: Any

    @property
    def capability_registry(self) -> CapabilityRegistryType:
        """Property to access the capability registry"""
        ...

    def estimate(self, **inputs) -> T:
        ...
