from SpaceNet.Capabilities.deep_augment import DeepAugment

from dataclasses import dataclass
from typing import Any, Protocol

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

    @property
    def raw(self) -> np.ndarray:
        return self._thetas


@dataclass(frozen=True, slots=True)
class DelayDoppler:
    delay: np.ndarray
    # todo
    doppler: np.ndarray

    @property
    def raw(self) -> np.ndarray:
        return self.delay


type RetDoa = tuple[Doa, Any]
type RetDD = tuple[DelayDoppler, Any]

type Capability = Any | DeepAugment
type CapabilityRegistryType = dict[type[Capability], Capability]


class Engine[T](Protocol):
    """
    Parameters
    ----------
    configs
        The correct config for a distinct type of engine.  The engine builder is responsible for
        registering the config.
    """

    capability_registry: CapabilityRegistryType
    configs: Any

    def estimate(self, **inputs) -> T: ...

    def save_config(self) -> None: ...
