from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
import tensorflow as tf


type SteeringType = (
    Callable[[np.ndarray, int], np.ndarray] | Callable[[np.ndarray], np.ndarray]
)


class ArrayGeometry(ABC):
    """
    It defines an array that consist of number of array sensors and incident signals sources.

    f.e. (Arrays x Signals)
    """

    def __init__(self, antennas=6):
        self.antennas = antennas

    @abstractmethod
    def get_steering(self, thetas: tf.Tensor, axis: int = 0) -> tf.Tensor:
        pass

    @property
    def m_antennas(self):
        m = self.antennas
        return m


class ULAArray(ArrayGeometry):
    def __init__(self, d_lambda: float = 1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_lambda = d_lambda

    def get_steering(
        self,
        thetas: tf.Tensor,
        axis: int = 0,
    ) -> tf.Tensor:
        """
        A.shape = (antennas, sources) for axis == 0
        A.shape = (sources, antennas) for axis == 1
        """
        antennas_idx = tf.cast(
            tf.range(self.m_antennas)[:, None],
            tf.float32,
        )

        A = tf.exp(
            tf.complex(
                tf.zeros_like(antennas_idx * tf.sin(thetas)),
                np.pi * self.d_lambda * antennas_idx * tf.sin(thetas),
            )
        )

        if axis == 1:
            A = tf.transpose(A)

        return A


class RandomArray(ArrayGeometry):
    """
    It defines an array that consist of number of array sensors and incident signals sources.

    f.e. (Arrays x Signals)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.default_rng()

    def get_steering(self, thetas: np.ndarray, axis: int = 0):
        if len(thetas) < 1:
            raise RuntimeError("thetas must be at least has the size of 1")

        M = self.m_antennas
        D = len(thetas)

        x = self.rng.standard_normal((D, M))
        y = self.rng.standard_normal((D, M))
        s = 1 / np.sqrt(2) * (x + 1j * y)
        return s
