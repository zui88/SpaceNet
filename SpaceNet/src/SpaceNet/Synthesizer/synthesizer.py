from SpaceNet.Synthesizer.signal import SignalGenerator, ObservationContext
from SpaceNet.Configs.DelayDoppler.config import Config as DDConfig
from SpaceNet.Configs.Doa.config import Config as DoaConfig
from SpaceNet.Synthesizer.geometry import ArrayGeometry

from abc import ABC, abstractmethod
from time import time

import tensorflow as tf
import numpy as np


class SignalSynthesizer(ABC):
    rng = np.random.default_rng(abs(hash(str(time()))))

    def __init__(
        self,
        array_geometry: ArrayGeometry,
        signal_generator: SignalGenerator,
        snr_db: float | tuple[float, float] = 30,
    ):
        self.array_geometry = array_geometry
        self.signal_generator = signal_generator

        if isinstance(snr_db, tuple):
            snr_db = self.rng.uniform(snr_db[0], snr_db[1])

        self.sigma2 = np.power(10, -snr_db / 10)

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def get_rand(self, N: int, M: int | None = None):
        return (
            self.rng.standard_normal(N)
            if M is None
            else self.rng.standard_normal((N, M))
        )

    def get_noise_matrix(self, dimension: int | tuple):
        """
        dimension : int | tuple
            either scalar or tuple of two that defines the matrix or vector

            (tuple(0) x tuple(M))
            (NxM)
        """
        w = None

        if isinstance(dimension, tuple):
            N = dimension[0]
            M = dimension[1]
            w = np.sqrt(self.sigma2 / 2) * (
                self.get_rand(N, M) + 1j * self.get_rand(N, M)
            )  # (NxM)

        if isinstance(dimension, int):
            N = dimension
            w = np.sqrt(self.sigma2 / 2) * (
                self.get_rand(N) + 1j * self.get_rand(N)
            )  # (N)

        if w is None:
            raise RuntimeError("wrong dimension set!")

        return w

    @staticmethod
    def _correlate_signals(
        Q_independant: np.ndarray,
        correlation_matrix: np.ndarray | None = None,
        mode: str = "m-space",
    ) -> np.ndarray:
        """

        Parameters
        ----------
        Q_independant
        correlation_matrix
        mode (default: "m-space")
            either "m-space" or "n-space"
            "n-space": for delay doppler estimation
            "m-space": for doa estimation

        Returns
        -------

        """
        D = Q_independant.shape[0] if mode == "m-space" else Q_independant.shape[1]

        if correlation_matrix is not None:
            # correlation matrix has square shape
            if correlation_matrix.shape[0] == D:
                L = np.linalg.cholesky(correlation_matrix)
                Q = L @ Q_independant if mode == "m-space" else Q_independant @ L
                return Q
            else:
                raise TypeError(
                    f"correlation_matrix dimensions [{correlation_matrix.shape[0]}] does not match with thetas [{D}]"
                )
        else:
            raise TypeError("correlation_matrix must be provided")


class DOASignalSynthesizer(SignalSynthesizer):
    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------

        """
        super().__init__(*args, **kwargs)

    def generate(
        self,
        thetas: np.ndarray,
        correlation_matrix: np.ndarray | None = None,
    ) -> tf.Tensor:
        """
        generate the sensed signal

        Parameters
        ----------

        thetas : np.ndarray
            direction of arrival angles

        correlation_matrix
        """
        D = thetas.shape[0]
        A = self.array_geometry.get_steering(
            tf.convert_to_tensor(thetas, tf.float32)
        )  # (M,N)
        Q = self.signal_generator.generate(array=D)  # (N,D)
        if correlation_matrix is not None:
            Q = self._correlate_signals(
                Q_independant=Q, correlation_matrix=correlation_matrix
            )
        w = self.get_noise_matrix((A.shape[0], Q.shape[1]))
        R = A @ Q + w  # (M,D)
        return R


class DoaSynthesizerWrapper:
    def __init__(self, config: DoaConfig):
        self.synthesizer = DOASignalSynthesizer(
            snr_db=config.base.snr_db,
            array_geometry=config.base.array_geometry,
            signal_generator=config.base.signal_provider,
        )

    def generate(
        self,
        thetas: tuple[float, ...],
        correlation_matrix: np.ndarray | None = None,
    ) -> tf.Tensor:
        return self.synthesizer.generate(
            thetas=np.array(thetas),
            correlation_matrix=correlation_matrix,
        )


class DelayDopplerSignalSynthesizer(SignalSynthesizer):
    def __init__(
        self,
        observ_ctx: ObservationContext,
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        T : ObservationContext
            Holds the duration of the entire observation window
        """
        super().__init__(*args, **kwargs)
        self.T = observ_ctx.window_length
        self.fs = self.signal_generator.fs

    @property
    def n_samples(self) -> int:
        # because in general T is a float
        n = self.T * self.fs
        return int(n)

    def generate(
        self,
        taus: np.ndarray,
        omegas: np.ndarray,
        thetas: np.ndarray | None = None,
        correlation_matrix: np.ndarray | None = None,
    ):
        M = self.array_geometry.m_antennas
        N = self.n_samples

        Q = self.generate_q(taus, omegas)
        if correlation_matrix is not None:
            Q = self._correlate_signals(
                Q_independant=Q, correlation_matrix=correlation_matrix, mode="n-space"
            )
        if thetas is None:
            # then it's a random array
            A = self.array_geometry.get_steering(tf.convert_to_tensor(taus, tf.float32))
        else:
            A = self.array_geometry.get_steering(
                tf.convert_to_tensor(thetas, tf.float32)
            )
        w = self.get_noise_matrix((N, M))
        R = Q @ A + w  # (NxM)
        return R

    def generate_q(
        self,
        taus: np.ndarray,
        omegas: np.ndarray,
    ):
        """
        RETURN
        ------
            signal matrix with shape(NxD)
        """
        if not taus.shape[0] == omegas.shape[0]:
            raise RuntimeError("taus and omega must be the same length!")

        t = np.arange(0, self.T, 1 / self.fs)
        s = self.signal_generator.generate(self.T)
        omegas_shift = np.exp(1j * omegas[None, :] * t[:, None])  # (NxD)
        n_omegas = omegas_shift.shape[1]
        signals = []
        for k in range(n_omegas):
            phase_shifted_signal = s * omegas_shift[:, k]
            pad = int(taus[k] * self.fs)
            time_shifted_signal = np.pad(phase_shifted_signal, (pad, 0))
            transformed_signal = time_shifted_signal[0 : self.n_samples]
            signals.append(transformed_signal)

        Q = np.stack(signals, axis=1)
        return Q


class DDSynthesizerWrapper:
    def __init__(self, config: DDConfig):
        self.synthesizer = DelayDopplerSignalSynthesizer(
            snr_db=config.base.snr_db,
            array_geometry=config.base.array_geometry,
            signal_generator=config.base.signal_provider,
            observ_ctx=config.observ_ctx,
        )

    def generate(
        self,
        taus: tuple[float, ...],
        omegas: tuple[float, ...] | None = None,
        thetas: tuple[float, ...] | None = None,
        correlation_matrix: np.ndarray | None = None,
    ):
        return self.synthesizer.generate(
            np.array(taus),
            np.array(omegas)
            if omegas is not None
            else np.array(
                taus
            ),  # basically omega doesn't matter because v0.1.0 the deep dd cannot estimate omegas
            np.array(thetas) if thetas is not None else None,
            correlation_matrix=correlation_matrix,
        )

    def generate_q(
        self,
        taus: tuple[float, ...],
        omegas: tuple[float, ...],
    ):
        return self.synthesizer.generate_q(np.array(taus), np.array(omegas))
