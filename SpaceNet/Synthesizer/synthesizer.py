from SpaceNet.Synthesizer.signal import SignalGenerator, ObservationContext
from SpaceNet.Configs.DelayDoppler.config import Config as DDConfig
from SpaceNet.Synthesizer.geometry import ArrayGeometry

from abc import ABC, abstractmethod

import numpy as np


class SignalSynthesizer(ABC):
    def __init__(self, snr_db: float | tuple[float, float] = 30):
        self.rng = np.random.default_rng()

        if isinstance(snr_db, tuple):
            snr_db = self.rng.uniform(snr_db[0], snr_db[1])

        self.sigma2 = np.power(10, -snr_db / 10)

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    def get_rand(self, N: int, M: int = None):
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
            raise RuntimeError("wront dimension set!")

        return w


class DOASignalSynthesizer(SignalSynthesizer):
    def __init__(self, array_geometry, signal_generator, *args, **kwargs):
        """
        Parameters
        ----------

        """
        super().__init__(*args, **kwargs)
        self.array_geometry = array_geometry
        self.signal_generator = signal_generator

    def generate(self, thetas: np.ndarray):
        """
        generate the sensed signal

        Parameters
        ----------

        thetas : list
            direction of arrival angles
        """
        D = len(thetas)
        A = self.array_geometry.get_steering(thetas)
        Q = self.signal_generator.generate(array=D)
        w = self.get_noise_matrix((A.shape[0], Q.shape[1]))
        R = A @ Q + w
        return R


class DelayDopplerSignalSynthesizer(SignalSynthesizer):
    def __init__(
        self,
        array_geometry: ArrayGeometry,
        signal_generator: SignalGenerator,
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
        self.array_geometry = array_geometry
        self.signal_generator = signal_generator
        self.T = observ_ctx.window_length
        self.fs = signal_generator.fs

    @property
    def n_samples(self) -> int:
        # because in general T is a float
        n = self.T * self.fs
        return int(n)

    def generate(
        self,
        taus: tuple[float, ...] = (0.5, 3),
        omegas: tuple[float, ...] = (0.01, -0.03),
        thetas: tuple[float, ...] | None = (0, 30),
    ):
        M = self.array_geometry.m_antennas
        N = self.n_samples

        Q = self.generate_q(taus, omegas)
        if thetas is None:
            # then it's a random array
            A = self.array_geometry.get_steering(np.array(taus))
        else:
            A = self.array_geometry.get_steering(np.array(thetas))
        w = self.get_noise_matrix((N, M))
        R = Q @ A + w  # (NxM)
        return R

    def generate_q(self, taus: tuple[float, ...], omegas: tuple[float, ...]):
        """
        RETURN
        ------
            signal matrix with shape(NxD)
        """
        if not len(taus) == len(omegas):
            raise RuntimeError("taus and omega must be the same length!")

        taus = np.array(taus)
        omegas = np.array(omegas)

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
        omegas: tuple[float, ...],
        thetas: tuple[float, ...] | None = None,
    ):
        return self.synthesizer.generate(taus, omegas, thetas)

    def generate_q(self, taus: tuple[float, ...], omegas: tuple[float, ...]):
        return self.synthesizer.generate_q(taus, omegas)
