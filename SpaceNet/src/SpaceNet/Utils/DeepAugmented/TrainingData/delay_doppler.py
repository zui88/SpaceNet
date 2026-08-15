from SpaceNet.Synthesizer.signal import ObservationContext
from SpaceNet.Synthesizer.geometry import RandomArray
from SpaceNet.Synthesizer.synthesizer import DelayDopplerSignalSynthesizer

import numpy as np

from time import time


module_rng = np.random.default_rng(abs(hash(str(time()))))

def sample_delay_doppler(
    rng: np.random.Generator,
    d_sources: int,
    delay_range: tuple[float, float],
    doppler_range: tuple[float, float],
    min_separation_delay: float,
    min_separation_doppler: float,
    max_attempts: int = 1_000,
) -> np.ndarray:
    """
    Sample `d_sources` delay-doppler pairs with independent minimum spacing
    constraints in delay and doppler.

    RETURNS
    -------
    delay

    doppler
    """

    low, high = delay_range
    if low > high:
        raise ValueError("delay_range must be (low, high) with low <= high.")
    if d_sources <= 0:
        raise ValueError("d_sources must be > 0.")
    if min_separation_delay < 0:
        raise ValueError("min_separation_delay must be >= 0.")

    doppler_low, doppler_high = doppler_range
    if doppler_low >= doppler_high:
        raise ValueError("doppler_range must be (low, high) with low < high.")
    if min_separation_doppler < 0:
        raise ValueError("min_separation_doppler must be >= 0.")

    span = high - low
    if min_separation_delay * (d_sources - 1) > span:
        raise ValueError(
            "Cannot fit all sources in delay_range with the required min_separation_delay."
        )
    doppler_span = doppler_high - doppler_low
    if min_separation_doppler * (d_sources - 1) > doppler_span:
        raise ValueError(
            "Cannot fit all sources in doppler_range with the required min_separation_doppler."
        )

    for _ in range(max_attempts):
        delays = np.sort(rng.uniform(low, high, size=d_sources))
        dopplers = np.sort(rng.uniform(doppler_low, doppler_high, size=d_sources))
        if np.all(np.diff(delays) >= min_separation_delay) and np.all(
            np.diff(dopplers) >= min_separation_doppler
        ):
            return np.stack([delays, dopplers], axis=0)

    raise RuntimeError(
        "Failed to sample valid delay-doppler pairs. Relax the separation constraints or widen the ranges."
    )


def generate_data_set(
    signal_generator,
    training_examples: int = 1_000,
    min_signal_sources: int | None = None,
    max_signal_sources: int = 4,
    n_samples: int = 50,
    snr_db: float = 10.0,
    snr_db_range: tuple[float, float] | None = None,
    delay_range: tuple[float, float] = (0.6, 78.0),
    min_delay_separation: float = 0.2,
    doppler_range: tuple[float, float] = (-2.3, 2.3),
    min_doppler_separation: float = 0.05,
    sort_pairs: bool = False,
    seed: int | None = None,
    array_geometry=None,
    observ_ctx: ObservationContext | None = None,
    correlation_coefficient: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build a synthetic training set for deep delay-doppler MUSIC.

    RETURNS
    -------
    X
        real-valued sensed signals with split real/imag channels;
        Shape: (n_examples, n_samples, 2 * n_antennas)

    y
        Delay-doppler labels; Shape: (n_examples, max_signal_sources, 2)
    """

    if training_examples <= 0:
        raise ValueError("n_examples must be > 0.")
    if max_signal_sources <= 0:
        raise ValueError("max_signal_sources must be > 0.")
    if n_samples <= 0:
        raise ValueError("n_samples must be > 0.")

    if min_signal_sources is None:
        min_signal_sources = max_signal_sources
    if min_signal_sources <= 0 or min_signal_sources > max_signal_sources:
        raise ValueError("min_signal_sources must be > 0 and <= max_signal_sources.")

    if observ_ctx is None:
        observ_ctx = ObservationContext(T=n_samples / signal_generator.fs)
    if array_geometry is None:
        array_geometry = RandomArray()

    rng = module_rng
    if seed is not None:
        rng = np.random.default_rng(seed)

    d_sources = int(rng.integers(min_signal_sources, max_signal_sources + 1))
    signal_set = []
    dd_set = []

    snr_db_set = [snr_db for _ in range(training_examples)]
    if snr_db_range is not None:
        snr_low, snr_high = snr_db_range
        if snr_low > snr_high:
            raise ValueError("snr_db_range must be (low, high) with low <= high.")
        snr_db_set = [
            float(rng.uniform(snr_low, snr_high)) for _ in range(training_examples)
        ]

    correlation_matrix = None
    if correlation_coefficient is not None:
        correlation_matrix = np.array(
            [[1.0, correlation_coefficient], [correlation_coefficient, 1.0]]
        )

    for snr in snr_db_set:
        dd_pair = sample_delay_doppler(
            rng=rng,
            d_sources=d_sources,
            delay_range=delay_range,
            doppler_range=doppler_range,
            min_separation_delay=min_delay_separation,
            min_separation_doppler=min_doppler_separation,
        )
        if sort_pairs:
            dd_pair = dd_pair[np.argsort(dd_pair[0])]

        synthesizer = DelayDopplerSignalSynthesizer(
            array_geometry=array_geometry,
            signal_generator=signal_generator,
            observ_ctx=observ_ctx,
            snr_db=snr,
        )
        signal = synthesizer.generate(
            taus=dd_pair[0],
            omegas=dd_pair[1],
            thetas=np.zeros(d_sources),
            correlation_matrix=correlation_matrix,
        )
        padded_pairs = np.zeros((max_signal_sources, 2), dtype=np.float32)
        padded_pairs[:d_sources] = dd_pair.astype(np.float32)

        signal_set.append(signal)
        dd_set.append(padded_pairs)

    X = np.stack(signal_set)
    y = np.stack(dd_set)

    return X, y
