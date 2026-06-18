from SpaceNet.Synthesizer.synthesizer import DOASignalSynthesizer
from SpaceNet.Synthesizer.geometry import ULAArray

import numpy as np


def _generate_checked_matrix(
        training_examples,
        max_signal_sources,
        doa_low,
        doa_high,
        min_doa_spacing,
        rng,
    ):
    """
    Produce a uniformly distributed training set with an ensured spacing between sources.

    Parameters
    ----------
    training_examples
    max_signal_sources
    doa_low
    doa_high
    min_doa_spacing
    rng

    Returns
    -------

    """

    doa_deg_set = rng.uniform(
        doa_low,
        doa_high,
        size=(training_examples, max_signal_sources),
    )

    valid = np.zeros(training_examples, dtype=bool)

    while not np.all(valid):
        idx = ~valid

        # resample only invalid rows
        doa_deg_set[idx] = rng.uniform(
            doa_low,
            doa_high,
            size=(idx.sum(), max_signal_sources),
        )

        # check spacing constraint
        valid[idx] = np.all(
            np.diff(np.sort(doa_deg_set[idx], axis=1), axis=1)
            >= min_doa_spacing,
            axis=1,
        )

    return doa_deg_set[:, np.newaxis, :]


def generate_data_set(
        signal_generator,
        array_geometry= None,
        samples: int = 1_000,
        max_signal_sources: int = 4,
        min_signal_sources: int | None = None,
        snr_db: tuple[float, float] | float = 30.0,
        deg_range: tuple[float, float] = (-70.0, 70.0),
        min_spacing: float | int = 7.5,
        seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray] | None:
    """

    Parameters
    ----------
    signal_generator
    array_geometry
    samples
    max_signal_sources
    min_signal_sources
    snr_db
    deg_range
    min_spacing
        the space between signal sources in one measurement

    seed

    Returns
    -------

    """

    doa_low, doa_high = deg_range
    if doa_low > doa_high:
        raise ValueError("low must be <= high.")

    if samples <= 0:
        raise ValueError("n_examples must be > 0.")
    if max_signal_sources <= 0:
        raise ValueError("max_signal_sources must be > 0.")

    if min_signal_sources is None:
        min_signal_sources = max_signal_sources
    if min_signal_sources <= 0 or min_signal_sources > max_signal_sources:
        raise ValueError(
            "min_signal_sources must be > 0 and <= max_signal_sources."
        )

    if type(snr_db) is tuple:
        snr_low, snr_high = snr_db
        if snr_low > snr_high:
            raise ValueError("snr_db high must be gr than snr_db low.")

    if array_geometry is None:
        array_geometry = ULAArray()

    if (max_signal_sources - 1) * min_spacing > (doa_high - doa_low):
        raise ValueError("Requested minimum spacing is impossible.")

    rng         = np.random.default_rng(seed)
    doa_deg_set = _generate_checked_matrix(
            samples,
            max_signal_sources,
            doa_low,
            doa_high,
            min_spacing,
            rng,
        )
    doa_rad_set = np.vectorize(np.deg2rad)(doa_deg_set)

    if type(snr_db) is float or type(snr_db) is int:
        synthesizer = DOASignalSynthesizer(
            array_geometry=array_geometry,
            signal_generator=signal_generator,
            snr_db=snr_db,
        )

    def generate_signals(doas):
        if type(snr_db) is tuple:
            return DOASignalSynthesizer(
                array_geometry=array_geometry,
                signal_generator=signal_generator,
                snr_db=rng.uniform(low=snr_low, high=snr_high),
            ).generate(doas.flatten())
        return synthesizer.generate(doas.flatten())

    signal_set = np.array(list(map(generate_signals, doa_rad_set)))

    return signal_set, np.reshape(doa_rad_set, (samples, max_signal_sources))
