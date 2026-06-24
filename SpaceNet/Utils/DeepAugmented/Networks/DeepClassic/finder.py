from tensorflow import keras


def build_network(scan_range, d_sources, m_antennas: int | None = None, n_samples: int | None = None) -> keras.Model:
    """Build the finder network.  One of m_antennas or n_samples is required.
    When m_antennas and n_samples are given at once, m_antennas will be used as sources.

    Parameters
    ----------
    scan_range
    d_sources
    m_antennas
        for m-space applications like DOA Estimation

    n_samples
        for n-space applications like Delay Doppler Estimation

    Returns
    -------

    """
    if m_antennas is None and n_samples is None:
        raise ValueError('m_antennas and n_samples cannot be None')

    if m_antennas is None:
        m_antennas = n_samples

    return keras.Sequential([
        keras.layers.Input((scan_range,)),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(d_sources),
    ])
