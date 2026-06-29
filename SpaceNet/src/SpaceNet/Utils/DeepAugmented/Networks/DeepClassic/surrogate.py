from tensorflow import keras


def build_network(
    m_antennas: int, n_samples: int, mode: str = "m-space"
) -> keras.Model:
    """

    Parameters
    ----------
    m_antennas
        number of sensors

    n_samples
        number of observations

    mode
        Defines if the model is composed for m-space or n-space.
        m-space for DOA estimations, n-space for Delay Doppler estimations.

    Returns
    -------

    """
    if mode == "n-space":
        m_antennas, n_samples = n_samples, m_antennas

    return keras.Sequential(
        [
            # The outputs size is given to each layer.  The input will be adjusted corresponding to the output of the previous layer.
            keras.layers.Input((n_samples, 2 * m_antennas)),
            keras.layers.BatchNormalization(),
            keras.layers.GRU(2 * m_antennas, dropout=0.2, recurrent_dropout=0.2),
            keras.layers.Dense((2 * m_antennas * m_antennas)),
            keras.layers.Reshape((2 * m_antennas, m_antennas)),
        ]
    )
