from tensorflow import keras


def build_network(m_antennas, n_samples) -> keras.Model:
    """

    Parameters
    ----------
    m_antennas
        number of sensors

    n_samples
        number of observations

    Returns
    -------

    """
    return keras.Sequential([
        # The outputs size is given to each layer.  The input will be adjusted corresponding to the output of the previous layer.
        keras.layers.Input((n_samples, 2 * m_antennas)),
        keras.layers.BatchNormalization(),
        keras.layers.GRU(2 * m_antennas, dropout=0.2, recurrent_dropout=0.2),
        keras.layers.Dense((2 * m_antennas * m_antennas)),
        keras.layers.Reshape((2 * m_antennas, m_antennas)),
    ])

