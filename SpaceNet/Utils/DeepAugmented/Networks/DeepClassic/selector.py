from tensorflow import keras


def build_network(m_antennas) -> keras.Model:
    return keras.Sequential([
        keras.Input((2 * m_antennas,)),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(m_antennas),
    ])

