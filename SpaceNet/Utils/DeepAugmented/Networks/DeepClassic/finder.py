from tensorflow import keras


def build_network(scan_range, m_antennas, d_sources) -> keras.Model:
    return keras.Sequential([
        keras.layers.Input((scan_range,)),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(2 * m_antennas, activation='relu'),
        keras.layers.Dense(d_sources),
    ])
