from tensorflow import keras


def build_network(m_antennas) -> keras.Model:
    selector_network = keras.Sequential(
        [
            keras.Input((2 * m_antennas,)),
            keras.layers.Dense(2 * m_antennas, activation="sigmoid"),
            keras.layers.Dense(2 * m_antennas, activation="sigmoid"),
            keras.layers.Dense(2 * m_antennas, activation="sigmoid"),
            keras.layers.Dropout(rate=0.2),
            keras.layers.Dense(m_antennas),
        ]
    )
    return selector_network
