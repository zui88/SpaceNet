from tensorflow import keras


def build_network(m_antennas: int, tau: int = 8, activation_value: float = 0.3) -> keras.Model:
    return keras.Sequential([
        # Input block #0
        keras.Input(shape=(2 * m_antennas, m_antennas, tau)),
        # CNN block #1
        keras.layers.Conv2D(16, kernel_size=2, name='conv1'),
        keras.layers.LeakyReLU(activation_value),
        # CNN block #2
        keras.layers.Conv2D(32, kernel_size=2, name='conv2'),
        keras.layers.LeakyReLU(activation_value),
        # CNN block #3
        keras.layers.Conv2D(64, kernel_size=2, name='conv3'),
        keras.layers.LeakyReLU(activation_value),
        # DCNN block #1
        keras.layers.Conv2DTranspose(32, kernel_size=2, name='deconv1'),
        keras.layers.LeakyReLU(activation_value),
        # dcnn block #2
        keras.layers.Conv2DTranspose(16, kernel_size=2, name='deconv2'),
        keras.layers.LeakyReLU(activation_value),
        # dcnn block #3
        keras.layers.Dropout(0.2),
        keras.layers.Conv2DTranspose(1, kernel_size=2, name='deconv3'),
    ])
