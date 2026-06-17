import sys
sys.path.append(".")

from SpaceNet.Utils.DeepAugmented.TrainingData.doa import generate_training_set
from SpaceNet.Utils.DeepAugmented.print import save_print_history
from SpaceNet.Builders.doa import create_deep_classic_music
from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Configs.Doa.deep_classic import config

from sklearn.model_selection import train_test_split
from tensorflow import keras


def train():
    config.base.d_sources        = 4
    config.base.signal.n_samples = 200
    config.base.array.antennas   = 8
    config.base.snr_db           = (-5, 30)

    R, doa = generate_training_set(
        signal_generator=config.base.signal_provider,
        array_geometry=config.base.array_geometry,
        doa_range_deg=(-70.0, 70.0),
        min_doa_spacing=5, # 15 grad guy
        training_examples=100_000,
        max_signal_sources=config.base.d_sources,
        snr_db=config.base.snr_db,
    )
    R_train, R_test, doa_train, doa_test = train_test_split(R, doa, test_size=0.12, random_state=42)

    model                            = create_deep_classic_music(config, False)
    history: keras.callbacks.History = model.capability_registry[DeepAugment].train_models(
        R_train,
        doa_train,
        (R_test, doa_test),
        batch_size=100,
        epochs=200,
        learning_rate=0.001,
        patience=10,
    )

    model.capability_registry[DeepAugment].save_models()
    save_print_history(history)


if __name__ == "__main__":
    train()
