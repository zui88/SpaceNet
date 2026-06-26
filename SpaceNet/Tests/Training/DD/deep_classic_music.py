from SpaceNet.Utils.DeepAugmented.TrainingData.delay_doppler import generate_data_set
from SpaceNet.Utils.DeepAugmented.LossFunctions.rmse_loss import RMSELoss
from SpaceNet.Utils.DeepAugmented.print import save_print_history
from SpaceNet.Builders.delay_doppler import create_deep_doppler_music as create_engine
from SpaceNet.Capabilities.deep_augment import DeepAugment
from SpaceNet.Configs.DelayDoppler.config import Config

from sklearn.model_selection import train_test_split
from tensorflow import keras


def train():
    config = Config()
    config.base.d_sources = 2
    config.base.array.antennas = 8
    config.base.scan_range = 200
    config.base.snr_db = (10, 40.0)

    R, dd = generate_data_set(
        training_examples=2_000,
        min_delay_separation=0.5,
        signal_generator=config.base.signal_provider,
        array_geometry=config.base.array_geometry,
        max_signal_sources=config.base.d_sources,
        min_signal_sources=config.base.d_sources,
        n_samples=config.n_sample_space,
        snr_db=config.base.snr_db,
        delay_range=(0.6, 8.0),
        observ_ctx=config.observ_ctx,
    )
    R_train, R_test, dd_train, dd_test = train_test_split(
        R, dd, test_size=0.12, random_state=42
    )

    engine = create_engine(config, define_models=True)
    history: keras.callbacks.History = engine.capability_registry[
        DeepAugment
    ].train_models(
        R_train,
        dd_train,
        (R_test, dd_test),
        batch_size=100,
        epochs=100,
        learning_rate=0.001,
        patience=8,
        loss=RMSELoss(d_sources=config.base.d_sources),
    )

    engine.capability_registry[DeepAugment].save_models()
    save_print_history(history)


if __name__ == "__main__":
    import os

    print(os.getcwd())
    train()
