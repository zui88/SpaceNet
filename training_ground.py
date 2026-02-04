import typer
from doa import DeepRootMusic, RMSPELoss, DeepMusic
from generator import DOASignalSynthesizer, RandomSignal, ULAArray
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras as keras
from utils import save_print_history


def generate_training_set(
    signal_generator,
    array_geometry = None,
    training_examples : int = 1_000,
    max_signal_sources : int = 4,
    min_signal_sources : int | None = None,
    snr_db : tuple[float, float] | float = 30.0,
    doa_range_deg : tuple[float, float] = (-70.0, 70.0),
    seed : int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None :
    """
    Build synthetic training set for DOA estimation.

    RETURNS
    -------
    X
        complex-valued signals; Shape: (n_examples, n_antennas, n_samples)
    
    y
        DOA labels in radians; Shape: (n_examples, max_signal_sources)
    """

    doa_low, doa_high = doa_range_deg
    if doa_low > doa_high:
        raise ValueError("low must be <= high.")

    if training_examples <= 0:
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

    rng         = np.random.default_rng(seed)
    doa_deg_set = rng.uniform(low=doa_low, high=doa_high, size=(training_examples, 1, max_signal_sources,))
    doa_rad_set = np.vectorize(np.deg2rad)(doa_deg_set)

    if type(snr_db) is float:
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

    return signal_set, np.reshape(doa_rad_set, (training_examples, max_signal_sources))


app = typer.Typer()


##################################################
# training configuration
##################################################
@app.command()
def deep_root():
    model_name       = "modelv1_root_doa_rcov.keras"
    array_geometry   = ULAArray(antennas=10)
    signal_generator = RandomSignal(n_samples=50)
    model            = DeepRootMusic(
        steering_provider=array_geometry,
        training=True,
        d_sources=4,
        inference_mode=False
    )
    R, doa = generate_training_set(
        signal_generator=signal_generator,
        array_geometry=array_geometry,
        doa_range_deg=(-89.0, 89.0),
        training_examples=1_000,
        max_signal_sources=4,
        snr_db=(10.0, 40.0),
        seed=42,
    )
    R_train, R_test, doa_train, doa_test = train_test_split(R, doa, test_size=0.2, random_state=42)

    optimizer = keras.optimizers.Adam(
        learning_rate=0.00001,
        weight_decay=1e-9,
    )
    model.compile(
        optimizer=optimizer,
        loss=RMSPELoss(),
        run_eagerly=True, # has to be set because of evaluation errors -> roots less than the unit circle cannot be found
    )
    history = model.fit(
        R_train,
        doa_train,
        epochs=200,
        batch_size=30,
    )

    model.model.save(model_name)

    save_print_history(history)


@app.command()
def deep_music():
    model_name       = "classic"
    array_geometry   = ULAArray(antennas=10)
    signal_generator = RandomSignal(n_samples=50)
    music_engine     = DeepMusic(
        steering_provider=array_geometry,
        signal_provider=signal_generator,
        model_name=model_name,
        training=True,
        transfer_learning = True,
        d_sources=4,
        inference_mode=False,
    )
    R, doa = generate_training_set(
        signal_generator=signal_generator,
        array_geometry=array_geometry,
        training_examples=300,
        doa_range_deg=(-40, 40),
        max_signal_sources=4,
        snr_db=(10.0, 40.0,),
        seed=42,
    )
    R_train, R_test, doa_train, doa_test = train_test_split(R, doa, test_size=0.2, random_state=42)

    optimizer = keras.optimizers.Adam(
        learning_rate=0.00001,
        weight_decay=1e-9,
    )
    music_engine.compile(
        optimizer=optimizer,
        loss=RMSPELoss(),
        run_eagerly=True, # has to be set because of evaluation errors -> roots less than the unit circle cannot be found
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,  # new_lr = lr * 0.1
        patience=5,  # wait epochs
        min_lr=1e-6,
        verbose=1,
    )

    stop_fit = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
    )

    history = music_engine.fit(
        R_train,
        doa_train,
        validation_data=(R_test, doa_test),
        epochs=200,
        batch_size=32,
        #callbacks=[ reduce_lr, stop_fit, ],
    )

    music_engine.save()

    save_print_history(history)


if __name__ == "__main__":
    app()