import typer
import matplotlib.pyplot as plt
import numpy as np
from generator import DOASignalSynthesizer, DelayDopplerSignalSynthesizer, ULAArray, RandomArray, ChirpSignal, RandomSignal, ObservationContext
from doa import RootMusic, ClassicMusic, DeepRootMusic, DelayDopplerMusic, DeepDopplerMusic, DeepMusic


app = typer.Typer()


@app.command()
def doa():
    ##################################################
    # generate the signal
    ##################################################
    thetas = np.deg2rad([-20, 20, 40, 60])
    
    array_geometry = ULAArray(antennas=10)
    r_sensed       = DOASignalSynthesizer(
        array_geometry,
        RandomSignal(n_samples=50),
        snr_db=30,
    ).generate(
        thetas,
    )
    print("true thetas: ", np.rad2deg(thetas))

    ##################################################
    # music
    ##################################################
    music_engine = ClassicMusic(steering_provider=array_geometry)
    est_thetas   = music_engine.estimate(r_batched=r_sensed[None,:])
    print("classic music: ", np.rad2deg(est_thetas[0]))

    ##################################################
    # deep music
    ##################################################
    deep_music_engine = DeepMusic(
        steering_provider=array_geometry,
        inference_mode=False,
        d_sources=thetas.shape[0],
    )
    est_thetas   = deep_music_engine.estimate(r_batched=r_sensed[None,:])
    print("deep music: ", np.rad2deg(est_thetas[0]))

    ##################################################
    # root music
    ##################################################
    root_engine = RootMusic(steering_provider=array_geometry)
    est_thetas  = root_engine.estimate(r_batched=r_sensed[None,:])
    print("root music: ", np.rad2deg(est_thetas[0]))

    ##################################################
    # deep root music
    ##################################################
    deep_root_engine = DeepRootMusic(
        steering_provider=array_geometry,
        model_load_name='model.keras',
        inference_mode=False,
        d_sources=thetas.shape[0],
    )
    est_thetas = deep_root_engine.estimate(r_batched=r_sensed[None,:])
    print("deep root music: ", np.rad2deg(est_thetas[0]))


@app.command()
def delay_doppler():
    ##################################################
    # generate the signal
    ##################################################
    taus           = [3, 5]
    omegas         = [1.3, -0.5]
    thetas         = [0, 30]
    signal_sources = len(taus)

    observ_ctx       = ObservationContext(T=8)
    signal_generator = ChirpSignal(T=0.5,fs=25)
    array_geometry   = RandomArray()

    r_sensed         = DelayDopplerSignalSynthesizer(
        array_geometry,
        signal_generator,
        observ_ctx,
        snr_db=40, # default 30
    ).generate(
        thetas=np.deg2rad(thetas),
        omegas=omegas,
        taus=taus,
    )
    print("true taus: ", taus)
    print("true omegas: ", omegas)
    print("true thetas: ", thetas)

    ##################################################
    # delay doppler classic music
    ##################################################
    music_engine = DelayDopplerMusic(
        observ_ctx,
        signal_provider=signal_generator,
        d_sources=signal_sources,
    )
    estimates = music_engine.estimate(r_batched=r_sensed[None,:])
    estimate  = estimates[0]
    
    print("model based music -- estimated taus: ", estimate.delays)
    print("model based music -- estimated omegas: ", estimate.dopplers)
    
    # print the value range over T to show where the cost function has its minima
    plt.plot(estimate.tau_grid, estimate.cost_function)
    plt.show()

    ##################################################
    # delay doppler deep music
    ##################################################
    deep_music_engine = DeepDopplerMusic(
        observ_ctx,
        signal_provider=signal_generator,
        d_sources=signal_sources,
        inference_mode=False,
    )
    estimates_dd = deep_music_engine.estimate(r_batched=r_sensed[None,:])
    estimate_dd  = estimates_dd[0]
    
    print("deep music -- estimated taus: ", estimate_dd)

    
if __name__ == "__main__":
    app()
