from SpaceNet.Utils.DeepAugmented.TrainingData.doa import generate_data_set
from SpaceNet.Utils.DeepAugmented.LossFunctions.rmspe_loss import RMSPELoss
from SpaceNet.Builders.doa import create_deep_classic_music
from SpaceNet.Engines.engine import RetDoa, Engine, Doa
from SpaceNet.Configs.Doa.config import Config

import numpy as np


def doa():
    deep_music_engine: Engine[RetDoa] = create_deep_classic_music()
    config: Config                    = deep_music_engine.configs

    ##################################################
    # generate the signal
    ##################################################
    R, doa = generate_data_set(
        signal_generator=config.base.signal_provider,
        array_geometry=config.base.array_geometry,
        deg_range=(-70.0, 70.0),
        min_spacing=5, # 15 grad guy
        samples=5,
        max_signal_sources=config.base.d_sources,
        snr_db=30,
    )

    print("true thetas: ", np.rad2deg(doa))

    ##################################################
    # deep augmented classic music engine
    ##################################################
    doa_ret: RetDoa = deep_music_engine.estimate(r_sensed=R)
    doa_result: Doa = doa_ret[0]
    print("deep augmented classic music: ", np.rad2deg(doa_result.thetas))

    rmspe      = RMSPELoss()
    loss_array = rmspe.loss(doa, doa_result._thetas)
    loss_mean  = np.mean(loss_array)
    print(f"loss array {loss_array}")
    print(f"loss mean: {loss_mean}")


if __name__ == "__main__":
    doa()
