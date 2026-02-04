from SpaceNet.Utils.DeepAugmented.LossFunctions.rmspe_loss import RMSPELoss
from SpaceNet.Synthesizer.synthesizer import DOASignalSynthesizer
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
    thetas = np.deg2rad([63, -17, 4, -46])

    r_sensed = DOASignalSynthesizer(
        array_geometry=config.base.array_geometry,
        signal_generator=config.base.signal_provider,
        snr_db=config.base.snr_db,
    ).generate(
        thetas,
    )
    print("true thetas: ", np.rad2deg(thetas))

    ##################################################
    # deep augmented classic music engine
    ##################################################
    doa_ret: RetDoa = deep_music_engine.estimate(r_sensed=r_sensed[None, :])
    doa_result: Doa = doa_ret[0]
    print("deep augmented classic music: ", np.rad2deg(doa_result.thetas[0]))
    print("loss: ", RMSPELoss().call(thetas[None,:], doa_result.thetas).numpy())


if __name__ == "__main__":
    doa()
