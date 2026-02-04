from generator import DOASignalSynthesizer, RandomSignal, ULAArray
from SpaceNet.Configs.Doa.config import Config as DoaConfig
from SpaceNet.Engines.engine import Engine, RetDoa, Doa
from SpaceNet.Builders.doa import create_classic_music

import numpy as np


def doa():
    config = DoaConfig()
    ##################################################
    # generate the signal
    ##################################################
    thetas = np.deg2rad([-30, 10, 25, 58])

    array_geometry = ULAArray(antennas=config.base.array.antennas)
    r_sensed       = DOASignalSynthesizer(
        array_geometry,
        RandomSignal(n_samples=config.base.signal.n_samples),
        snr_db=config.base.snr_db,
    ).generate(
        thetas,
    )
    print("true thetas: ", np.rad2deg(thetas))

    ##################################################
    # music
    ##################################################
    music_engine: Engine[RetDoa] = create_classic_music(config)
    DoaRet: RetDoa               = music_engine.estimate(r_sensed=r_sensed[None, :])
    doa: Doa                     = DoaRet[0]
    print("classic music: ", np.rad2deg(doa.thetas[0]))

    
if __name__ == "__main__":
    doa()
