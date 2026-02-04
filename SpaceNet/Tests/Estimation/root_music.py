from generator import DOASignalSynthesizer, RandomSignal, ULAArray
from SpaceNet.Configs.Doa.config import Config as DoaConfig
from SpaceNet.Engines.engine import RetDoa, Engine, Doa
from SpaceNet.Builders.doa import create_root_music

import numpy as np


def doa():
    ##################################################
    # generate the signal
    ##################################################
    thetas = np.deg2rad([-30, 10, 25, 58])

    array_geometry = ULAArray(antennas=10)
    r_sensed = DOASignalSynthesizer(
        array_geometry,
        RandomSignal(n_samples=50),
        snr_db=30,
    ).generate(
        thetas,
    )
    print("true thetas: ", np.rad2deg(thetas))

    ##################################################
    # root music engine
    ##################################################
    config = DoaConfig()

    root_engine: Engine[RetDoa] = create_root_music(config)
    doa_ret: RetDoa = root_engine.estimate(r_sensed=r_sensed[None, :])
    doa_result: Doa = doa_ret[0]
    print("root music: ", np.rad2deg(doa_result.thetas[0]))


if __name__ == "__main__":
    doa()
