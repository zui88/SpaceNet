from SpaceNet.Synthesizer.synthesizer import DOASignalSynthesizer
from SpaceNet.Synthesizer.signal import RandomSignal
from SpaceNet.Synthesizer.geometry import ULAArray
from SpaceNet.Configs.Doa.config import Config as DoaConfig
from SpaceNet.Engines.engine import Engine, RetDoa, Doa
from SpaceNet.Builders.doa import create_classic_music

import numpy as np


def doa():
    config = DoaConfig()
    ##################################################
    # generate the signal
    ##################################################
    thetas = np.deg2rad([-30, 10, 75, -68])

    corr_coef_12 = 0.31
    corr_coef_13 = 0.92
    corr_coef_14 = 0.01
    corr_coef_23 = 0.01
    corr_coef_24 = 0.51
    corr_coef_34 = 0.01

    array_geometry = ULAArray(antennas=config.base.array.antennas)
    r_sensed = DOASignalSynthesizer(
        array_geometry,
        RandomSignal(n_samples=config.base.signal.n_samples),
        snr_db=config.base.snr_db,
    ).generate(
        thetas,
        correlation_matrix=np.array(
            [
                [1.0, corr_coef_12, corr_coef_13, corr_coef_14],
                [corr_coef_12, 1.0, corr_coef_23, corr_coef_24],
                [corr_coef_13, corr_coef_23, 1.0, corr_coef_34],
                [corr_coef_14, corr_coef_24, corr_coef_34, 1.0],
            ]
        ),
    )
    print("true thetas: ", np.rad2deg(thetas))

    ##################################################
    # music
    ##################################################
    music_engine: Engine[RetDoa] = create_classic_music(config)
    DoaRet: RetDoa = music_engine.estimate(r_sensed=r_sensed[None, :])
    doa: Doa = DoaRet[0]
    print("classic music: ", np.rad2deg(doa.thetas[0]))


if __name__ == "__main__":
    doa()
