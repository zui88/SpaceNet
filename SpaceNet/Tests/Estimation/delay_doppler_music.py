from SpaceNet.Builders.delay_doppler import create_delay_doppler_music
from SpaceNet.Engines.engine import RetDD, DelayDoppler, Engine
from SpaceNet.Synthesizer.synthesizer import DDSynthesizerWrapper
from SpaceNet.Configs.DelayDoppler.config import Config as DDConfig


def delay_doppler():
    config = DDConfig()
    config.base.snr_db = 40  # default 30

    ##################################################
    # generate the signal
    ##################################################
    taus = (3, 5)
    omegas = (1.3, -0.5)
    r_sensed = DDSynthesizerWrapper(config).generate(
        taus=taus,
        omegas=omegas,
    )
    print("true taus: ", taus)
    print("true omegas: ", omegas)

    ##################################################
    # music
    ##################################################
    config.base.inference_mode = False  # just to be explicit
    config.base.d_sources = len(taus)
    music_engine: Engine[RetDD] = create_delay_doppler_music(config, "normal")
    DDRet: RetDD = music_engine.estimate(r_sensed=r_sensed[None, :])
    dd: DelayDoppler = DDRet[0]
    print(
        "classic delay doppler music: [delay - {}], [doppler - {}]".format(
            dd.delay[0], dd.doppler[0]
        )
    )


if __name__ == "__main__":
    delay_doppler()
