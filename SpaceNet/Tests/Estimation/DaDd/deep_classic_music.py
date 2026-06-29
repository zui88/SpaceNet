import numpy as np

from SpaceNet.Builders.delay_doppler import create_deep_doppler_music as create_engine
from SpaceNet.Synthesizer.synthesizer import DDSynthesizerWrapper
from Engines.engine import Engine, RetDD, DelayDoppler


def estimate_delay():

    delay = (7.3, 3.6)

    correlation_coef = 0.01

    engine: Engine[RetDD] = create_engine()
    config = engine.configs

    r = DDSynthesizerWrapper(config).generate(
        delay,
        correlation_matrix=np.array([[1.0, correlation_coef], [correlation_coef, 1.0]]),
    )
    result: RetDD = engine.estimate(r_sensed=r[None, :])

    dd: DelayDoppler = result[0]
    print(f"true delays: {delay}")
    print(f"estimated delays: {dd.delay}")


if __name__ == "__main__":
    import os

    print(os.getcwd())
    estimate_delay()
