from SpaceNet.Synthesizer.synthesizer import DoaSynthesizerWrapper
from SpaceNet.Builders.doa import create_classic_music
from SpaceNet.Configs.Doa.config import Config
from SpaceNet.Configs.base import SignalKind, ArrayKind

import numpy as np

import typer
from rich import print


app = typer.Typer(no_args_is_help=True)


@app.command()
def estimate(
    thetas: list[float],
):
    """
    Takes one list of ground truth that the engine will try to estimate.  The simple command uses the classical MUSIC algorithm for DoA estimation.  Configuration:
    - samples: 100
    - snr: 35 db
    - std ula array
    - random signal
    """
    config = Config()
    config.base.signal.kind = SignalKind.RANDOM_SIGNAL
    config.base.array.kind = ArrayKind.ULA_ARRAY
    config.base.signal.n_samples = 100

    print("thetas given: {}".format(thetas))
    ground_truth = np.deg2rad(thetas)

    signal_synthesizer = DoaSynthesizerWrapper(config)
    r = signal_synthesizer.generate(tuple(ground_truth))
    music_engine = create_classic_music(config)
    ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(np.rad2deg(ret[0].thetas)))


if __name__ == "__main__":
    app()
