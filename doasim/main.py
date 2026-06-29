import numpy as np

from SpaceNet.Synthesizer.synthesizer import DoaSynthesizerWrapper
from SpaceNet.Builders.doa import create_classic_music
from SpaceNet.Configs.Doa.config import Config
from SpaceNet.Configs.base import SignalKind, ArrayKind
import SpaceNet as sn

import re
from typing_extensions import Annotated

import typer


app = typer.Typer()


@app.command()
def music(
        thetas: Annotated[
            str,
            typer.Argument(help="separated contiguous string of angles in degrees"),
        ]
):
    print("SpaceNet Version: {}".format(sn.__version__))

    config = Config()
    config.base.signal.kind = SignalKind.RANDOM_SIGNAL
    config.base.array.kind = ArrayKind.ULA_ARRAY
    config.base.signal.n_samples = 100

    try:
        groud_truth_thetas = np.deg2rad([float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", thetas)])
    except TypeError:
        raise typer.BadParameter("doas must contain doas as floats")
    print("provided values: {}".format(np.rad2deg(groud_truth_thetas)))

    signal_synthesizer = DoaSynthesizerWrapper(config)
    r = signal_synthesizer.generate(tuple(groud_truth_thetas))
    music_engine = create_classic_music(config)
    ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(np.rad2deg(ret[0].thetas)))


@app.command()
def test():
    print("hello from doa estimator")


if __name__ == "__main__":
    app()
