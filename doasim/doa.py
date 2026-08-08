from SpaceNet.Synthesizer.synthesizer import DoaSynthesizerWrapper
from SpaceNet.Configs.base import SignalKind, ArrayKind
from SpaceNet.Configs.Doa.config import Config
from SpaceNet.Engines.engine import RetDoa, Engine, Doa
from SpaceNet.Builders.doa import create_classic_music, create_root_music

import numpy as np

from typing import Annotated

import typer
from rich import print


app = typer.Typer(no_args_is_help=True)


@app.callback()
def doa(
        ctx: typer.Context,
        estimator: Annotated[
            str,
            typer.Option(
                "--estimator",
                help="cm :: classic-music, rm :: root-music, dacm :: da-classic-music, darm :: da-root-music",
            ),
        ] = "cm",
):
    """
    Using the DoA estimators.
    """
    ctx.ensure_object(dict)
    ctx.obj["estimator"] = estimator


@app.command(context_settings={"allow_interspersed_args": False})
def estimation(
        ctx: typer.Context,
        thetas: list[float],
):
    """
    Takes one list of ground truth that the engine will try to estimate.
    """
    print("thetas given: {}".format(thetas))
    ground_truth = np.deg2rad(thetas)

    config = Config()
    config.base.signal.kind = SignalKind.RANDOM_SIGNAL
    config.base.array.kind = ArrayKind.ULA_ARRAY
    config.base.signal.n_samples = 100
            
    signal_synthesizer = DoaSynthesizerWrapper(config)
    r = signal_synthesizer.generate(tuple(ground_truth))

    match ctx.obj["estimator"]:
        case "cm":
            print("classic music")
            music_engine = create_classic_music(config)
            
        case "rm":
            print("root music")
            music_engine = create_root_music(config)

        case "dacm":
            print("deep augmented classic music")
        case "darm":
            print("deep augmented root music")
        case _:
            print(f"estimator [red]{ctx.obj["estimator"]}[/red] not supported")
            typer.Exit

    ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(np.rad2deg(ret[0].thetas)))


@app.command()
def simulation():
    """
    Takes one list of ground truth that the engine will try to estimate.
    """
    print("not implemented yet")

@app.command()
def benchmark():
    """
    Takes one list of ground truth that the engine will try to estimate.
    """
    print("not implemented yet")

@app.command()
def plot():
    """
    Takes one list of ground truth that the engine will try to estimate.
    """
    print("not implemented yet")


if __name__ == "__main__":
    app()
