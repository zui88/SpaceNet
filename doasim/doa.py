from SpaceNet.Synthesizer.synthesizer import DoaSynthesizerWrapper
from SpaceNet.Configs.base import SignalKind, ArrayKind
from SpaceNet.Configs.Doa.config import Config
from SpaceNet.Engines.engine import RetDoa, Engine, Doa
from SpaceNet.Builders.doa import (
    create_classic_music,
    create_root_music,
    create_deep_classic_music,
    create_deep_root_music,
)
from SpaceNet.Utils.DeepAugmented.LossFunctions.rmspe_loss import RMSPELoss

from Utils.exit import exit_application

import numpy as np

from typing import Annotated
import os
import sys
from pathlib import Path

import typer
from rich import print


def get_music_engine(ctx: typer.Context) -> Engine[RetDoa]:
    pass


def get_config(engine: Engine[RetDoa]) -> Config:
    pass


def alter_config_from_app_options(ctx: typer.Context, config: Config) -> Config:
    if "snr" in ctx.obj:
        snr = ctx.obj["snr"]
        config.base.snr_db = snr
        print(f"set snr [green]{snr}[/green]")

    return config


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
    snr: Annotated[
        float | None,
        typer.Option(
            "--snr",
            help="signal-to-noise-ratio",
        ),
    ] = None,
    d_signals: Annotated[
        int | None,
        typer.Option(
            "--d-signals",
            help="number of impinging signals",
        ),
    ] = None,
):
    """
    Using the DoA estimators.
    """
    ctx.ensure_object(dict)

    ctx.obj["estimator"] = estimator

    if snr is not None:
        ctx.obj["snr"] = snr

    if d_signals is not None:
        ctx.obj["d_signals"] = d_signals


@app.command(context_settings={"allow_interspersed_args": False})
def estimation(
    ctx: typer.Context,
    thetas: list[float],
    da_selector: Annotated[
        bool,
        typer.Option(
            "--da-selector",
            help="Will be parsed just in DA classical MUSIC estimator.  Either using deep augmented noise-signal selector or classical formulation.",
        ),
    ] = False,
):
    """
    Takes one list of ground truth that the engine will try to estimate.
    """
    print("thetas given: {}".format(thetas))
    ground_truth = np.deg2rad(thetas)

    match ctx.obj["estimator"]:
        case "cm":
            print("classic music")
            config = Config()
            config = alter_config_from_app_options(ctx, config)
            config.base.signal.kind = SignalKind.RANDOM_SIGNAL
            config.base.array.kind = ArrayKind.ULA_ARRAY
            config.base.signal.n_samples = 100
            music_engine: Engine[RetDoa] = create_classic_music(config)

        case "rm":
            print("root music")
            config = Config()
            config = alter_config_from_app_options(ctx, config)
            config.base.signal.kind = SignalKind.RANDOM_SIGNAL
            config.base.array.kind = ArrayKind.ULA_ARRAY
            config.base.signal.n_samples = 100
            music_engine: Engine[RetDoa] = create_root_music(config)

        case "dacm":
            print("deep augmented classic music")
            app_dir = Path(os.getcwd())
            if da_selector:
                os.chdir(app_dir / "doasim" / "da-cl-mu-da-selector")
                music_engine: Engine[RetDoa] = create_deep_classic_music(kind="sl")
            else:
                os.chdir(app_dir / "doasim" / "da-cl-mu")
                music_engine: Engine[RetDoa] = create_deep_classic_music()
            config: Config = music_engine.configs

            if len(ground_truth) != config.base.d_sources:
                exit_application(f"[red]thetas len must be {config.base.d_sources}[/red]")

            config = alter_config_from_app_options(ctx, config)

        case "darm":
            print("deep augmented root music")
            app_dir = Path(os.getcwd())
            os.chdir(app_dir / "doasim" / "da-rm-mu")
            music_engine: Engine[RetDoa] = create_deep_root_music()
            config: Config = music_engine.configs
            if len(ground_truth) != config.base.d_sources:
                exit_application(f"[red]thetas len must be {config.base.d_sources}[/red]")

        case _:
            print(f"estimator [red]{ctx.obj['estimator']}[/red] not supported")
            sys.exit("close application")

    signal_synthesizer = DoaSynthesizerWrapper(config)
    r = signal_synthesizer.generate(tuple(ground_truth))

    ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(np.rad2deg(ret[0].thetas)))


@app.command()
def simulation(
    ctx: typer.Context,
    thetas: list[float],
    da_selector: Annotated[
        bool,
        typer.Option(
            help="Will be parsed just in DA classical MUSIC estimator.  Either using deep augmented noise-signal selector or classical formulation.",
        ),
    ] = False,
    samples: Annotated[
        int,
        typer.Option(
            "--samples",
            "-s",
            help="The number of samples that is going to drive the experiment.",
        ),
    ] = 5000,
    deg_range: Annotated[
        tuple[float],
        typer.Option(
            "--deg-range",
            "-r",
            help="Minimum and maximum angle of arrival in degrees.",
        ),
    ] = (-70.0, 70.0),
):
    """
    Runs one Monte-Carlo experiment.
    """
    def check_user_options(
            samples,
            deg_range: tuple[float],
            ) -> None:
        MAX_SAMPLES = 100_000
        if samples < 1 or samples > MAX_SAMPLES:
            exit_application(f"Amount of samples not supported [red]{samples}[/red]")

        if len(deg_range) != 2:
            exit_application(f"deg-range must be a tuple of (min, max)")
        if deg_range[0] < -90 or deg_range[1] > 90:
            exit_application(f"deg-range out of range [red]{deg_range[0]}[/red], [red]{deg_range[1]}[/red]")


    check_user_options(
        samples,
        deg_range,
        )

    music_engine: Engine[RetDoa] = get_music_engine(ctx)
    config: Config = get_config(music_engine)
    config = alter_config_from_app_options(ctx, config)
    
    r, doa = generate_data_set(
        signal_generator=config.base.signal_provider,
        array_geometry=config.base.array_geometry,
        deg_range=deg_range,
        min_spacing=5,  # 15 grad guy
        samples=samples,
        max_signal_sources=config.base.d_sources,
        snr_db=config.base.snr_db,
    )

    ret: RetDoa = music_engine.estimate(r_sensed=r)
    doa_result: Doa = doa_ret[0]
    loss_function: Loss = RMSPELoss()
    loss_array = loss_function.compute_error(doa, doa_result._thetas)
    loss_mean = np.mean(loss_array)
    print(f"loss mean: {loss_mean}")


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
