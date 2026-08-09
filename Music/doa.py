from SpaceNet.Utils.DeepAugmented.TrainingData.doa import generate_data_set
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
    verbose: bool = False
    if "verbose" in ctx.obj:
        verbose = ctx.obj["verbose"]

    music_engine: Engine[RetDoa] = None
    config: Config = Config()

    match ctx.obj["estimator"]:
        case "cm":
            if verbose:
                print("classic music")
            config = alter_config_from_app_options(ctx, config)
            config.base.signal.kind = SignalKind.RANDOM_SIGNAL
            config.base.array.kind = ArrayKind.ULA_ARRAY
            config.base.signal.n_samples = 100
            config.base.d_sources = 4  # MAGIC NUMBER :(
            music_engine = create_classic_music(config)

        case "rm":
            if verbose:
                print("root music")
            config = alter_config_from_app_options(ctx, config)
            config.base.signal.kind = SignalKind.RANDOM_SIGNAL
            config.base.array.kind = ArrayKind.ULA_ARRAY
            config.base.signal.n_samples = 100
            config.base.d_sources = 4
            music_engine = create_root_music(config)

        case "dacm":
            if verbose:
                print("deep augmented classic music")
            app_dir = Path(os.getcwd())
            if "da_selector" in ctx.obj and ctx.obj["da_selector"]:
                os.chdir(app_dir / "Music" / "da-cl-mu-da-selector")
                music_engine = create_deep_classic_music(kind="sl")
            else:
                os.chdir(app_dir / "Music" / "da-cl-mu")
                music_engine = create_deep_classic_music()
            config = music_engine.configs

        case "darm":
            if verbose:
                print("deep augmented root music")
            app_dir = Path(os.getcwd())
            os.chdir(app_dir / "Music" / "da-rm-mu")
            music_engine: Engine[RetDoa] = create_deep_root_music()
            config = music_engine.configs

        case _:
            if verbose:
                print(f"estimator [red]{ctx.obj['estimator']}[/red] not supported")
            sys.exit("close application")

    config = alter_config_from_app_options(ctx, config)
    music_engine.configs = config

    return music_engine


def alter_config_from_app_options(ctx: typer.Context, config: Config) -> Config:
    verbose = False
    if "verbose" in ctx.obj:
        verbose = ctx.obj["verbose"]

    if "snr" in ctx.obj:
        snr = ctx.obj["snr"]
        config.base.snr_db = snr
        if verbose:
            print(f"set snr [green]{snr}[/green]")

    return config


app = typer.Typer(no_args_is_help=True)


@app.callback()
def doa(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Display verbosely",
        ),
    ] = False,
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
            help="Number of impinging signals",
        ),
    ] = None,
):
    """
    Using the DoA estimators.
    """
    ctx.ensure_object(dict)

    ctx.obj["estimator"] = estimator
    ctx.obj["verbose"] = verbose

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
                os.chdir(app_dir / "Music" / "da-cl-mu-da-selector")
                music_engine: Engine[RetDoa] = create_deep_classic_music(kind="sl")
            else:
                os.chdir(app_dir / "Music" / "da-cl-mu")
                music_engine: Engine[RetDoa] = create_deep_classic_music()
            config: Config = music_engine.configs

            if len(ground_truth) != config.base.d_sources:
                exit_application(
                    f"[red]thetas len must be {config.base.d_sources}[/red]"
                )

            config = alter_config_from_app_options(ctx, config)

        case "darm":
            print("deep augmented root music")
            app_dir = Path(os.getcwd())
            os.chdir(app_dir / "Music" / "da-rm-mu")
            music_engine: Engine[RetDoa] = create_deep_root_music()
            config: Config = music_engine.configs
            if len(ground_truth) != config.base.d_sources:
                exit_application(
                    f"[red]thetas len must be {config.base.d_sources}[/red]"
                )

        case _:
            print(f"estimator [red]{ctx.obj['estimator']}[/red] not supported")
            sys.exit("close application")

    signal_synthesizer = DoaSynthesizerWrapper(config)
    r = signal_synthesizer.generate(tuple(ground_truth))

    doa_ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(np.rad2deg(doa_ret[0].thetas)))


def run_simulation(
    ctx: typer.Context,
    deg_range,
    deg_space,
    experiments,
) -> (RetDoa, np.ndarray):
    """run one single simulation

    RETURN
    ------
        doa_ret: RetDoa
            the engine return object

        doa: np.ndarray
            ground truth
    """
    music_engine: Engine[RetDoa] = get_music_engine(ctx)
    config: Config = music_engine.configs

    r, doa = generate_data_set(
        signal_generator=config.base.signal_provider,
        array_geometry=config.base.array_geometry,
        deg_range=deg_range,
        min_spacing=deg_space,
        samples=experiments,
        max_signal_sources=config.base.d_sources,
        snr_db=config.base.snr_db,
    )

    doa_ret: RetDoa = music_engine.estimate(r_sensed=r)
    return doa_ret, doa


@app.command()
def simulation(
    ctx: typer.Context,
    da_selector: Annotated[
        bool,
        typer.Option(
            help="Will be parsed just in DA classical MUSIC estimator.  Either using deep augmented noise-signal selector or classical formulation.",
        ),
    ] = False,
    experiments: Annotated[
        int,
        typer.Option(
            "--experiments",
            "-e",
            help="The number of experiments that is going to drive the experiment.",
        ),
    ] = 100,
    deg_range: Annotated[
        tuple[float, float],
        typer.Option(
            "--range",
            "-r",
            help="Minimum and maximum angle of arrival in degrees.",
        ),
    ] = (-70.0, 70.0),
    deg_space: Annotated[
        float,
        typer.Option(
            "--space",
            "-s",
            help="Minimum space between impinging signals in degrees.",
        ),
    ] = 5,  # 15 grad guy
):
    """
    Run one Monte-Carlo experiment.
    """
    ctx.obj["da_selector"] = da_selector

    def check_user_options(
        experiments: int,
        deg_range: tuple[float, float],
        deg_space: float,
    ) -> None:
        MAX_SAMPLES = 100_000
        if experiments < 1 or experiments > MAX_SAMPLES:
            exit_application(
                f"Amount of experiments not supported [red]{experiments}[/red]"
            )

        if deg_range[0] > deg_range[1] or deg_range[0] < -90 or deg_range[1] > 90:
            exit_application(
                f"deg-range out of range min: [red]{deg_range[0]}[/red], max: [red]{deg_range[1]}[/red]"
            )

        if deg_space >= 180:
            exit_application(f"deg-space too high [red]{deg_space}[/red]")

    check_user_options(
        experiments,
        deg_range,
        deg_space,
    )

    doa_ret: RetDoa
    doa: np.ndarray
    doa_ret, doa = run_simulation(
        ctx,
        deg_range,
        deg_space,
        experiments,
    )
    doa_result: Doa = doa_ret[0]
    loss: Loss = RMSPELoss()
    loss_array = loss.loss(doa, doa_result._thetas)
    loss_mean = np.mean(loss_array)
    print(f"loss mean: {loss_mean}")


@app.command()
def benchmark():
    """Runs multible simulations of different estimators.  Each
    simulation will be plotted.

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
