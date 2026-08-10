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

import matplotlib.pyplot as plt
import numpy as np

from typing import Annotated, Any
import os
import sys
from pathlib import Path
from threading import Thread, Lock

import typer
from rich import print


def get_music_engine(ctx_obj: dict[str, Any]) -> Engine[RetDoa]:
    verbose: bool = False
    if "verbose" in ctx_obj:
        verbose = ctx_obj["verbose"]

    music_engine: Engine[RetDoa] = None
    config: Config = Config()

    match ctx_obj["estimator"]:
        case "cm":
            if verbose:
                print("classic music")
            config = alter_config_from_options(ctx_obj, config, False)
            config.base.signal.kind = SignalKind.RANDOM_SIGNAL
            config.base.array.kind = ArrayKind.ULA_ARRAY
            config.base.signal.n_samples = 100
            config.base.d_sources = 4  # MAGIC NUMBER :(
            music_engine = create_classic_music(config)

        case "rm":
            if verbose:
                print("root music")
            config = alter_config_from_options(ctx_obj, config, False)
            config.base.signal.kind = SignalKind.RANDOM_SIGNAL
            config.base.array.kind = ArrayKind.ULA_ARRAY
            config.base.signal.n_samples = 100
            config.base.d_sources = 4
            music_engine = create_root_music(config)

        case "dacm":
            if verbose:
                print("deep augmented classic music")
            app_dir = Path(os.getcwd())
            if "da_selector" in ctx_obj and ctx_obj["da_selector"]:
                music_engine = create_deep_classic_music(
                    kind="sl",
                    config_dir=Path(app_dir / "Music" / "da-cl-mu-da-selector"),
                )
            else:
                music_engine = create_deep_classic_music(
                    config_dir=Path(app_dir / "Music" / "da-cl-mu")
                )
            config = music_engine.configs

        case "darm":
            if verbose:
                print("deep augmented root music")
            app_dir = Path(os.getcwd())
            music_engine: Engine[RetDoa] = create_deep_root_music(
                config_dir=Path(app_dir / "Music" / "da-rm-mu")
            )
            config = music_engine.configs

        case _:
            if verbose:
                print(f"estimator [red]{ctx_obj['estimator']}[/red] not supported")
            sys.exit("close application")

    config = alter_config_from_options(ctx_obj, config)
    music_engine.configs = config

    return music_engine


def alter_config_from_options(
    ctx_obj: dict[str, Any], config: Config, printable: bool = True
) -> Config:
    verbose = False
    if "verbose" in ctx_obj:
        verbose = ctx_obj["verbose"] and printable

    if "snr" in ctx_obj:
        snr = ctx_obj["snr"]
        config.base.snr_db = snr
        if verbose:
            print(f"set snr [green]{snr}[/green]")

    if "d_sources" in ctx_obj:
        d_sources = ctx_obj["d_sources"]
        config.base.d_sources = d_sources
        if verbose:
            print(f"set d-sources [green]{d_sources}[/green]")

    if "inference" in ctx_obj:
        inference = ctx_obj["inference"]
        config.base.inference_mode = inference
        if verbose:
            print(f"set inference_mode [green]{inference}[/green]")

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
):
    """
    Using the DoA estimators.
    """
    ctx.ensure_object(dict)

    ctx.obj["verbose"] = verbose


@app.command(context_settings={"allow_interspersed_args": False})
def estimation(
    ctx: typer.Context,
    thetas: list[float],
    estimator: Annotated[
        str,
        typer.Option(
            "--estimator",
            "-e",
            help="cm :: classic-music, rm :: root-music, dacm :: da-classic-music, darm :: da-root-music",
        ),
    ] = "cm",
    snr: Annotated[
        float | None,
        typer.Option(
            "--snr",
            help="signal-to-noise-ratio",
        ),
    ] = 35,
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

    ctx.obj["estimator"] = estimator
    ctx.obj["snr"] = snr

    print("thetas given: {}".format(thetas))
    ground_truth = np.deg2rad(thetas)

    music_engine: Engine[RetDoa] = get_music_engine(ctx.obj)
    config: Config = music_engine.configs

    signal_synthesizer = DoaSynthesizerWrapper(config)
    r = signal_synthesizer.generate(tuple(ground_truth))

    doa_ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(np.rad2deg(doa_ret[0].thetas)))


def run_simulation(
    ctx_obj: dict[str, Any],
    deg_range,
    deg_space,
    experiments,
    music_engine: Engine[RetDoa] | None = None,
) -> (RetDoa, np.ndarray):
    """run one single simulation

    RETURN
    ------
        doa_ret: RetDoa
            the engine return object

        doa: np.ndarray
            ground truth
    """
    if music_engine is None:
        music_engine = get_music_engine(ctx_obj)

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


def check_user_options_simulation(
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


@app.command()
def simulation(
    ctx: typer.Context,
    estimator: Annotated[
        str,
        typer.Option(
            "--estimator",
            "-e",
            help="cm :: classic-music, rm :: root-music, dacm :: da-classic-music, darm :: da-root-music",
        ),
    ] = "cm",
    snr: Annotated[
        float | None,
        typer.Option(
            "--snr",
            help="signal-to-noise-ratio",
        ),
    ] = 35,
    d_signals: Annotated[
        int | None,
        typer.Option(
            "--signals",
            "-d",
            help="Number of impinging signals",
        ),
    ] = None,
    inference: Annotated[
        bool,
        typer.Option(
            help=".",
        ),
    ] = True,
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
            "-n",
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
    ctx.obj["estimator"] = estimator
    ctx.obj["d_signals"] = d_signals
    ctx.obj["snr"] = snr

    check_user_options_simulation(
        experiments,
        deg_range,
        deg_space,
    )

    doa_ret: RetDoa
    doa: np.ndarray
    doa_ret, doa_gt = run_simulation(
        ctx.obj,
        deg_range,
        deg_space,
        experiments,
    )
    doa_result: Doa = doa_ret[0]
    loss: Loss = RMSPELoss()
    loss_array = loss.loss(doa_gt, doa_result._thetas)
    loss_mean = np.mean(loss_array)
    print(f"loss mean: {loss_mean}")


def check_user_options_benchmark(
    metric: str,
    *args,
    **argsv,
):
    pass


@app.command()
def benchmark(
    ctx: typer.Context,
    grid_space: Annotated[
        float,
        typer.Option(
            "--grid-space",
            "-g",
            help="Spacing between adjacent points in the Monte-Carlo evaluation grid.",
        ),
    ] = 5,
    grid_range: Annotated[
        tuple[float, float],
        typer.Option(
            "--grid-range",
            "-r",
            help=".",
        ),
    ] = (-5, 35),
    estimators: Annotated[
        list[str],
        typer.Option(
            "--estimators",
            "-e",
            help="Estimators to evaluate: cm, rm, dacm, darm.",
        ),
    ] = [
        "cm",
        "rm",
        "dacm",
        "darm",
    ],
    metric: Annotated[
        str,
        typer.Option(
            help="signal-to-noise-ration: snr",
        ),
    ] = "snr",
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            "-p",
            help="plot the result with pyplot",
        ),
    ] = False,
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
            "-n",
            help="The number of experiments that is going to drive the experiment.",
        ),
    ] = 100,
    deg_range: Annotated[
        tuple[float, float],
        typer.Option(
            "--range",
            help="Minimum and maximum angle of arrival in degrees.",
        ),
    ] = (-70.0, 70.0),
    deg_space: Annotated[
        float,
        typer.Option(
            "--space",
            help="Minimum space between impinging signals in degrees.",
        ),
    ] = 5,  # 15 grad guy
):
    """Runs multible simulations of different estimators.  Each
    simulation will be plotted.

    """
    check_user_options_benchmark(
        metric,
        grid_space,
        estimators,
    )
    check_user_options_simulation(
        experiments,
        deg_range,
        deg_space,
    )

    Estimator = str
    GridPoint = float
    DoaGT = np.ndarray
    simulation_results: dict[
        Estimator, list[tuple[GridPoint, tuple[RetDoa, DoaGT]]]
    ] = {}
    simulation_mutex: Lock = Lock()
    simulation_threads: list[Thread] = []

    ctx.obj["d_sources"] = 4
    ctx.obj["inference"] = False

    for estimator in estimators:
        tmp_ctx_obj = ctx.obj.copy()
        tmp_ctx_obj["estimator"] = estimator

        def simulate(
            ctx_obj,
            estimator,
            experiments,
            deg_range,
            deg_space,
        ):
            start, stop = grid_range
            step = grid_space
            grid = np.arange(start, stop + 1, step)

            music_engine: Engine[RetDoa] = get_music_engine(ctx_obj)
            configs = music_engine.configs

            doa_ret: RetDoa
            doa_gt: np.ndarray
            metric_data: list[tuple[GridPoint, tuple[RetDoa, DoaGT]]] = []
            for x in grid:
                configs.base.snr_db = float(x)

                doa_ret, doa_gt = run_simulation(
                    ctx_obj,
                    deg_range,
                    deg_space,
                    experiments,
                    music_engine,
                )
                metric_data.append((x, (doa_ret, doa_gt)))

            with simulation_mutex:
                simulation_results[estimator] = metric_data

        simulation_threads.append(
            Thread(
                target=simulate,
                args=(tmp_ctx_obj, estimator, experiments, deg_range, deg_space),
            )
        )

    for t in simulation_threads:
        t.start()

    for t in simulation_threads:
        t.join()

    fig, ax = plt.subplots()
    loss: Loss = RMSPELoss()
    for estimator, metric_data in simulation_results.items():
        xs = []
        losses = []

        if ctx.obj["verbose"]:
            print(f"{estimator}")
        for x, (doa_ret, doa_gt) in metric_data:
            doa_result: Doa = doa_ret[0]
            loss_array = loss.loss(doa_gt, doa_result._thetas)
            loss_mean = np.mean(loss_array)

            xs.append(x)
            losses.append(loss_mean)

            if ctx.obj["verbose"]:
                print(f"\tgrid: {x} -- loss mean: {loss_mean}")

        if plot:
            ax.plot(xs, losses, marker="o", label=estimator)

    if plot:
        ax.set_yscale("log")
        ax.set_xlabel("Grid")
        ax.set_ylabel("RMSPE")
        ax.grid(True, which="both")
        ax.legend()
        plt.show()


@app.command()
def plot():
    """
    Takes one list of ground truth that the engine will try to estimate.
    """
    print("not implemented yet")


if __name__ == "__main__":
    app()
