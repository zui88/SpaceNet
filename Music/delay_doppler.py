from SpaceNet.Builders.delay_doppler import (
    create_delay_doppler_music as create_classic_music,
    create_deep_doppler_music as create_deep_augmented_classic_engine,
)
from SpaceNet.Engines.engine import RetDD, DelayDoppler, Engine
from SpaceNet.Synthesizer.synthesizer import DDSynthesizerWrapper
from SpaceNet.Configs.DelayDoppler.config import Config
from SpaceNet.Configs.base import Signal, Array
from SpaceNet.Utils.DeepAugmented.LossFunctions.rmse_loss import RMSELoss
from SpaceNet.Utils.DeepAugmented.TrainingData.delay_doppler import generate_data_set

from Utils.exit import exit_application

import matplotlib.pyplot as plt
import numpy as np

from typing import Annotated, Any
import os
import sys
from pathlib import Path
from threading import Thread, Lock
from time import time

import typer
from rich import print


def get_music_engine(ctx_obj: dict[str, Any]) -> Engine[RetDD]:
    verbose: bool = False
    if "verbose" in ctx_obj:
        verbose = ctx_obj["verbose"]

    music_engine: Engine[RetDD] = None
    config: Config = Config()

    config.base.inference_mode = False  # just to be explicit
    config.base.d_sources = 2
    config.base.signal = Signal(fs=25)
    config.base.array = Array(antennas=8)

    match ctx_obj["estimator"]:
        case "cm":
            if verbose:
                print("classic music")
            alter_config_from_options(ctx_obj, config, False)
            music_engine = create_classic_music(config, "normal")

        case "cmf":
            if verbose:
                print("classic music fast algorithm")
            alter_config_from_options(ctx_obj, config, False)
            music_engine = create_classic_music(config, "fast")

        case "dacm":
            if verbose:
                print("deep augmented classic music")
            app_dir = Path(os.getcwd())
            music_engine = create_deep_augmented_classic_engine(
                config_dir=Path(app_dir / "Music" / "dd-cl-mu")
            )
            config = music_engine.configs

        case "rnd":
            if verbose:
                print("random device")
            config = alter_config_from_options(ctx_obj, config, False)

            class RandomEngine:
                rng = np.random.default_rng(abs(hash(str(time()))))

                def estimate(self, r_sensed):
                    batch_size, _, _ = r_sensed.shape

                    class Ret:
                        pass

                    ret = Ret()
                    ret.delay = self.rng.uniform(
                        0.5, 8, size=(batch_size, config.base.d_sources)
                    )
                    return (ret,)

            music_engine = RandomEngine()

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
def dd(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Display verbosely",
        ),
    ] = False,
    snr: Annotated[
        float,
        typer.Option(
            "--snr",
            help="signal-to-noise-ration",
        ),
    ] = 35,
):
    """
    Using the delay doppler estimators.
    """
    ctx.ensure_object(dict)

    ctx.obj["verbose"] = verbose
    ctx.obj["snr"] = snr


@app.command(context_settings={"allow_interspersed_args": False})
def estimation(
    ctx: typer.Context,
    delays: list[float],
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
):
    """
    Takes one list of ground truth that the engine will try to estimate.
    """

    ctx.obj["estimator"] = estimator
    ctx.obj["snr"] = snr

    print("delays given: {}".format(delays))
    ground_truth = delays

    music_engine: Engine[RetDD] = get_music_engine(ctx.obj)
    config: Config = music_engine.configs

    signal_synthesizer = DDSynthesizerWrapper(config)
    r = signal_synthesizer.generate(
        taus=ground_truth,
        omegas=(0.5, -0.5),
    )

    dd_ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(dd_ret[0].delay))


def run_simulation(
    ctx_obj: dict[str, Any],
    delay_range,
    delay_space,
    experiments,
    music_engine: Engine[RetDD] | None = None,
) -> (RetDD, np.ndarray):
    """run one single Monte-Carlo simulation

    RETURN
    ------
        dd_ret: RetDD
            the engine return object

        dd: np.ndarray
            ground truth
    """
    if music_engine is None:
        music_engine = get_music_engine(ctx_obj)

    config: Config = music_engine.configs

    correlation_coefficient = None
    if "correlation" in ctx_obj:
        correlation_coefficient = ctx_obj["correlation"]

    r, dd = generate_data_set(
        signal_generator=config.base.signal_provider,
        array_geometry=config.base.array_geometry,  # todo evaluate
        delay_range=delay_range,
        doppler_range=(-0.5, 0.5),
        min_delay_separation=delay_space,
        training_examples=experiments,
        max_signal_sources=config.base.d_sources,
        snr_db=config.base.snr_db,
        correlation_coefficient=correlation_coefficient,
        observ_ctx=config.observ_ctx,
    )
    dd_ret: RetDD = music_engine.estimate(r_sensed=r)
    return dd_ret, dd


@app.command()
def benchmark(
    ctx: typer.Context,
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            "-m",
            help="signal-to-noise-ration: snr, correlation coefficient: cor, source distance: sd",
        ),
    ] = "snr",
    grid_range: Annotated[
        tuple[float, float],
        typer.Option(
            "--grid-range",
            "-r",
            help="The minimum and maximum value that the range a scanned through.",
        ),
    ] = (-5, 35),
    grid_space: Annotated[
        float,
        typer.Option(
            "--grid-space",
            "-s",
            help="Spacing between adjacent points in the Monte-Carlo evaluation grid.",
        ),
    ] = 5,
    estimators: Annotated[
        list[str],
        typer.Option(
            "--estimators",
            "-e",
            help="Estimators to evaluate: rnd, cm, dacm.",
        ),
    ] = [
        "rnd",
        "cm",
        "dacm",
    ],
    plot: Annotated[
        bool,
        typer.Option(
            "--plot",
            "-p",
            help="plot the result with pyplot",
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
    delay_range: Annotated[
        tuple[float, float],
        typer.Option(
            "--delay",
            help="Time range where impinging signals are emulated.",
        ),
    ] = (0.6, 8.0),
    delay_space: Annotated[
        float,
        typer.Option(
            "--space",
            help="Minimum time space between impinging signals in seconds.",
        ),
    ] = 1,
):
    """Runs multible simulations of different estimators.  Each
    simulation will be plotted.

    """
    Estimator = str

    GridPoint = float
    DDGroundTruth = np.ndarray
    simulation_results: dict[
        Estimator, list[tuple[GridPoint, tuple[RetDD, DDGroundTruth]]]
    ] = {}
    simulation_mutex: Lock = Lock()
    simulation_threads: list[Thread] = []

    for estimator in estimators:
        tmp_ctx_obj = ctx.obj.copy()
        tmp_ctx_obj["estimator"] = estimator

        def simulate(
            ctx_obj,
            estimator,
            experiments,
            delay_range,
            delay_space,
        ):
            start, stop = grid_range
            step = grid_space
            grid = np.arange(start, stop, step)
            if grid[-1] != stop:
                grid = np.append(grid, stop)

            music_engine: Engine[RetDD] = get_music_engine(ctx_obj)
            configs = music_engine.configs

            delay_doppler_ret: RetDD
            dd_ground_truth: np.ndarray
            metric_data: list[tuple[GridPoint, tuple[RetDD, DDGroundTruth]]] = []
            verbose = ctx_obj["verbose"]
            for x in grid:
                match metric:
                    case "snr":
                        if verbose:
                            print("signal-to-noise-ration")
                        configs.base.snr_db = float(x)
                    case "cor":
                        if verbose:
                            print("correlation coefficient")
                        ctx_obj["correlation"] = x
                    case "sd":
                        if verbose:
                            print("source distance")
                        T_point = 3
                        if T_point < x:
                            delay_range = (T_point, x)
                            delay_space = x - T_point
                        else:
                            delay_range = (x, T_point)
                            delay_space = T_point - x
                    case _:
                        configs.base.snr_db = float(x)

                delay_doppler_ret, dd_ground_truth = run_simulation(
                    ctx_obj,
                    delay_range,
                    delay_space,
                    experiments,
                    music_engine,
                )
                metric_data.append((x, (delay_doppler_ret, dd_ground_truth)))

            with simulation_mutex:
                simulation_results[estimator] = metric_data

        simulation_threads.append(
            Thread(
                target=simulate,
                args=(tmp_ctx_obj, estimator, experiments, delay_range, delay_space),
            )
        )

    for t in simulation_threads:
        t.start()

    for t in simulation_threads:
        t.join()

    fig, ax = plt.subplots()
    loss: Loss = RMSELoss(d_source=2)
    for estimator, metric_data in simulation_results.items():
        xs = []
        losses = []

        if ctx.obj["verbose"]:
            print(f"{estimator}")
        for x, (delay_doppler_ret, dd_ground_truth) in metric_data:
            dd_result: DelayDoppler = delay_doppler_ret[0]
            loss_array = loss.loss(dd_ground_truth, dd_result.delay)
            loss_mean = np.mean(loss_array)

            xs.append(x)
            losses.append(loss_mean)

            if ctx.obj["verbose"]:
                print(f"\tgrid: {x} -- loss mean: {loss_mean}")

        if plot:
            ax.plot(xs, losses, marker="o", label=estimator)

    if plot:
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_ylabel("RMSE [sec]")
        match metric:
            case "snr":
                ax.set_xlabel("SNR [dB]")
            case "cor":
                ax.set_xlabel(r"$\sigma^2$")
            case "sd":
                ax.axvline(
                    3,
                    color="r",
                    linestyle="-",
                    label="sig2",
                )
                ax.set_xlabel(r"sig1 $(\Delta\tau)$[sec]")
            case _:
                ax.set_xlabel("SNR [dB]")
        ax.grid(True, which="both")
        ax.legend()
        plt.show()
