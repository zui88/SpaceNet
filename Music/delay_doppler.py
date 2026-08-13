from SpaceNet.Builders.delay_doppler import (
    create_delay_doppler_music as create_classic_music,
    create_deep_doppler_music as create_deep_augmented_classic_engine,
)
from SpaceNet.Engines.engine import RetDD, DelayDoppler, Engine
from SpaceNet.Synthesizer.synthesizer import DDSynthesizerWrapper
from SpaceNet.Configs.DelayDoppler.config import Config

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


def get_music_engine(ctx_obj: dict[str, Any]) -> Engine[RetDD]:
    verbose: bool = False
    if "verbose" in ctx_obj:
        verbose = ctx_obj["verbose"]

    music_engine: Engine[RetDD] = None
    config: Config = Config()

    config.base.inference_mode = False  # just to be explicit
    config.base.d_sources = 2

    match ctx_obj["estimator"]:
        case "cm":
            if verbose:
                print("classic music")
            alter_config_from_options(ctx_obj, config, False)
            music_engine = create_classic_music(config, "normal")

        case "cmf":
            if verbose:
                print("classic music fast algorithm")
            alter_config_from_options(ctx_obj, config)
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
            config = alter_config_fnrom_options(ctx_obj, config, False)

            class RandomEngine:
                rng = np.random.default_rng()

                def estimate(self, r_sensed):
                    batch_size, _, _ = r_sensed.shape

                    class Ret:
                        pass

                    ret = Ret()
                    ret._thetas = rng.uniform(
                        0.5, 16, size=(batch_size, config.base.d_sources)
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
    r = signal_synthesizer.generate(tuple(ground_truth))

    dd_ret = music_engine.estimate(r_sensed=r[None, :])
    print("estimated values: {}".format(dd_ret[0].delay))


def run_simulation(
    ctx_obj: dict[str, Any],
    delay_range,
    ddelay_space,
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
        min_delay_separation=ddelay_space,
        training_examples=experiments,
        max_signal_sources=config.base.d_sources,
        snr_db=config.base.snr_db,
        correlation_coefficient=correlation_coefficient,
    )

    dd_ret: RetDD = music_engine.estimate(r_sensed=r)
    return dd_ret, dd
