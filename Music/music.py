import SpaceNet as sn

import delay_doppler
import doa

from typing import Annotated

import typer
from rich import print


app = typer.Typer(
    no_args_is_help=True,
    help="This application is to drive Direction Of Arrival (DoA) or Delay Doppler (DD) simulations.",
)


app.add_typer(doa.app, name="doa")
app.add_typer(delay_doppler.app, name="dd")


def version_cb(value: bool):
    if value:
        typer.echo("Music version: 0.4.3")
        typer.echo("SpaceNet version: {}".format(sn.__version__))
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the used SpaceNet version and exit.",
            callback=version_cb,
            is_eager=True,
        ),
    ] = False,
):
    pass


if __name__ == "__main__":
    app()
