import SpaceNet as sn

import simple
import doa

from typing import Annotated

import typer
from rich import print


app = typer.Typer(
    no_args_is_help=True,
    help="This application is to drive Direction Of Arrival (DoA) or Delay Doppler (DD) simulations.",
)


app.add_typer(simple.app, name="simple")
app.add_typer(doa.app, name="doa")


def version_cb(value: bool):
    if value:
        typer.echo("music version: 0.0.1")
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
