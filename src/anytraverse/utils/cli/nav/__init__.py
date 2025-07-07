import typer
from anytraverse.utils.cli.nav.oakd import main as oakd_main
import asyncio


app = typer.Typer(name="navigate", help="Navigation with AnyTraverse")


app.command(name="oakd", help="Run Unitree Go 1 with OAK-D Camera")(oakd_main)
