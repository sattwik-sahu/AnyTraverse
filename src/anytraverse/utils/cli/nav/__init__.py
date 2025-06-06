import typer
from anytraverse.utils.cli.nav.oakd import main as oakd_main


app = typer.Typer(name="navigate", help="Navigation with AnyTraverse")


app.command(name="oakd")(oakd_main)
