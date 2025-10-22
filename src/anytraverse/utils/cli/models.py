import typer
from anytraverse.utils.models.clipseg import clipseg


app = typer.Typer(name="models", help="Run inference on a model")

app.command(name="clipseg")(clipseg)
