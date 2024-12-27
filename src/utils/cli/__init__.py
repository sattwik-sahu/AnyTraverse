import typer
from utils.cli.models import app as models_app
from utils.cli.eval import eval
from utils.cli.frames import app as frames_app

app = typer.Typer(
    name="anytraverse", help="Vision-Language based segmentation for offroad navigation"
)

app.add_typer(models_app)
app.add_typer(frames_app)
app.command(name="evaluate")(eval)
