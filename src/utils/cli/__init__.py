import typer
from utils.cli.models import app as models_app
from utils.cli.eval import eval

app = typer.Typer(
    name="anytraverse", help="Vision-Language based segmentation for offroad navigation"
)

app.add_typer(models_app)
app.command(name="evaluate")(eval)
