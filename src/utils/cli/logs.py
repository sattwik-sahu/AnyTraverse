import subprocess
from pathlib import Path
from typing import Final, List

import typer
from rich.console import Console
from typing_extensions import Annotated

app = typer.Typer(name="logs", help="Manage AnyTraverse logs")
console = Console()


@app.command(name="sync", help="Syncs logs to another folder")
def sync(
    bkp_path: Annotated[
        Path,
        typer.Option(
            help="The folder to sync the logs to",
        ),
    ] = Path("/mnt/toshiba_hdd/logs/anytraverse"),
) -> None:
    src_path: Final[Path] = Path("data/logs/")
    command: List[str] = [
        "rsync",
        "-av",
        "--ignore-existing",
        f"{src_path.as_posix()}/",
        bkp_path.as_posix(),
    ]

    # Run the command and capture output
    subprocess.run(command, text=True)
