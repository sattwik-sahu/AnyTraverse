import os
import shutil
import glob
from pathlib import Path
from typing import List
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress
from rich.text import Text

from uuid import uuid4 as uuid

app = typer.Typer(name="frames", help="Help with frames and videos from collected data")
console = Console()


def copy_files_with_glob(pattern: str, destination_dir: str) -> int:
    """
    Copies all files matching a given glob pattern to a specified directory.

    Parameters:
        pattern (str): The glob pattern to match files.
        destination_dir (str): The directory where files will be copied.

    Returns:
        int: Total number of files copied.
    """
    # Ensure the destination directory exists
    destination_path = Path(destination_dir)
    destination_path.mkdir(parents=True, exist_ok=True)

    # Find all files matching the glob pattern
    matching_files: List[str] = glob.glob(pattern)

    if not matching_files:
        console.print(
            Text(f"No files found for pattern: {pattern}", style="bold italic red")
        )
        return 0

    total_files = len(matching_files)

    # Copy files with a progress bar
    with Progress(console=console) as progress:
        task = progress.add_task("Copying files...", total=total_files)
        for file in matching_files:
            try:
                # Generate a unique filename if a file with the same name already exists
                file_name = Path(file).name
                destination_file = destination_path / file_name

                # Create a unique name for each file
                # timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                # unique_name = (
                #     f"{destination_file.stem}_{timestamp}{destination_file.suffix}"
                # )
                unique_name: str = f"frame__{uuid()}{destination_file.suffix}"
                destination_file = destination_path / unique_name

                shutil.copy(file, destination_file)
                progress.advance(task)
            except Exception as e:
                console.print(Text(f"Failed to copy {file}: {e}", style="italic red"))

    return total_files


@app.command(name="collect")
def collect(
    pattern: str = typer.Argument(
        ..., help="Glob pattern to match files (e.g., /path/to/files/*.txt)."
    ),
    destination: str = typer.Argument(
        ..., help="Destination directory where files will be copied."
    ),
):
    """
    Copy files matching a glob pattern to a destination directory.
    """
    console.print(Text("Starting file copy operation...", style="bold blue"))
    total_files = copy_files_with_glob(pattern, destination)

    if total_files > 0:
        console.print(
            f"\nSuccessfully copied [bold green]{total_files}[/bold green] file(s) to [bold]{destination}[/bold].",
        )
    else:
        console.print(Text("\nNo files were copied.", style="bold yellow"))


if __name__ == "__main__":
    app()
