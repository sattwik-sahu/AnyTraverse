from pathlib import Path
from rich.console import Console
from rich.prompt import IntPrompt
import typer
from typer import Typer

from anytraverse.utils.cli.human_op.control import HumanOperatorController
from anytraverse.utils.cli.human_op.io import (
    get_image_embedding_model,
    select_thresholds,
    select_video_from_menu,
    video_list_table,
    get_history_pickle_path,
)
from anytraverse.utils.cli.human_op.models import (
    HumanOperatorCallLogs,
    LoopbackLogsModel,
)
from anytraverse.utils.cli.human_op.prompt_store import save_store
from typing_extensions import Annotated


app = Typer(name="human_op", help="Run the human operator evaluation pipeline")
console = Console()


@app.command("videos", help="List all available videos")
def videos() -> None:
    console.print(video_list_table())


@app.command("clear_logs", help="Clears all logs for videos")
def clear_logs(
    human_op_log_file: Annotated[
        Path,
        typer.Option("--hoc", "-o", help="The path to the log file"),
    ] = Path("data/logs/human-op.json"),
    replay_log_file: Annotated[
        Path, typer.Option("--rep", "-r", help="The path to the replay log file")
    ] = Path("data/logs/loopback.json"),
):
    with console.status("Clearing loopback logs..."):
        with open(replay_log_file, "r") as f:
            loopback: LoopbackLogsModel = LoopbackLogsModel.model_validate_json(
                f.read()
            )
            loopback.logs.clear()
        with open(replay_log_file, "w") as f:
            f.write(loopback.model_dump_json())
    console.print("Cleared loopback logs", style="cyan")

    with console.status("Clearing human operation logs..."):
        with open(human_op_log_file, "r") as f:
            human_op: HumanOperatorCallLogs = HumanOperatorCallLogs.model_validate_json(
                f.read()
            )
            human_op.logs.clear()
        with open(human_op_log_file, "w") as f:
            f.write(human_op.model_dump_json())
    console.print("Cleared human operation logs", style="cyan")


@app.command("run", help="Run the human operator pipeline")
def main():
    video, video_path = select_video_from_menu()
    console.log(f"📹 User chose video: {video_path.as_posix()}")
    print()

    # Choose the image embedding model
    image_embedding_model, image_embedding_choice = get_image_embedding_model()
    console.log(
        f"🖼️  Using image embedding model: [bold bright_blue]{image_embedding_choice.value}[/]"
    )

    # Create human operator controller
    thresholds = select_thresholds()
    operator = HumanOperatorController(
        video=video,
        video_path=video_path,
        console=console,
        n_frames_skip=IntPrompt(console=console).ask(
            "How many frames to skip", default=0
        ),
        roi_unc_thresh=thresholds["roi_unc"],
        seg_thresh=thresholds["seg"],
        ref_sim_thresh=thresholds["ref_sim"],
        image_embedding_model=image_embedding_model,
    )
    console.log("Ready to start human operator control loop")
    input("\n\nPress [ENTER] to start the journey")
    print("=" * 40, end="\n\n")
    operator()
    print()

    store_save_path: Path = get_history_pickle_path(video=video, console=console)

    with console.status(f"Saving prompts store to {store_save_path.as_posix()}"):
        save_store(
            image_embeddings=image_embedding_choice,
            filepath=store_save_path,
            scene_prompts_store_manager=operator._prompt_store,
        )
    console.log(f"[lightgreen]Saved store to {store_save_path.as_posix()}\n\n")
    console.print("/// BYE BYE ///", justify="center", style="bold green")


if __name__ == "__main__":
    main()
