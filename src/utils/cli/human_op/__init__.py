from rich.console import Console
from rich.prompt import IntPrompt
from typer import Typer

from utils.cli.human_op.control import HumanOperatorController
from utils.cli.human_op.io import (
    select_thresholds,
    select_video_from_menu,
    video_list_table,
)
from utils.cli.human_op.models import LoopbackLogsModel, HumanOperatorCallLogs

app = Typer(name="human_op", help="Run the human operator evaluation pipeline")
console = Console()


@app.command("videos", help="List all available videos")
def videos() -> None:
    console.print(video_list_table())


@app.command("clear_logs", help="Clears all logs for videos")
def clear_logs():
    with open("data/logs/human-op.json", "w") as f:
        f.write("""{\n\t"logs": []\n}\n""")

    with console.status("Clearing loopback logs..."):
        with open("data/logs/loopback.json", "r") as f:
            loopback: LoopbackLogsModel = LoopbackLogsModel.model_validate_json(f.read())
            loopback.logs.clear()
        with open("data/logs/loopback.json", "w") as f:
            f.write(loopback.model_dump_json())
    console.log("Cleared loopback logs")

    with console.status("Clearing human operation logs..."):
        with open("data/logs/human-op.json", "r") as f:
            human_op: HumanOperatorCallLogs = HumanOperatorCallLogs.model_validate_json(f.read())
            human_op.logs.clear()
        with open("data/logs/human-op.json", "w") as f:
            f.write(human_op.model_dump_json())
    console.log("Cleared human operation logs")



@app.command("run", help="Run the human operator pipeline")
def main():
    video, video_path = select_video_from_menu()
    console.log(f"User chose video: {video_path.as_posix()}")
    print()

    # Create human operator controller
    thresholds = select_thresholds()
    operator = HumanOperatorController(
        video=video,
        video_path=video_path,
        console=console,
        n_frames_skip=IntPrompt(console=console).ask(
            "How many frames to skip", default=0
        ),
        roi_thresh=thresholds["roi"],
        seg_thresh=thresholds["seg"],
        ref_sim_thresh=thresholds["ref_sim"],
    )
    console.log("Ready to start human operator control loop")
    input("\n\nPress [ENTER] to start the journey")
    print("=" * 40, end="\n\n")
    operator()


if __name__ == "__main__":
    main()
