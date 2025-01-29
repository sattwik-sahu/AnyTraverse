from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Type, TypeVar

from rich.console import Console
from rich.prompt import Confirm, FloatPrompt, IntPrompt, Prompt
from rich.table import Table

from config.utils import WeightedPrompt
from utils.cli.human_op.models import (
    DatasetVideo,
    HumanOperatorCallLog,
    HumanOperatorCallLogs,
    Thresholds,
)
from utils.cli.human_op.video import get_video_path

# Define a type variable for the enum class
EnumT = TypeVar("EnumT", bound=Enum)


def display_menu_and_get_selection(enum_class: Type[EnumT]) -> EnumT:
    """
    Displays a simple menu using the rich library and gets the selected item from the user.

    Args:
        enum_class (Type[EnumT]): The enum class whose members will be displayed as menu items.

    Returns:
        EnumT: The selected enum item.
    """
    console = Console()

    # Display the menu title
    console.print("[bold magenta]Menu[/bold magenta]", justify="center")

    # Display each enum member as a menu item
    for index, member in enumerate(enum_class, start=1):
        console.print(f"[cyan][{index}][/cyan] {member.value}")

    # Get the user's selection
    selected_index = Prompt.ask(
        "Please select an option",
        choices=[str(i) for i in range(1, len(enum_class) + 1)],
    )

    # Convert the selected index to the corresponding enum member
    selected_member = list(enum_class)[int(selected_index) - 1]

    return selected_member


def select_video_from_menu() -> Tuple[DatasetVideo, Path]:
    video: DatasetVideo = display_menu_and_get_selection(enum_class=DatasetVideo)
    return video, get_video_path(dataset=video)


def select_thresholds() -> Thresholds:
    return Thresholds(
        ref_sim=FloatPrompt.ask(
            "Threshold for similarity to scene reference frame",
            default=0.9,
            show_default=True,
        ),
        roi=FloatPrompt.ask("Threshold for ROI", default=0.5, show_default=True),
        seg=FloatPrompt.ask("Segmentation threshold", default=0.25, show_default=True),
    )


def get_prompts(console: Console = Console()) -> List[WeightedPrompt]:
    console.print(
        "[bold]Format:[/] <prompt_1>: <weight_1>; <prompt_2>: <weight_2>; ...; <prompt_k>: <weight_k>",
        style="dim",
    )
    prompts: List[WeightedPrompt] = []
    while True:
        try:
            prompts_str: str = Prompt.ask(
                "Enter prompts and weights", console=console
            ).strip()

            if not prompts_str:
                return []

            if prompts_str.count(";") != prompts_str.count(":") - 1:
                raise
            for ps in prompts_str.split(";"):
                ps = ps.strip()
                p, w = ps.split(":")
                p, w = p.strip(), w.strip()
                prompts.append((p, float(w)))
        except Exception:
            console.log("ERROR: Prompt format wrong", style="red")
            continue
        else:
            break
    return prompts


def dict_to_table(kv: Dict[str, str]) -> Table:
    table: Table = Table()
    table.add_column("Parameter", style="light_green", justify="left")
    table.add_column("Value", style="bold cyan", justify="left")

    for k, v in kv.items():
        table.add_row(k, v)

    return table


def op_call_req_confirm(console: Console = Console()) -> bool:
    return Confirm.ask(
        "🗿 Human Operator call required at this frame?", default=False, console=console
    )


def save_log(log: HumanOperatorCallLog) -> None:
    with open("data/logs/human-op.json", "r") as f:
        logs = HumanOperatorCallLogs.model_validate_json(f.read())
    logs.logs.append(log)
    with open("data/logs/human-op.json", "w") as f:
        f.write(logs.model_dump_json())

def video_list_table() -> Table:
    table = Table(title="Available Videos")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="green")
    for dv in DatasetVideo:
        table.add_row(dv.name, get_video_path(dv).as_posix())
    return table
