import typer
from enum import Enum
from anytraverse.utils.cli.eval.rugd import run_eval as eval_rugd
from anytraverse.utils.cli.eval.rellis import run_eval as eval_rellis
from anytraverse.utils.cli.eval.deepscene import run_eval as eval_deepscene
from typing_extensions import Annotated


class Dataset(Enum):
    RUGD = "rugd"
    RELLIS = "rellis"
    DEEPSCENE = "deepscene"


def eval(
    dataset: Annotated[
        Dataset, typer.Argument(help="The dataset to run the evals on.")
    ],
    show_viz: Annotated[
        bool, typer.Option(help="Whether to show the visualization plots.")
    ] = True,
) -> None:
    match dataset:
        case Dataset.RUGD:
            eval_rugd(show_viz=show_viz)
        case Dataset.RELLIS:
            eval_rellis(show_viz=show_viz)
        case Dataset.DEEPSCENE:
            eval_deepscene(show_viz=show_viz)
