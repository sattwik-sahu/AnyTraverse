import typer

# from anytraverse.utils.cli.nav.oakd import main as oakd_planner
# from anytraverse.utils.cli.nav.oakd_simple import main as oakd_simple
# from anytraverse.utils.cli.nav.oakd_hybrid import main as oakd_hybrid
# from anytraverse.utils.cli.nav.oakd_vision import main as oakd_vision
from anytraverse.utils.cli.nav.stack import run as nav_full
# from enum import Enum
# from typing_extensions import Annotated


# class NavigationType(Enum):
#     PLANNER = "planner"
#     SIMPLE = "simple"
#     HYBRID = "hybrid"
#     VISION = "vision"


app = typer.Typer(name="navigate", help="Navigation with AnyTraverse")

app.command(name="full")(nav_full)


# @app.command(name="oakd", help="Run Unitree Go 1 with OAK-D Camera")
# def oakd(
#     nav_type: Annotated[
#         NavigationType, typer.Argument(help="The type of navigation pipeline")
#     ] = NavigationType.SIMPLE,
# ) -> None:
#     match nav_type:
#         case NavigationType.PLANNER:
#             oakd_planner()
#         case NavigationType.SIMPLE:
#             oakd_simple()
#         case NavigationType.HYBRID:
#             oakd_hybrid()
#         case NavigationType.VISION:
#             oakd_vision()
#         case _:
#             raise ValueError("Wrong value provided for argument `nav_type`")
