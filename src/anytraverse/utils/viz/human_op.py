from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.axes import Axes
import seaborn as sns

from typing import Tuple, List
from dataclasses import dataclass

from PIL import Image


sns.set_theme()


@dataclass
class HumanOperatorUI_Axes:
    ref_frame: Axes
    curr_frame: Axes
    trav_map: Axes
    unc_map: Axes
    hoc: Axes
    sim_dist: Axes

    @property
    def axes(self) -> List[Axes]:
        return [
            self.ref_frame,
            self.curr_frame,
            self.trav_map,
            self.unc_map,
            self.hoc,
            self.sim_dist,
        ]

    def reset(self) -> None:
        # Clear all axes
        for ax in self.axes:
            ax.clear()

        # Set axes titles
        self.curr_frame.set_title("Current Frame")
        self.hoc.set_title("Human Operator Calls")
        self.ref_frame.set_title("Scene Reference Frame")
        self.sim_dist.set_title("Reference Scene Similarity")
        self.trav_map.set_title("Traversability Map")
        self.unc_map.set_title("Uncertainty Map")

        # Switch off axes for image ones
        self.curr_frame.grid(visible=False)
        self.curr_frame.set_axis_off()
        self.ref_frame.grid(visible=False)
        self.ref_frame.set_axis_off()
        self.trav_map.grid(visible=False)
        self.trav_map.set_axis_off()
        self.unc_map.grid(visible=False)
        self.unc_map.set_axis_off()


def get_human_op_ui_axes() -> Tuple[Figure, HumanOperatorUI_Axes]:
    """
    Returns the figure and axes for the human operator UI.
    """
    fig: Figure = plt.figure(layout="constrained", figsize=(32, 18))

    gs_root: GridSpec = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[5, 3])

    gs_img: GridSpecFromSubplotSpec = gs_root[0].subgridspec(nrows=1, ncols=3)
    gs_plots: GridSpecFromSubplotSpec = gs_root[1].subgridspec(
        nrows=1, ncols=2, width_ratios=[2, 1]
    )

    gs_refs: GridSpecFromSubplotSpec = gs_img[0].subgridspec(nrows=2, ncols=1)
    ax_ref_frame: Axes = fig.add_subplot(gs_refs[0, 0])
    ax_curr_frame: Axes = fig.add_subplot(gs_refs[1, 0])

    ax_trav_map: Axes = fig.add_subplot(gs_img[1])
    ax_unc_map: Axes = fig.add_subplot(gs_img[2])

    ax_hoc: Axes = fig.add_subplot(gs_plots[0])
    ax_sim_dist: Axes = fig.add_subplot(gs_plots[1])

    return fig, HumanOperatorUI_Axes(
        ref_frame=ax_ref_frame,
        curr_frame=ax_curr_frame,
        hoc=ax_hoc,
        sim_dist=ax_sim_dist,
        trav_map=ax_trav_map,
        unc_map=ax_unc_map,
    )


if __name__ == "__main__":
    fig, ax = get_human_op_ui_axes()
    image = Image.open(
        "/mnt/toshiba_hdd/datasets/rellis-3d/Rellis-3D-images/00002/pylon_camera_node/frame003602-1581797510_609.jpg"
    )
    ax.curr_frame.imshow(image)
    ax.ref_frame.imshow(image)
    ax.trav_map.imshow(image)
    ax.unc_map.imshow(image)

    plt.show()
