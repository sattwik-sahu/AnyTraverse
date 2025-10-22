from typing import List, TypedDict

import pandas as pd
import streamlit as st
import torch
from PIL import Image

from anytraverse.config.utils import WeightedPrompt
from anytraverse.utils.helpers.streamlit.eval_dog_vs_rover.image import (
    GroundTruthMask,
    overlay_mask,
)
from anytraverse.utils.helpers.streamlit.eval_dog_vs_rover.models import Vehicles
from anytraverse.utils.metrics.iou import iou_torch as calc_iou
from anytraverse.utils.pipelines.base import PipelineOutput


class VehicleForm(TypedDict):
    prompts: List[WeightedPrompt]
    height: bool
    color: str


ConfigForm = Vehicles[VehicleForm]
VehicleBinaryMasks = Vehicles[torch.BoolTensor]


def upload_csv_file() -> pd.DataFrame | None:
    csv_file = st.file_uploader(
        label="Upload the CSV file",
        type=".csv",
        accept_multiple_files=False,
        key="csv_uploader",
        help="Upload your CSV file here containing the `rgb_path`, `dog_path`, `rover_path` columns",
    )
    if csv_file is None:
        st.info("Please upload a CSV file")
        return None
    else:
        df = pd.read_csv(csv_file)  # type: ignore
        return df


def create_prompt_weight_input(key_prefix: str) -> List[WeightedPrompt]:
    prompts: List[WeightedPrompt] = []

    # Get the current number of prompt pairs
    n_prompts = st.session_state.get(f"{key_prefix}_n_prompts", 1)

    # Create input fields for each prompt-weight pair
    for i in range(n_prompts):
        col1, col2, col3 = st.columns([5, 2, 1], vertical_alignment="bottom")
        with col1:
            prompt = st.text_input(f"Prompt {i+1}", key=f"{key_prefix}_prompt_{i}")
        with col2:
            weight = st.number_input(
                f"Weight {i+1}", value=1.0, key=f"{key_prefix}_weight_{i}"
            )
        with col3:
            # Only show remove button if there's more than one prompt
            if n_prompts > 1 and st.button(
                "",
                key=f"{key_prefix}_remove_{i}",
                type="tertiary",
                icon=":material/close:",
            ):
                # Remove this prompt's session state entries
                if f"{key_prefix}_prompt_{i}" in st.session_state:
                    del st.session_state[f"{key_prefix}_prompt_{i}"]
                if f"{key_prefix}_weight_{i}" in st.session_state:
                    del st.session_state[f"{key_prefix}_weight_{i}"]

                # Shift all subsequent prompts up
                for j in range(i + 1, n_prompts):
                    if f"{key_prefix}_prompt_{j}" in st.session_state:
                        st.session_state[
                            f"{key_prefix}_prompt_{
                                j-1}"
                        ] = st.session_state[f"{key_prefix}_prompt_{j}"]
                        del st.session_state[f"{key_prefix}_prompt_{j}"]
                    if f"{key_prefix}_weight_{j}" in st.session_state:
                        st.session_state[
                            f"{key_prefix}_weight_{
                                j-1}"
                        ] = st.session_state[f"{key_prefix}_weight_{j}"]
                        del st.session_state[f"{key_prefix}_weight_{j}"]

                # Update number of prompts
                st.session_state[f"{key_prefix}_n_prompts"] = n_prompts - 1
                st.rerun()

        if prompt and weight:
            prompts.append((prompt, weight))

    # Add button to add more prompt-weight pairs
    if st.button("Add Prompt", key=f"{key_prefix}_add_prompt"):
        st.session_state[f"{key_prefix}_n_prompts"] = n_prompts + 1
        st.rerun()

    return prompts


def dog_and_rover_form_cols() -> ConfigForm:
    dog_col, rover_col = st.columns(2, gap="medium")
    with dog_col:
        st.subheader("Dog Prompts")
        dog_prompts = create_prompt_weight_input(key_prefix="dog")
        dog_height = st.toggle(label="Height Scoring", key="dog_height")
        dog_color = st.color_picker(
            label="Mask color",
            help="Choose color for overlayed ground truth mask",
            key="dog_color",
        )
    with rover_col:
        st.subheader("Rover Prompts")
        rover_prompts = create_prompt_weight_input(key_prefix="rover")
        rover_height = st.toggle(label="Height Scoring", key="rover_height")
        rover_color = st.color_picker(
            label="Mask color",
            help="Choose color for overlayed ground truth mask",
            key="rover_color",
        )

    return {
        "dog": {"prompts": dog_prompts, "height": dog_height, "color": dog_color},
        "rover": {
            "prompts": rover_prompts,
            "height": rover_height,
            "color": rover_color,
        },
    }


def image_and_ground_truths(
    image: Image.Image,
    gt_dog: GroundTruthMask,
    gt_rover: GroundTruthMask,
    dog_color: str,
    rover_color: str,
):
    st.subheader("Input Image and Ground Truths")
    image_col, dog_col, rover_col = st.columns(
        3, gap="medium", vertical_alignment="center"
    )
    WIDTH: int = 420
    with image_col:
        st.image(image=image, caption="Input image", width=WIDTH)
    with dog_col:
        st.image(
            image=overlay_mask(image=image, mask=gt_dog["tensor"], color=dog_color),
            caption="Dog Ground Truth",
            width=WIDTH,
        )
    with rover_col:
        st.image(
            image=overlay_mask(image=image, mask=gt_rover["tensor"], color=rover_color),
            caption="Rover Ground Truth",
            width=WIDTH,
        )


def overlayed_mask_with_thresholding(
    image: Image.Image,
    proba_mask: torch.Tensor,
    overlay_color: str,
    key_prefix: str,
    overlay_alpha: float = 0.3,
) -> torch.BoolTensor:
    def threshold_callback():
        # This will be called whenever the slider value changes
        st.session_state.current_binary_mask = st.session_state[
            f"{key_prefix}_image_thresh"
        ]

    title_container = st.empty()
    slider_container = st.empty()
    image_container = st.empty()

    title_container.markdown(f"#### {key_prefix.capitalize()} Output")
    threshold = slider_container.slider(
        label="Threshold",
        key=f"{key_prefix}_image_thresh",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        on_change=threshold_callback,
    )
    binary_mask: torch.BoolTensor = torch.BoolTensor(proba_mask.cpu() > threshold)
    overlayed = overlay_mask(
        image=image,
        mask=binary_mask.cpu(),
        color=overlay_color,
        alpha=overlay_alpha,
    )
    image_container.image(
        overlayed, caption=f"Output for {key_prefix.capitalize()}", width=480
    )
    return binary_mask


def dog_rover_output_masks(
    image: Image.Image,
    output_dog: PipelineOutput,
    output_rover: PipelineOutput,
    dog_color: str,
    rover_color: str,
) -> VehicleBinaryMasks:
    dog_col, rover_col = st.columns(2, gap="medium", vertical_alignment="top")
    with dog_col:
        dog_mask = overlayed_mask_with_thresholding(
            image=image,
            proba_mask=output_dog.output,
            overlay_color=dog_color,
            key_prefix="dog",
            overlay_alpha=0.3,
        )
    with rover_col:
        rover_mask = overlayed_mask_with_thresholding(
            image=image,
            proba_mask=output_rover.output,
            overlay_color=rover_color,
            key_prefix="rover",
            overlay_alpha=0.3,
        )
    return {"dog": dog_mask, "rover": rover_mask}


def iou_cols(
    masks_pred: VehicleBinaryMasks, masks_true: VehicleBinaryMasks
) -> Vehicles[float]:
    dog_col, rover_col = st.columns(2, gap="medium", vertical_alignment="top")

    def _iou_col(
        col, y_pred: torch.BoolTensor, y_true: torch.BoolTensor, vehicle: str
    ) -> float:
        with col:
            iou = calc_iou(y_true=y_true, y_pred=y_pred)
            st.metric(
                label=f"{vehicle.capitalize()} IoU",
                value=round(iou, 3),
            )
        return iou

    dog_iou = _iou_col(
        col=dog_col, y_pred=masks_pred["dog"], y_true=masks_true["dog"], vehicle="dog"
    )
    rover_iou = _iou_col(
        col=rover_col,
        y_pred=masks_pred["rover"],
        y_true=masks_true["rover"],
        vehicle="rover",
    )

    return {"dog": dog_iou, "rover": rover_iou}
