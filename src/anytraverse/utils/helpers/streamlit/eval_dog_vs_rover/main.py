from pathlib import Path

import streamlit as st
from PIL import Image

from anytraverse.utils.helpers.streamlit.eval_dog_vs_rover import models as models_
from anytraverse.utils.helpers.streamlit.eval_dog_vs_rover import ui
from anytraverse.utils.helpers.streamlit.eval_dog_vs_rover.image import (
    load_binary_mask,
)
from anytraverse.utils.helpers.streamlit.eval_dog_vs_rover.pipeline import (
    create_pipeline,
    run_pipeline,
)


LOG_FILEPATH = Path("data/dog_rover_prompts.json")


def main():
    st.set_page_config(page_icon="car", page_title="AnyTraverse", layout="wide")

    # Title
    st.title("AnyTraverse")

    # Upload CSV file
    df = ui.upload_csv_file()
    if df is None:
        return

    st.divider()

    c = st.number_input(label="Index", min_value=0, max_value=df.shape[0] - 1)

    # Prompts and Weights
    forms = ui.dog_and_rover_form_cols()
    dog, rover = forms["dog"], forms["rover"]

    # Load the image
    rgb_path, dog_path, rover_path, _ = df.iloc[c]
    st.info(f"Image path: `{Path(rgb_path)}`")
    image = Image.open(rgb_path)
    gt_dog = load_binary_mask(path=dog_path)
    gt_rover = load_binary_mask(path=rover_path)
    ui.image_and_ground_truths(
        image=image,
        gt_dog=gt_dog,
        gt_rover=gt_rover,
        dog_color=dog["color"],
        rover_color=rover["color"],
    )

    # Load the pipeline
    pipeline = create_pipeline()

    st.divider()

    if forms["dog"]["prompts"] and forms["rover"]["prompts"]:
        st.subheader("AnyTraverse Output")

        with st.spinner("Running pipeline for dog"):
            output_dog = run_pipeline(
                pipeline=pipeline,
                prompts=dog["prompts"],
                perform_height_scoring=dog["height"],
                image=image,
            )
        with st.spinner("Running pipeline for rover"):
            output_rover = run_pipeline(
                pipeline=pipeline,
                prompts=rover["prompts"],
                perform_height_scoring=rover["height"],
                image=image,
            )
        masks_pred = ui.dog_rover_output_masks(
            image=image,
            output_dog=output_dog,
            output_rover=output_rover,
            dog_color=dog["color"],
            rover_color=rover["color"],
        )
        vehicle_ious = ui.iou_cols(
            masks_pred=masks_pred,
            masks_true={"dog": gt_dog["tensor"], "rover": gt_rover["tensor"]},
        )

        metadata: models_.ExampleMetadataModel | None = None

        st.subheader("Example Metadata")
        example_type = st.selectbox(
            label="Example type",
            options=models_.ExampleType,
            index=None,
            placeholder="Select example type",
            key="example_type",
        )
        remarks = st.text_area(
            label="Remarks about example",
            placeholder="Enter remarks about current example",
            key="example_remarks",
        )
        if example_type is not None:
            metadata = models_.ExampleMetadataModel(type=example_type, remarks=remarks)

        def _create_data() -> models_.ExampleDataModel:
            img_data = models_.ExampleDataModel(
                rgb=Path(rgb_path),
                gt_dog=Path(dog_path),
                gt_rover=rover_path,
                vehicles=models_.VehiclesModel[models_.VehicleConfig](
                    dog=models_.VehicleConfig(
                        prompts=dog["prompts"], perform_height_scoring=dog["height"]
                    ),
                    rover=models_.VehicleConfig(
                        prompts=rover["prompts"], perform_height_scoring=rover["height"]
                    ),
                ),
                iou=models_.VehiclesModel[float](
                    dog=vehicle_ious["dog"], rover=vehicle_ious["rover"]
                ),
                metadata=metadata,  # type: ignore
            )
            return img_data

        if metadata is not None:
            btn_save = st.button(
                "Save",
                key="btn_save",
                help="Save current example and settings to log file.",
                icon=":material/check:",
                type="primary",
            )

            if btn_save:
                try:
                    with open(LOG_FILEPATH, "r") as logfile:
                        # Load the data
                        log_json = models_.LogsModel.model_validate_json(logfile.read())
                        # Append new data
                        log_json.logs.append(_create_data())

                    with open(LOG_FILEPATH, "w") as logfile:
                        # Write output to logfile
                        logfile.write(log_json.model_dump_json())
                except Exception:
                    st.toast(
                        "Error saving data to log file!", icon=":material/scan_delete:"
                    )
                else:
                    st.toast("Saved current data to log file!", icon=":material/task:")


if __name__ == "__main__":
    main()
