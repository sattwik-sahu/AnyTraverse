import streamlit as st
import torch
from PIL import Image
import numpy as np
from typing import TypedDict


class GroundTruthMask(TypedDict):
    image: Image.Image
    tensor: torch.BoolTensor


def overlay_mask(
    image: Image.Image, mask: torch.Tensor, alpha: float = 0.5, color=(255, 0, 0)
):
    """
    Overlay a boolean segmentation mask on an image.

    Args:
        image (PIL.Image.Image): The original image.
        mask (torch.Tensor): Boolean segmentation mask (2D tensor).
        alpha (float): Transparency level for the overlay.
        color (tuple): RGB color for the mask overlay.

    Returns:
        PIL.Image.Image: Image with the overlay.
    """
    # Ensure the mask is binary and convert it to a PIL Image
    mask = mask.squeeze().byte()  # Convert boolean to 0/1 and ensure it's 2D
    mask_pil = Image.fromarray(mask.numpy() * 255).convert("L")

    # Create a color overlay
    color_overlay = Image.new("RGB", image.size, color=color)
    mask_colored = Image.composite(color_overlay, image, mask_pil)

    # Blend the original image and the mask overlay
    blended = Image.blend(image, mask_colored, alpha=alpha)
    return blended


def load_binary_mask(path: str) -> GroundTruthMask:
    image = Image.open(path)
    image_L = image.convert("L")
    mask_tensor = torch.BoolTensor(np.array(image_L) > 0)
    return {"image": image, "tensor": mask_tensor}


if __name__ == "__main__":
    # Example usage in Streamlit
    st.title("Segmentation Mask Overlay")

    # Upload image
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_image:
        # Open the image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Create a dummy mask (random example, replace with your actual mask)
        mask = torch.zeros((image.height, image.width), dtype=torch.uint8)
        mask[100:200, 100:200] = 1  # Example region to mark

        # Overlay the mask on the image
        overlaid_image = overlay_mask(image, mask, alpha=0.6, color=(255, 0, 0))

        # Display the result
        st.image(overlaid_image, caption="Image with Overlay", use_column_width=True)
