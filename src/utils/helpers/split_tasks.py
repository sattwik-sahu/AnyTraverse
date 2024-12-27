import os
import random
from rich.prompt import Prompt
from rich.console import Console

console = Console()


def shuffle_and_split_images(folder_path, num_splits, split_names):
    # Get all PNG images from the folder
    images = [img for img in os.listdir(folder_path) if img.endswith(".png")]

    # Shuffle the images
    random.shuffle(images)

    # Split images into equal parts
    splits = [images[i::num_splits] for i in range(num_splits)]

    # Ensure no common frames across splits
    used_images = set()
    for split in splits:
        for image in split:
            if image in used_images:
                console.print(
                    f"Error: Duplicate image found across splits: {image}",
                    style="bold red",
                )
                return None
            used_images.add(image)

    # Ensure all splits have at least one image
    for i, split in enumerate(splits):
        if not split:
            console.print(
                f"Split '{split_names[i]}' has no images! Exiting.", style="bold red"
            )
            return None

    return splits


def transfer_split_images(folder_path, splits, split_names):
    for i, split_name in enumerate(split_names):
        split_dir = os.path.join(folder_path, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for image in splits[i]:
            src_path = os.path.join(folder_path, image)
            dest_path = os.path.join(split_dir, image)
            try:
                os.rename(src_path, dest_path)
                console.print(f"Moved {image} to {split_dir}", style="bold green")
            except Exception as e:
                console.print(
                    f"Error moving {image} to {split_dir}: {e}", style="bold red"
                )


def main():
    folder_path = Prompt.ask(
        "[bold blue]Enter the folder path containing images[/]"
    ).strip()

    if not os.path.isdir(folder_path):
        console.print("Invalid folder path!", style="bold red")
        return

    try:
        num_splits = int(Prompt.ask("[bold blue]Enter the number of splits[/]"))
        split_names = [
            Prompt.ask(f"[bold blue]Enter name for split {i+1}[/]").strip()
            for i in range(num_splits)
        ]
    except ValueError:
        console.print("Invalid input!", style="bold red")
        return

    splits = shuffle_and_split_images(folder_path, num_splits, split_names)

    if splits is None:
        return

    transfer_split_images(folder_path, splits, split_names)


if __name__ == "__main__":
    main()
