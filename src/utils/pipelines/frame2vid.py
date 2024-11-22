import cv2
import os


def frames_to_video(input_dir, output_file, fps):
    # Get list of image files in the directory
    images = [
        img
        for img in os.listdir(input_dir)
        if img.endswith(".png") or img.endswith(".jpg")
    ]
    images.sort()  # Ensure the frames are in the correct order

    if not images:
        raise ValueError("No images found in the directory.")

    # Read the first image to get the dimensions
    first_image_path = os.path.join(input_dir, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can use other codecs as well
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in images:
        image_path = os.path.join(input_dir, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert frames in a directory to a video."
    )
    parser.add_argument("input_dir", help="Directory containing the frames.")
    parser.add_argument("output_file", help="Output video file.")
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the output video."
    )

    args = parser.parse_args()
    frames_to_video(args.input_dir, args.output_file, args.fps)
