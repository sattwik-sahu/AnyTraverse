import streamlit as st
import threading
import subprocess
from pathlib import Path
import glob

def run_ffmpeg_command(video_path: str, dest_folder: str, skip_frames: int):
    """
    Run the ffmpeg command to extract frames from a video with a specific frame skipping rate.
    """
    try:
        # Create destination folder if it doesn't exist
        Path(dest_folder).mkdir(parents=True, exist_ok=True)

        # Build the ffmpeg command
        command = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select=not(mod(n\\,{skip_frames})),setpts=N/FRAME_RATE/TB",
            "-vsync", "vfr",
            f"{dest_folder}/frame_%06d.png"
        ]

        # Run the command
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            st.error(f"Error processing video {video_path}:\n{stderr.decode()}")
        else:
            st.success(f"Finished processing {video_path}.")

    except Exception as e:
        st.error(f"An error occurred while processing {video_path}: {e}")

def process_videos(video_glob: str, dest_root: str, skip_frames: int):
    """
    Process videos matching the glob pattern and extract frames into corresponding folders.
    """
    video_paths = glob.glob(video_glob)
    if not video_paths:
        st.error("No videos found matching the provided glob pattern.")
        return

    threads = []

    for video_path in video_paths:
        video_name = Path(video_path).stem
        dest_folder = Path(dest_root) / video_name

        thread = threading.Thread(
            target=run_ffmpeg_command,
            args=(video_path, dest_folder, skip_frames)
        )
        threads.append(thread)

    # Start all threads
    for thread in threads:
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Streamlit app
st.title("Video Frame Extraction Tool")

# Inputs
video_glob = st.text_input("Glob Pattern for Videos", value="*.mp4", key="video_glob_pattern")
dest_root = st.text_input("Destination Root Folder", value="./output_frames", key="output_folder")
skip_frames = st.number_input("Frames to Skip", value=5, min_value=1, key="skip_frames")

if st.button("Start Processing", key="start"):
    if not video_glob or not dest_root:
        st.error("Please provide both a glob pattern and a destination folder.")
    else:
        process_videos(video_glob, dest_root, skip_frames)
