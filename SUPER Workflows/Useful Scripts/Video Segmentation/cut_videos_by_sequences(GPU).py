"""
A script to automatically split video files into smaller clips based on detected
scene changes. It uses FFmpeg for the heavy lifting and can leverage NVIDIA GPUs
(h264_nvenc) for faster re-encoding of longer clips.

This script is interactive:
- It checks if FFmpeg is installed and provides instructions if it is not.
- It prompts the user for the scene detection sensitivity.
- It finds and processes all video files in the directory where it is run.
"""

import os
import subprocess
import sys
import shutil

def check_ffmpeg_dependency():
    """
    Checks if the FFmpeg executable is available in the system's PATH.
    If not, it prints an informative error message and exits.
    """
    print("Checking for FFmpeg dependency...")
    if shutil.which("ffmpeg") is None:
        print("\n--- ERROR: FFmpeg not found ---")
        print("This script requires FFmpeg to be installed and accessible in your system's PATH.")
        print("Please install it from the official website: https://ffmpeg.org/download.html")
        print("\nAfter installation, please restart your terminal or command prompt and try again.")
        sys.exit(1) # Exit the script
    else:
        print("[✓] FFmpeg is installed and ready.\n")

def split_video_into_scenes_in_folder(threshold=0.3):
    """
    Finds video files in the current directory and splits them into scenes
    using the provided threshold.
    """
    # Get the current directory and list all video files
    current_dir = os.getcwd()
    video_files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm'))]

    if not video_files:
        print(f"No video files found in the current directory: {current_dir}")
        return

    print(f"Found {len(video_files)} video file(s) to process.")
    output_dir = os.path.join(current_dir, "Scene_Splits")

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for video_file in video_files:
        base_name = os.path.splitext(video_file)[0]
        video_path = os.path.join(current_dir, video_file)

        # Create a subfolder for each video
        video_output_dir = os.path.join(output_dir, base_name)
        if not os.path.exists(video_output_dir):
            os.makedirs(video_output_dir)

        # Temporary file to store scene change timestamps
        timestamps_file = os.path.join(video_output_dir, f"{base_name}_timestamps.txt")

        # Step 1: Detect scene changes and save timestamps
        # Using -hide_banner to make the output cleaner
        ffmpeg_scene_detect_cmd = [
            "ffmpeg", "-hide_banner",
            "-i", video_path,
            "-filter_complex", f"select='gt(scene,{threshold})',metadata=print",
            "-an", "-f", "null", "-", # -an ignores audio, -f null doesn't write a file
        ]

        print(f"\n[{video_file}] Detecting scene changes...")
        # Use subprocess.PIPE to capture stdout/stderr directly
        result = subprocess.run(ffmpeg_scene_detect_cmd, capture_output=True, text=True)
        
        # Write the captured output to the timestamps file
        with open(timestamps_file, "w") as f:
            f.write(result.stderr) # Scene detection info goes to stderr

        # Step 2: Parse timestamps from the metadata
        with open(timestamps_file, "r") as f:
            lines = f.readlines()

        timestamps = [0.0] # Start at the beginning
        for line in lines:
            if "pts_time:" in line:
                try:
                    timestamps.append(float(line.split("pts_time:")[1].strip()))
                except ValueError:
                    continue

        timestamps = sorted(list(set(timestamps)))

        if len(timestamps) <= 1:
            print(f"No scenes detected in {video_file} with threshold {threshold}. Skipping.")
            continue

        print(f"Detected {len(timestamps) - 1} scene changes in {video_file}.")

        # Step 3: Split the video into scenes
        for i in range(len(timestamps)):
            start_time = timestamps[i]
            # Determine end time: either the next timestamp or the end of the video
            end_time = timestamps[i+1] if i + 1 < len(timestamps) else None
            
            duration = (end_time - start_time) if end_time is not None else None

            # Skip creating clips that are too short
            if duration is not None and duration < 0.5:
                continue

            output_file = os.path.join(video_output_dir, f"{base_name}_{i + 1:03d}.mp4")
            
            # Base command
            ffmpeg_split_cmd = ["ffmpeg", "-hide_banner", "-y", "-ss", str(start_time), "-i", video_path]
            
            # Add duration if it's not the last clip
            if duration is not None:
                ffmpeg_split_cmd.extend(["-t", str(duration)])

            # Use GPU encoding for clips over 2 seconds, otherwise just copy streams
            if duration and duration > 2:
                ffmpeg_split_cmd.extend([
                    "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "24",
                    "-c:a", "aac", "-b:a", "192k"
                ])
            else:
                ffmpeg_split_cmd.extend(["-c:v", "copy", "-c:a", "copy"])

            ffmpeg_split_cmd.append(output_file)

            print(f"  Exporting scene {i + 1:03d}...")
            # Using DEVNULL to hide ffmpeg output for a cleaner interface
            subprocess.run(ffmpeg_split_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"Finished splitting {video_file}.")
        os.remove(timestamps_file) # Clean up the temporary file

    print(f"\nAll scenes have been exported to the '{output_dir}' folder.")

def main():
    """
    Main function to handle user interaction.
    """
    print("--- Video Scene Splitter (GPU Accelerated) ---")
    print("This script will find all video files in its directory and split them into scenes.")
    
    # 1. Ask for the scene detection threshold
    default_threshold = 0.3
    threshold = 0.0
    while True:
        prompt = f"Enter scene detection threshold (0.1=very sensitive, 0.5=less sensitive, default: {default_threshold}): "
        user_input = input(prompt).strip()
        
        if not user_input:
            threshold = default_threshold
            break
        try:
            val = float(user_input)
            if 0.0 < val < 1.0:
                threshold = val
                break
            else:
                print("Invalid input. Please enter a number between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number (e.g., 0.3).")

    print("\n--- Configuration ---")
    print(f"Processing videos in: {os.getcwd()}")
    print(f"Scene detection threshold: {threshold}")
    print("---------------------\n")
    
    # 2. Start the main process
    split_video_into_scenes_in_folder(threshold)

if __name__ == "__main__":
    # First, ensure dependencies are met
    check_ffmpeg_dependency()
    # Then, run the interactive main function
    main()