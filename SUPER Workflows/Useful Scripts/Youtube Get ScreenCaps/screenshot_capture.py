"""
A script to capture screenshots from a YouTube video stream at regular intervals.

This script is fully interactive and will prompt the user for all necessary information.
It also includes a dependency checker that will automatically try to install missing
required packages (opencv-python, yt-dlp) using pip.
"""

import sys
import subprocess
import os

def check_and_install_dependencies():
    """
    Checks if required packages are installed and prompts to install them if they are not.
    """
    # A dictionary of required packages: {package_name_for_pip: import_name}
    required_packages = {
        'opencv-python': 'cv2',
        'yt-dlp': 'yt_dlp'
    }

    print("Checking for required packages...")
    packages_to_install = []
    for package_name, import_name in required_packages.items():
        try:
            # Try to import the package
            __import__(import_name)
            print(f"  [✓] {package_name} is already installed.")
        except ImportError:
            print(f"  [!] {package_name} is not installed.")
            packages_to_install.append(package_name)

    if not packages_to_install:
        print("All required packages are present.\n")
        return

    print("\nThe following required packages are missing:", ", ".join(packages_to_install))
    # Ask the user for permission to install
    try:
        # Use input() and check for 'y'
        # .strip() handles whitespace, .lower() handles 'Y' or 'y'
        answer = input("Do you want to try and install them now? (y/n): ").strip().lower()
        if answer != 'y':
            print("Installation aborted by user. The script cannot continue.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nInstallation aborted by user.")
        sys.exit(1)


    print("\nAttempting to install missing packages using pip...")
    for package_name in packages_to_install:
        try:
            # Using sys.executable ensures we use the pip of the current python environment
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"  [✓] Successfully installed {package_name}.")
        except subprocess.CalledProcessError:
            print(f"  [✗] Failed to install {package_name}.")
            print("Please try to install it manually by running:")
            print(f"   {os.path.basename(sys.executable)} -m pip install {package_name}")
            sys.exit(1)
    
    print("\nAll required packages have been installed successfully.\n")


# --- Run the dependency check before importing the packages ---
check_and_install_dependencies()


# --- Imports are placed after the dependency check ---
# This ensures that they are available before the script tries to use them.
import cv2
import yt_dlp


def get_video_url(video_url):
    """Extract the best video URL using yt-dlp."""
    print("Extracting video stream URL... this may take a moment.")
    ydl_opts = {
        'format': 'bestvideo[height<=1080]+bestaudio/best',  # Limit to 1080p video quality
        'quiet': True,
        'noplaylist': True,  # Avoid extracting playlists if the URL is a playlist
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            # Check for 'url' in the info_dict
            if 'formats' in info_dict:
                for fmt in info_dict['formats']:
                    if fmt.get('height') == 1080:  # Look for the 1080p stream
                        return fmt['url']
            # Fall back to the best available if 1080p is not found
            return info_dict['url']  # Return the best quality available
    except Exception as e:
        print(f"Error extracting video URL: {e}")
        return None

def capture_screenshots_from_stream(video_url, interval, output_dir):
    """Opens a video stream and saves a screenshot at a specified interval."""
    if not os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' not found. Creating it.")
        os.makedirs(output_dir)
    
    # Get the video stream URL
    video_stream_url = get_video_url(video_url)
    if video_stream_url is None:
        print("Failed to retrieve video stream URL.")
        return

    # Open the video stream
    print("Opening video stream...")
    video_capture = cv2.VideoCapture(video_stream_url)
    if not video_capture.isOpened():
        print("Error: Unable to open video stream.")
        return

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    
    # Handle cases where FPS might be 0 or invalid
    if fps == 0:
        print("Warning: Could not determine video FPS. Using a default of 30 for interval calculation.")
        fps = 30
        
    frame_interval = int(fps * interval)  # Calculate frame interval based on FPS

    frame_count = 0
    screenshot_count = 0

    print("\nStarting screenshot capture. Press Ctrl+C in the terminal to stop.")
    while video_capture.isOpened():
        try:
            ret, frame = video_capture.read()
            if not ret:
                break
            
            # Capture a screenshot every 'frame_interval' frames
            if frame_count % frame_interval == 0:
                screenshot_path = os.path.join(output_dir, f'screenshot_{screenshot_count}.png')
                cv2.imwrite(screenshot_path, frame)
                print(f"Saved screenshot: {screenshot_path}")
                screenshot_count += 1

            frame_count += 1
        except KeyboardInterrupt:
            print("\nStopping capture due to user request (Ctrl+C).")
            break

    video_capture.release()
    print("Screenshot capture completed.")

def main():
    """
    Main function to interactively ask the user for necessary information
    and then start the screenshot capture process.
    """
    print("--- YouTube Video Screenshot Capturer ---")

    # 1. Ask for the YouTube video URL
    video_url = ""
    while not video_url:
        video_url = input("Please enter the URL of the YouTube video: ").strip()
        if not video_url:
            print("URL cannot be empty. Please try again.")

    # 2. Ask for the output directory
    output_dir = ""
    while not output_dir:
        output_dir = input("Enter the directory where screenshots will be saved (e.g., 'my_screenshots'): ").strip()
        if not output_dir:
            print("Directory name cannot be empty. Please try again.")
    
    # 3. Ask for the screenshot interval
    interval = 5  # Default interval
    while True:
        interval_str = input(f"Enter the interval between screenshots in seconds (default: {interval}): ").strip()
        if not interval_str:
            # User pressed Enter, use the default value
            break
        try:
            # Try to convert the input to a positive integer
            interval = int(interval_str)
            if interval <= 0:
                print("Please enter a positive number for the interval.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    print("\n--- Configuration ---")
    print(f"Video URL: {video_url}")
    print(f"Output Directory: {output_dir}")
    print(f"Screenshot Interval: {interval} seconds")
    print("---------------------\n")
    
    # Start the main process with the collected information
    capture_screenshots_from_stream(video_url, interval, output_dir)


if __name__ == "__main__":
    main()