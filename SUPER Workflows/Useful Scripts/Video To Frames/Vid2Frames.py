"""
A script to extract frames from video files.

This script is fully interactive and provides a robust user experience:
- It automatically checks for required dependencies (OpenCV, FFmpeg) and
  provides installation instructions if they are missing.
- It asks the user for the source and destination directories.
- It offers multiple frame extraction modes (all, interval, or capped).
- It allows choosing the output image format (.jpg or .png).
- It can optionally create accompanying .txt files for each frame.
- It provides a summary of all settings and requires confirmation before starting.
"""

import os
import sys
import subprocess
import shutil

# --- Dependency Checker ---
def check_dependencies():
    """Checks for required software (OpenCV, FFmpeg) and provides guidance."""
    print("Checking for required dependencies...")
    # Check for opencv-python (cv2)
    try:
        __import__('cv2')
        print("[✓] OpenCV (cv2) is installed.")
    except ImportError:
        print("[!] The 'opencv-python' library is required but not found.")
        answer = input("Do you want to try and install it now? (y/n): ").strip().lower()
        if answer == 'y':
            try:
                print("Installing opencv-python...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
                print("[✓] opencv-python installed successfully. Please run the script again.")
                sys.exit()
            except Exception as e:
                print(f"[✗] Error installing: {e}\nPlease install manually: pip install opencv-python")
                sys.exit(1)
        else:
            print("Installation declined. The script cannot continue.")
            sys.exit(1)

    # Check for ffmpeg
    if shutil.which("ffmpeg") is None:
        print("\n[!] WARNING: FFmpeg not found in your system's PATH.")
        print("    This script can still process most videos (like .mp4) with OpenCV,")
        print("    but will fail on formats like .webm.")
        print("    For full compatibility, please install FFmpeg from: https://ffmpeg.org/download.html")
        input("Press Enter to continue without FFmpeg support...")
    else:
        print("[✓] FFmpeg is installed.")

check_dependencies()
import cv2
# --- End of Dependency Checker ---

def extract_frames_ffmpeg(video_path, output_folder, output_format, frame_limit=None, interval=None):
    """Uses FFmpeg to extract frames. More robust for various codecs."""
    os.makedirs(output_folder, exist_ok=True)
    print(f"Processing '{os.path.basename(video_path)}' with FFmpeg...")
    
    # Base command
    command = ['ffmpeg', '-hide_banner', '-i', video_path]
    
    # Add filters based on user choice
    if interval is not None:
        command.extend(['-vf', f'fps=1/{interval}'])
    if frame_limit is not None:
        command.extend(['-vframes', str(frame_limit)])

    # Output format and path
    command.extend(['-qscale:v', '2', os.path.join(output_folder, f'%06d{output_format}')])

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"  [✓] Successfully extracted frames.")
    except subprocess.CalledProcessError as e:
        print(f"  [✗] FFmpeg Error for {os.path.basename(video_path)}:")
        print(f"      {e.stderr.strip().splitlines()[-1]}") # Print last line of error

def extract_frames_opencv(video_path, output_folder, output_format, frame_limit=None, interval=None):
    """Uses OpenCV to extract frames. Good fallback if FFmpeg is not present."""
    print(f"Processing '{os.path.basename(video_path)}' with OpenCV...")
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"  [✗] Error: Could not open video file with OpenCV.")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    fps = video.get(cv2.CAP_PROP_FPS) or 30 # Default to 30 if FPS is not available
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    saved_frames = 0
    
    # Determine which frames to save
    frames_to_save = set()
    if interval is not None:
        frame_interval = int(fps * interval)
        if frame_interval > 0:
            frames_to_save = set(range(0, total_frames, frame_interval))
    elif frame_limit is not None and frame_limit > 0 and total_frames > 0:
        step = total_frames // frame_limit
        if step > 0:
            frames_to_save = set(range(0, total_frames, step))
    else: # Export all
        frames_to_save = set(range(total_frames))

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        if frame_count in frames_to_save:
            frame_filename = os.path.join(output_folder, f"{saved_frames:06d}{output_format}")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1
            if frame_limit is not None and saved_frames >= frame_limit:
                break
        
        frame_count += 1
    
    video.release()
    print(f"  [✓] Finished. Saved {saved_frames} frames.")


def create_txt_files(image_folder, text_content, output_format):
    """Creates a .txt file for each image in a folder."""
    print(f"Creating .txt files in '{image_folder}'...")
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(output_format):
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(image_folder, txt_filename)
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text_content)

def main():
    """Main function to handle user interaction and orchestrate the process."""
    print("\n--- Video to Frames Extractor ---")
    
    # 1. Get source directory
    source_dir = ""
    while not os.path.isdir(source_dir):
        user_input = input("Enter the source directory with your videos (or press Enter for current): ").strip()
        source_dir = user_input or os.getcwd()
        if not os.path.isdir(source_dir):
            print(f"Error: Directory not found at '{source_dir}'.")
            
    # 2. Get output directory
    default_output = os.path.join(source_dir, "_video_frames_output")
    output_dir = input(f"Enter the main output directory (or press Enter for default: '{default_output}'): ").strip()
    if not output_dir:
        output_dir = default_output

    # 3. Get export mode
    frame_limit, interval = None, None
    while True:
        print("\nChoose how to export frames:")
        print("  1. Export all frames (can create a very large number of files)")
        print("  2. Export frames at a specific interval (e.g., one frame every 5 seconds)")
        print("  3. Export a specific total number of frames (e.g., exactly 100 frames)")
        choice = input("Enter your choice (1/2/3, default: 2): ").strip() or "2"
        try:
            if choice == '1': break
            if choice == '2':
                interval = float(input("Enter interval in seconds (e.g., 0.5): ").strip())
                if interval <= 0: print("Interval must be a positive number."); continue
                break
            if choice == '3':
                frame_limit = int(input("Enter total number of frames to export (e.g., 100): ").strip())
                if frame_limit <= 0: print("Frame limit must be a positive number."); continue
                break
            print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    # 4. Get output format
    output_format = ""
    while output_format not in ['.jpg', '.png']:
        choice = input("Choose output format (1: JPG - smaller, 2: PNG - lossless, default: 1): ").strip() or "1"
        if choice == '1': output_format = '.jpg'
        elif choice == '2': output_format = '.png'
        else: print("Invalid choice.")

    # 5. Get .txt file option
    generate_txt = input("\nDo you want to generate a matching .txt file for each image? (y/n, default: n): ").strip().lower()
    text_to_fill = ""
    if generate_txt == 'y':
        text_to_fill = input("Enter the text to put in each .txt file: ").strip()

    # --- Find videos and show summary ---
    video_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
    if not video_files:
        print(f"\nNo video files found in '{source_dir}'. Exiting.")
        return
        
    print("\n--- Summary ---")
    print(f"Source Directory: {source_dir}")
    print(f"Output Directory: {output_dir}")
    print(f"Videos to Process: {len(video_files)}")
    if interval: print(f"Export Mode: One frame every {interval} seconds")
    elif frame_limit: print(f"Export Mode: {frame_limit} total frames per video")
    else: print("Export Mode: All frames")
    print(f"Output Format: {output_format.upper()}")
    print(f"Generate .txt files: {'Yes' if generate_txt == 'y' else 'No'}")
    if generate_txt == 'y': print(f"  -> Text content: '{text_to_fill}'")
    print("-----------------\n")

    try:
        input("Press Enter to begin, or Ctrl+C to cancel.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
        
    # --- Main Processing Loop ---
    use_ffmpeg = shutil.which("ffmpeg") is not None
    for video_file in video_files:
        video_path = os.path.join(source_dir, video_file)
        subfolder_name = os.path.splitext(video_file)[0]
        video_output_folder = os.path.join(output_dir, subfolder_name)

        if use_ffmpeg:
            extract_frames_ffmpeg(video_path, video_output_folder, output_format, frame_limit, interval)
        else:
            extract_frames_opencv(video_path, video_output_folder, output_format, frame_limit, interval)
        
        if generate_txt == 'y' and os.path.exists(video_output_folder):
            create_txt_files(video_output_folder, text_to_fill, output_format)

if __name__ == "__main__":
    main()