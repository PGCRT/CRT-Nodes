"""
A script to create matching "caption" files for a batch of video files.

This script helps you quickly generate placeholder text files (e.g., .txt)
for all videos in a directory, filling each one with the same starting text.

It is fully interactive and prioritizes safety:
- It asks for the directory to process and can scan subfolders.
- It asks for the video extensions to look for (e.g., mp4, mov).
- It asks what to name the new caption files (e.g., .txt).
- It asks what to do if a caption file already exists (skip or overwrite).
- It shows a summary of actions and requires confirmation before starting.
"""

import os
import sys

def create_captions(directory, video_exts, caption_ext, text_content, overwrite, recursive):
    """
    Scans for video files and creates corresponding caption files.
    """
    print("\nStarting process...")
    
    # Ensure the caption extension starts with a dot
    if not caption_ext.startswith('.'):
        caption_ext = '.' + caption_ext

    files_found = 0
    captions_created = 0
    captions_skipped = 0

    # Define the folders to scan
    scan_paths = [os.path.join(directory, root) for root, _, _ in os.walk(directory)] if recursive else [directory]
    
    for folder_path in scan_paths:
        for filename in os.listdir(folder_path):
            # Check if the file has one of the target video extensions
            if filename.lower().endswith(video_exts):
                files_found += 1
                
                # Create the new caption filename
                basename = os.path.splitext(filename)[0]
                caption_filename = basename + caption_ext
                caption_filepath = os.path.join(folder_path, caption_filename)

                # Check if the caption file already exists
                if os.path.exists(caption_filepath) and not overwrite:
                    print(f"[SKIPPED] Caption already exists: {caption_filename}")
                    captions_skipped += 1
                    continue
                
                # Write the new caption file
                try:
                    with open(caption_filepath, 'w', encoding='utf-8') as f:
                        f.write(text_content)
                    print(f"[CREATED] {caption_filename}")
                    captions_created += 1
                except IOError as e:
                    print(f"[ERROR] Could not write file {caption_filename}: {e}")

    print("\n--- Process Complete ---")
    print(f"Video files found: {files_found}")
    print(f"Caption files created: {captions_created}")
    print(f"Caption files skipped: {captions_skipped}")
    print("------------------------")

def main():
    """Main function to handle all user interaction."""
    print("--- Batch Caption File Creator ---")
    
    # 1. Get Directory
    target_dir = ""
    while not os.path.isdir(target_dir):
        user_input = input("\nEnter the directory to process (or press Enter for current): ").strip()
        target_dir = user_input or os.getcwd()
        if not os.path.isdir(target_dir):
            print(f"Error: Directory not found at '{target_dir}'.")
    
    # 2. Get Video Extensions
    vid_ext_str = input("Enter video extensions to look for, separated by commas (e.g., mp4,mov,webm): ").strip().lower()
    video_extensions = tuple('.' + ext.strip() for ext in vid_ext_str.split(',') if ext.strip())
    if not video_extensions:
        print("No extensions provided. Defaulting to '.mp4'.")
        video_extensions = ('.mp4',)

    # 3. Get Caption Extension
    caption_extension = input("Enter the extension for the new caption files (e.g., txt, caption): ").strip().lower()
    if not caption_extension:
        print("No extension provided. Defaulting to '.txt'.")
        caption_extension = 'txt'

    # 4. Get Text Content
    text_content = input("Enter the text to put inside each new caption file: ").strip()
    
    # 5. Get Recursive Option
    recursive_choice = input("\nScan subdirectories as well? (y/n, default: n): ").strip().lower()
    is_recursive = recursive_choice == 'y'

    # 6. Get Overwrite Policy
    overwrite_choice = input("If a caption file already exists, should it be overwritten? (y/n, default: n): ").strip().lower()
    should_overwrite = overwrite_choice == 'y'

    # --- Summary and Confirmation ---
    print("\n--- Summary ---")
    print(f"Directory:       {target_dir}")
    print(f"Recursive Scan:  {'Yes' if is_recursive else 'No'}")
    print(f"Target Videos:   *{' | *'.join(video_extensions)}")
    print(f"Caption Files:   will be named *.{caption_extension}")
    print(f"Caption Content: \"{text_content}\"")
    print(f"Overwrite Existing: {'YES' if should_overwrite else 'NO (Safe)'}")
    print("-----------------\n")
    
    try:
        input("Press Enter to begin, or Ctrl+C to cancel.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    create_captions(target_dir, video_extensions, caption_extension, text_content, should_overwrite, is_recursive)

if __name__ == "__main__":
    main()