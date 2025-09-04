"""
A script to find and process images based on the presence of a transparency
(alpha) channel.

This script is fully interactive and prioritizes safety:
- It checks for the 'Pillow' dependency and offers to install it.
- It asks for the directory to scan and can do so recursively.
- It asks whether to target images WITH or WITHOUT transparency.
- It provides safe actions: List (default), Move, or Delete.
- It presents a summary and requires final confirmation before starting.
"""

import os
import sys
import subprocess
import shutil

# --- Dependency Checker ---
def check_and_install_dependencies():
    """Checks for the Pillow library and offers to install it if missing."""
    try:
        __import__('PIL')
        print("[✓] Pillow (PIL) is already installed.")
    except ImportError:
        print("[!] The 'Pillow' library is required but not found.")
        answer = input("Do you want to try and install it now? (y/n): ").strip().lower()
        if answer == 'y':
            try:
                print("Installing Pillow...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
                print("[✓] Pillow installed successfully. Please run the script again.")
                sys.exit()
            except Exception as e:
                print(f"[✗] Error installing Pillow: {e}")
                print("Please install it manually by running: pip install Pillow")
                sys.exit(1)
        else:
            print("Installation declined. The script cannot continue.")
            sys.exit(1)

check_and_install_dependencies()
# --- End of Dependency Checker ---

from PIL import Image

def has_alpha(image):
    """
    Check if an image has any transparency.
    Returns True if the image has an alpha channel and at least one pixel is not fully opaque.
    """
    # 'RGBA' is for color images, 'LA' for grayscale. Both have alpha.
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        # An alpha channel exists. Check if it's actually used.
        # .getextrema() returns a (min, max) tuple. If min is less than 255, some pixels are transparent.
        try:
            alpha = image.getchannel('A')
            return alpha.getextrema()[0] < 255
        except (ValueError, KeyError):
            # Some palette-based images ('P' mode) might not have a simple alpha channel.
            # The 'transparency' in info is a good fallback.
            return True
    return False

def process_images(directory, action, mode, recursive):
    """
    Finds and processes images based on their alpha channel.
    """
    print("\nScanning for images... This may take a moment.")
    
    # Define which images we're interested in
    image_extensions = ('.png', '.gif', '.tiff', '.webp') # Formats that commonly support transparency
    
    # Prepare a destination folder for the 'move' action
    move_folder_path = ""
    if action == 'move':
        folder_name = f"_moved_images_{mode.replace('_', '_')}"
        move_folder_path = os.path.join(directory, folder_name)
        os.makedirs(move_folder_path, exist_ok=True)

    images_found = 0
    images_processed = 0

    # Define the folders to scan
    scan_paths = [os.path.join(directory, root) for root, _, _ in os.walk(directory)] if recursive else [directory]

    for folder_path in scan_paths:
        # Avoid processing files in the folder we are moving to
        if folder_path == move_folder_path:
            continue
            
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(image_extensions):
                file_path = os.path.join(folder_path, filename)
                try:
                    with Image.open(file_path) as img:
                        image_has_alpha = has_alpha(img)
                        
                        # Determine if this image is a target based on the user's choice
                        is_target = (mode == 'without_alpha' and not image_has_alpha) or \
                                    (mode == 'with_alpha' and image_has_alpha)
                        
                        if is_target:
                            images_found += 1
                            target_type = "HAS_ALPHA" if image_has_alpha else "NO_ALPHA"
                            print(f"\n[TARGET: {target_type}] Found: {file_path}")
                            
                            if action == 'move':
                                shutil.move(file_path, move_folder_path)
                                print(f"  [MOVED] to '{os.path.basename(move_folder_path)}'")
                                images_processed += 1
                            elif action == 'delete':
                                os.remove(file_path)
                                print(f"  [DELETED]")
                                images_processed += 1
                            # For 'list', just printing is enough.
                            
                except (IOError, Image.UnidentifiedImageError) as e:
                    print(f"Skipping {filename} (Could not open as image: {e})")

    print("\n--- Process Complete ---")
    print(f"Found {images_found} target image(s).")
    if action != 'list':
        print(f"Successfully {action}d {images_processed} file(s).")
    print("------------------------")


def main():
    """Main function to handle all user interaction."""
    print("--- Alpha Channel Image Processor ---")
    
    # 1. Get Directory and Recursion
    target_dir = ""
    while not os.path.isdir(target_dir):
        user_input = input("\nEnter the directory to scan (or press Enter for current): ").strip()
        target_dir = user_input or os.getcwd()
        if not os.path.isdir(target_dir): print(f"Error: Directory not found at '{target_dir}'.")
    
    recursive_choice = input("Scan subdirectories as well? (y/n, default: n): ").strip().lower()
    is_recursive = recursive_choice == 'y'

    # 2. Get Mode (which images to target)
    mode = ""
    while mode not in ['with_alpha', 'without_alpha']:
        print("\nWhich images do you want to target?")
        print("  1. Images WITHOUT transparency (fully opaque)")
        print("  2. Images WITH transparency (have an alpha channel with transparent pixels)")
        choice = input("Enter your choice (1 or 2, default: 1): ").strip() or "1"
        if choice == '1': mode = 'without_alpha'
        elif choice == '2': mode = 'with_alpha'
        else: print("Invalid choice.")

    # 3. Get Action
    action = ""
    while action not in ['list', 'move', 'delete']:
        print("\nWhat action should be taken on the targeted images?")
        print("  1. List them only (Safe - Recommended for first run)")
        print("  2. Move them to a new subfolder")
        print("  3. Permanently DELETE them (Irreversible!)")
        choice = input("Enter your choice (1, 2, or 3, default: 1): ").strip() or "1"
        if choice == '1': action = 'list'
        elif choice == '2': action = 'move'
        elif choice == '3': action = 'delete'
        else: print("Invalid choice.")

    # 4. Final Confirmation
    if action == 'delete':
        print("\n--- !!! WARNING !!! ---")
        print(f"You have chosen to PERMANENTLY DELETE files in '{target_dir}'.")
        print("This action CANNOT be undone.")
        confirm = input("Are you absolutely sure? Type 'yes' to proceed: ").strip().lower()
        if confirm != 'yes':
            print("Deletion aborted by user.")
            sys.exit()

    print("\n--- Summary ---")
    print(f"Directory:    {target_dir}")
    print(f"Recursive:    {'Yes' if is_recursive else 'No'}")
    print(f"Targeting:    Images {mode.replace('_', ' ').upper()}")
    print(f"Action:       {action.upper()}")
    print("-----------------\n")

    try:
        input("Press Enter to begin, or Ctrl+C to cancel.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return

    process_images(target_dir, action, mode, is_recursive)

if __name__ == "__main__":
    main()