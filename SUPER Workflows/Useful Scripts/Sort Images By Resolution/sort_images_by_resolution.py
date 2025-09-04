"""
A script to sort image files into different folders based on their resolution.

This script is fully interactive and prioritizes safety and flexibility:
- It checks for the 'Pillow' dependency and offers to install it.
- It asks the user for the target directory and scans it recursively.
- It asks for the number of folders (levels) to create.
- It offers two different sorting methods: by equal resolution ranges or by
  equal quantity of files per folder.
- It asks whether to MOVE or COPY the files (defaulting to the safer COPY).
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

def get_image_resolution(image_path):
    """Calculates pixel count (width * height) for an image."""
    try:
        with Image.open(image_path) as img:
            return img.width * img.height
    except (IOError, OSError, Image.UnidentifiedImageError):
        # This will catch corrupted files, non-images, etc.
        return None

def sort_images(directory, num_levels, sort_method, action):
    """Finds, sorts, and processes images based on user-defined criteria."""
    print("\nScanning for images recursively... This may take a moment.")
    image_data = []
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    
    for root, _, files in os.walk(directory):
        # Avoid processing images in folders we might create
        if os.path.basename(root).startswith("resolution_level_"):
            continue
            
        for filename in files:
            if filename.lower().endswith(image_extensions):
                file_path = os.path.join(root, filename)
                resolution = get_image_resolution(file_path)
                if resolution:
                    image_data.append((file_path, resolution))

    if not image_data:
        print("Scan complete. No valid images found in the specified directory.")
        return

    print(f"Found {len(image_data)} valid images. Sorting now...")
    # Sort all images by their resolution, from smallest to largest
    image_data.sort(key=lambda x: x[1])

    # --- Create Output Folders ---
    output_folders = []
    for i in range(num_levels):
        folder_name = f"resolution_level_{i+1:02d}"
        folder_path = os.path.join(directory, folder_name)
        output_folders.append(folder_path)
        os.makedirs(folder_path, exist_ok=True)
    
    # --- Determine Thresholds for each level based on sort_method ---
    thresholds = []
    total_images = len(image_data)
    
    if sort_method == 'ranges':
        # Method 1: Split the resolution range into equal parts
        min_res = image_data[0][1]
        max_res = image_data[-1][1]
        range_size = (max_res - min_res) / num_levels
        for i in range(1, num_levels):
            thresholds.append(min_res + i * range_size)
        print("Sorting by equal resolution ranges.")

    elif sort_method == 'quantity':
        # Method 2: Split the images into equal quantity chunks
        chunk_size = total_images / num_levels
        for i in range(1, num_levels):
            index = int(i * chunk_size)
            # The threshold is the resolution of the image at the boundary
            thresholds.append(image_data[index][1])
        print("Sorting by equal quantity of files per folder.")
    
    # The last threshold is always the highest resolution
    thresholds.append(image_data[-1][1] + 1)

    # --- Process Files ---
    files_processed = 0
    for image_path, resolution in image_data:
        target_level = 0
        for i, threshold in enumerate(thresholds):
            if resolution < threshold:
                target_level = i
                break
        
        target_folder = output_folders[target_level]
        image_name = os.path.basename(image_path)
        new_path = os.path.join(target_folder, image_name)

        try:
            if action == 'copy':
                shutil.copy2(image_path, new_path)
            elif action == 'move':
                shutil.move(image_path, new_path)
            files_processed += 1
            print(f"[{action.capitalize()}] '{image_name}' -> '{os.path.basename(target_folder)}'")
        except Exception as e:
            print(f"[Error] Could not {action} {image_name}: {e}")

    print(f"\n--- Process Complete ---")
    print(f"Total files {action}ed: {files_processed}/{len(image_data)}")
    print(f"Files are sorted into {num_levels} folders in: {directory}")

def main():
    """Main function to handle all user interaction."""
    print("--- Image Sorter by Resolution ---")

    # 1. Get Directory
    target_dir = ""
    while not os.path.isdir(target_dir):
        user_input = input("Enter the directory to scan (press Enter for current folder): ").strip()
        target_dir = user_input or os.getcwd()
        if not os.path.isdir(target_dir):
            print(f"Error: Directory not found at '{target_dir}'.")
    
    # 2. Get Number of Levels
    num_levels = 0
    while num_levels < 2:
        try:
            choice = input("How many folders (levels) to sort into? (e.g., 5, default: 5): ").strip()
            num_levels = int(choice or 5)
            if num_levels < 2: print("Please enter a number of 2 or more.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    # 3. Get Sorting Method
    sort_method = ""
    while sort_method not in ['ranges', 'quantity']:
        print("\nHow should the images be sorted?")
        print("  1. By Equal Resolution Ranges (Good for evenly distributed resolutions)")
        print("  2. By Equal Quantity per Folder (Good for a balanced number of files in each folder)")
        choice = input("Enter your choice (1 or 2, default: 2): ").strip()
        if choice == '1': sort_method = 'ranges'
        elif choice == '2' or not choice: sort_method = 'quantity'
        else: print("Invalid choice.")
        
    # 4. Get Action Type (Move or Copy)
    action = ""
    while action not in ['copy', 'move']:
        choice = input("\nDo you want to MOVE or COPY the files? (1: Copy, 2: Move, default: 1): ").strip()
        if choice == '1' or not choice: action = 'copy'
        elif choice == '2': action = 'move'
        else: print("Invalid choice.")

    # 5. Summary and Final Confirmation
    print("\n--- Summary ---")
    print(f"Directory to Scan: {target_dir}")
    print(f"Number of Levels:  {num_levels}")
    print(f"Sorting Method:    {sort_method.replace('_', ' ').capitalize()}")
    print(f"Action:            {action.upper()}")
    if action == 'move':
        print("\nWARNING: You have selected to MOVE files. This action is not easily reversible.")
    print("-----------------\n")

    try:
        input("Press Enter to begin the process, or Ctrl+C to cancel.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit()
    
    sort_images(target_dir, num_levels, sort_method, action)


if __name__ == "__main__":
    main()