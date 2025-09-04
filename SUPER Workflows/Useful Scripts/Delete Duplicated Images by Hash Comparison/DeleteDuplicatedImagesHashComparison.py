"""
A script to find and process duplicate image files within a specified directory
and its subdirectories by comparing file hashes.

This script is fully interactive and prioritizes safety:
- It asks the user to specify the target directory.
- It asks the user what action to take on duplicates:
  1. List them (default, 100% safe).
  2. Move them to a separate folder (reversible).
  3. Permanently delete them (requires extra confirmation).
"""

import os
import sys
import hashlib
import shutil

def generate_file_hash(file_path, hash_algorithm=hashlib.md5):
    """Generate a hash for the file using the specified hash algorithm."""
    hash_obj = hash_algorithm()
    try:
        with open(file_path, 'rb') as f:
            # Read the file in chunks to avoid memory issues with large files
            while chunk := f.read(8192):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except (IOError, OSError) as e:
        print(f"Warning: Could not read file {file_path}. Skipping. Reason: {e}")
        return None

def find_and_process_duplicates(directory, action='list'):
    """
    Finds duplicate images and performs the specified action: list, move, or delete.
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp')
    hashes = {}  # Dictionary to store file hashes: {hash: original_filepath}
    duplicates_found = 0
    
    # Prepare a directory for moved duplicates if needed
    duplicates_folder = None
    if action == 'move':
        duplicates_folder = os.path.join(directory, "_duplicates")
        os.makedirs(duplicates_folder, exist_ok=True)
        print(f"Duplicates will be moved to: {duplicates_folder}")

    print("\nScanning for duplicate images... this may take a while for large directories.")
    # Walk through all files and subdirectories
    for foldername, subfolders, filenames in os.walk(directory):
        # Skip the duplicates folder if we are in 'move' mode
        if action == 'move' and foldername == duplicates_folder:
            continue
            
        for filename in filenames:
            if filename.lower().endswith(image_extensions):
                file_path = os.path.join(foldername, filename)
                file_hash = generate_file_hash(file_path)

                if file_hash is None:
                    continue # Skip files that couldn't be read

                if file_hash in hashes:
                    original_file = hashes[file_hash]
                    print(f"\nDuplicate found:")
                    print(f"  - Original: {original_file}")
                    print(f"  - Duplicate: {file_path}")
                    
                    duplicates_found += 1

                    # Perform the chosen action
                    if action == 'move':
                        try:
                            # To avoid name clashes, add the original subfolder name to the moved file
                            relative_path = os.path.relpath(os.path.dirname(file_path), directory)
                            new_filename = f"{relative_path.replace(os.sep, '_')}_{filename}" if relative_path != '.' else filename
                            shutil.move(file_path, os.path.join(duplicates_folder, new_filename))
                            print(f"  [MOVED] '{filename}' to the duplicates folder.")
                        except Exception as e:
                            print(f"  [ERROR] Could not move {file_path}: {e}")
                    
                    elif action == 'delete':
                        try:
                            os.remove(file_path)
                            print(f"  [DELETED] {file_path}")
                        except Exception as e:
                            print(f"  [ERROR] Could not delete {file_path}: {e}")
                    # If action is 'list', we just print, which we've already done.

                else:
                    # It's the first time we've seen this hash, so store it
                    hashes[file_hash] = file_path

    if duplicates_found == 0:
        print("\nScan complete. No duplicate images were found.")
    else:
        print(f"\nProcess completed. Found and processed {duplicates_found} duplicate(s).")


def main():
    """Main function to handle all user interaction."""
    print("--- Duplicate Image Finder & Cleaner ---")
    print("This script will scan a folder and its subfolders for duplicate images.")
    
    # 1. Ask for the directory to scan
    target_dir = ""
    while not target_dir:
        default_dir = os.getcwd()
        user_input = input(f"Enter the directory to scan (press Enter to use current directory: {default_dir}): ").strip()
        
        target_dir = user_input if user_input else default_dir
        
        if not os.path.isdir(target_dir):
            print(f"Error: The directory '{target_dir}' does not exist. Please try again.")
            target_dir = "" # Reset to loop again

    # 2. Ask what to do with duplicates
    action = ''
    while action not in ['list', 'move', 'delete']:
        print("\nWhat action should be taken on duplicate files?")
        print("  1. List duplicates only (Safe - Recommended for first run)")
        print("  2. Move duplicates to a subfolder named '_duplicates'")
        print("  3. Permanently DELETE duplicates (Irreversible!)")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        
        if choice == '1':
            action = 'list'
        elif choice == '2':
            action = 'move'
        elif choice == '3':
            action = 'delete'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # 3. Add a final, critical confirmation if deletion is chosen
    if action == 'delete':
        print("\n--- !!! WARNING !!! ---")
        print(f"You have chosen to PERMANENTLY DELETE files in '{target_dir}'.")
        print("This action CANNOT be undone.")
        confirm = input("Are you absolutely sure you want to proceed? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("Deletion aborted by user.")
            sys.exit()

    print("\n--- Configuration ---")
    print(f"Directory to Scan: {target_dir}")
    print(f"Action on Duplicates: {action.upper()}")
    print("---------------------\n")
    
    input("Press Enter to begin the process...")

    find_and_process_duplicates(target_dir, action)


if __name__ == "__main__":
    main()