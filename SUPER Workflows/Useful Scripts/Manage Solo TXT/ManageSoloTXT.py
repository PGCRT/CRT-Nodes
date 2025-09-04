"""
A script to clean up a directory by finding and managing "lonely" files.

This script helps you find files that don't have a matching partner file
with a different extension.

A common use case is cleaning a dataset where you might have leftover
caption files (e.g., 'image1.txt') without a corresponding image file
(e.g., 'image1.png'). This script will find those "lonely" .txt files for you.

It is interactive and safe:
- Asks which folder to clean.
- Asks for the two file extensions to compare.
- Lets you choose to just LIST, MOVE, or DELETE the lonely files.
"""

import os
import sys
import shutil

def process_lonely_files(directory, main_extension, lonely_extension, action, recursive):
    """
    Finds and processes "lonely" files that don't have a matching main file.
    """
    print("\nScanning for files... This may take a moment.")
    
    # --- Step 1: Collect the names of all main files ---
    # We store just the 'basename' (filename without extension)
    main_file_basenames = set()
    
    # --- Step 2: Collect the full paths of all files to check ---
    files_to_check_paths = []

    # Ensure extensions start with a dot
    if not main_extension.startswith('.'): main_extension = '.' + main_extension
    if not lonely_extension.startswith('.'): lonely_extension = '.' + lonely_extension

    # Define the folders to scan
    scan_folders = [os.path.join(directory, root) for root, _, _ in os.walk(directory)] if recursive else [directory]

    for folder_path in scan_folders:
        try:
            for filename in os.listdir(folder_path):
                basename, ext = os.path.splitext(filename)
                ext = ext.lower()
                
                if ext == main_extension:
                    main_file_basenames.add(os.path.join(folder_path, basename))
                elif ext == lonely_extension:
                    files_to_check_paths.append(os.path.join(folder_path, filename))
        except FileNotFoundError:
            print(f"Warning: Directory not found during scan: {folder_path}. Skipping.")
            continue
            
    # --- Step 3: Find the lonely files and perform the chosen action ---
    lonely_files_found = 0
    
    # Prepare a destination folder for the 'move' action
    move_destination_folder = os.path.join(directory, f"lonely_{lonely_extension.strip('.')}_files")
    if action == 'move':
        os.makedirs(move_destination_folder, exist_ok=True)

    print("Checking for lonely files...")
    for check_path in files_to_check_paths:
        check_basename = os.path.splitext(check_path)[0]
        
        # If the file's basename isn't in the set of main files, it's lonely.
        if check_basename not in main_file_basenames:
            lonely_files_found += 1
            print(f"\n[LONELY FILE] Found: {check_path}")
            
            if action == 'move':
                try:
                    shutil.move(check_path, move_destination_folder)
                    print(f"  [MOVED] to '{os.path.basename(move_destination_folder)}'")
                except Exception as e:
                    print(f"  [ERROR] Could not move file: {e}")
            elif action == 'delete':
                try:
                    os.remove(check_path)
                    print(f"  [DELETED]")
                except Exception as e:
                    print(f"  [ERROR] Could not delete file: {e}")
            # If action is 'list', we just print the "Found" message, which is already done.
                
    print("\n--- Process Complete ---")
    if lonely_files_found == 0:
        print(f"No lonely '*{lonely_extension}' files were found.")
    else:
        print(f"Found a total of {lonely_files_found} lonely file(s).")
    print("------------------------")

def main():
    """Main function to handle all user interaction."""
    print("--- Lonely File Cleaner ---")
    print("This finds files that don't have a matching partner with another extension.")
    
    # 1. Get Directory
    target_dir = ""
    while not os.path.isdir(target_dir):
        user_input = input("\nEnter the directory to clean (or press Enter for current): ").strip()
        target_dir = user_input or os.getcwd()
        if not os.path.isdir(target_dir):
            print(f"Error: Directory not found at '{target_dir}'.")
            
    # 2. Get File Extensions
    print("\nNow, tell me which file extensions to compare.")
    print("Example: To find '.txt' files without a matching '.png', you would enter 'png' then 'txt'.")
    main_ext = input("Enter the MAIN file extension (e.g., png, jpg): ").strip().lower()
    lonely_ext = input(f"Enter the extension to check for being 'lonely' (e.g., txt): ").strip().lower()

    # 3. Get Recursive Option
    recursive_choice = input("\nScan subdirectories as well? (y/n, default: n): ").strip().lower()
    is_recursive = recursive_choice == 'y'

    # 4. Get Action
    action = ""
    while action not in ['list', 'move', 'delete']:
        print("\nWhat should I do with the lonely files I find?")
        print("  1. List them only (Safe - Recommended for first run)")
        print("  2. Move them to a new subfolder")
        print("  3. Permanently DELETE them (Irreversible!)")
        choice = input("Enter your choice (1, 2, or 3, default: 1): ").strip() or "1"
        if choice == '1': action = 'list'
        elif choice == '2': action = 'move'
        elif choice == '3': action = 'delete'
        else: print("Invalid choice.")

    # 5. Final Confirmation
    if action == 'delete':
        print("\n--- !!! WARNING !!! ---")
        print(f"You have chosen to PERMANENTLY DELETE files in '{target_dir}'.")
        print("This action CANNOT be undone.")
        confirm = input("Are you absolutely sure? Type 'yes' to proceed: ").strip().lower()
        if confirm != 'yes':
            print("Deletion aborted by user.")
            sys.exit()

    print("\n--- Summary ---")
    print(f"Directory to Clean: {target_dir}")
    print(f"Recursive Scan:     {'Yes' if is_recursive else 'No'}")
    print(f"Looking for '*{lonely_ext}' files that do NOT have a matching '*{main_ext}' file.")
    print(f"Action to Perform:  {action.upper()}")
    print("-----------------\n")

    try:
        input("Press Enter to begin, or Ctrl+C to cancel.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        return
        
    process_lonely_files(target_dir, main_ext, lonely_ext, action, is_recursive)

if __name__ == "__main__":
    main()