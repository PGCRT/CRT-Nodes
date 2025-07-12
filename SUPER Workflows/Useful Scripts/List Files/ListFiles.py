"""
A powerful and interactive script for listing the contents of a directory.

This script asks the user for all necessary options:
- The target directory to scan.
- Whether to scan recursively into subdirectories.
- What to list: files, folders, or both.
- The format of the output: simple names or full paths.
- The name of the output text file.
"""

import os
import sys

def generate_file_list(directory, recursive, list_type, path_format, output_filename):
    """
    Generates a list of files and/or folders and saves it to a text file.
    
    Args:
        directory (str): The directory to scan.
        recursive (bool): True to scan subdirectories, False otherwise.
        list_type (str): 'files', 'folders', or 'both'.
        path_format (str): 'name_only' or 'full_path'.
        output_filename (str): The name of the file to save the list to.
    """
    results = []
    print("\nScanning directory... please wait.")

    if recursive:
        # Use os.walk for an efficient recursive scan
        for root, dirs, files in os.walk(directory):
            # Decide what to add based on user's choice
            items_to_add = []
            if list_type in ('files', 'both'):
                items_to_add.extend(files)
            if list_type in ('folders', 'both'):
                items_to_add.extend(dirs)
            
            # Format the path based on user's choice
            for item in items_to_add:
                if path_format == 'full_path':
                    results.append(os.path.join(root, item))
                else: # name_only
                    results.append(item)
    else:
        # Non-recursive scan of the top-level directory only
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            # Check if the item matches the type the user wants to list
            is_file = os.path.isfile(full_path)
            is_dir = os.path.isdir(full_path)

            if (list_type == 'files' and is_file) or \
               (list_type == 'folders' and is_dir) or \
               (list_type == 'both'):
                
                if path_format == 'full_path':
                    results.append(full_path)
                else: # name_only
                    results.append(item)
    
    # Write the collected results to the output file
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            for line in sorted(results):
                f.write(line + '\n')
        print(f"\n[✓] Success! List of {len(results)} items has been saved to '{output_filename}'")
    except IOError as e:
        print(f"\n[✗] Error: Could not write to file '{output_filename}'. Reason: {e}")

def main():
    """Main function to handle all user interaction."""
    print("--- Advanced File Lister ---")
    
    # 1. Get Target Directory
    target_dir = ""
    while not os.path.isdir(target_dir):
        user_input = input(f"Enter the directory to list (press Enter for current folder): ").strip()
        target_dir = user_input or os.getcwd()
        if not os.path.isdir(target_dir):
            print(f"Error: Directory not found at '{target_dir}'. Please try again.")
    
    # 2. Get Recursive Option
    recursive_choice = input("Include subdirectories (recursive scan)? (y/n, default: n): ").strip().lower()
    is_recursive = recursive_choice == 'y'

    # 3. Get List Type
    list_type = ""
    while list_type not in ['files', 'folders', 'both']:
        choice = input("What do you want to list? (1: Files only, 2: Folders only, 3: Both, default: 1): ").strip()
        if choice == '1' or not choice:
            list_type = 'files'
        elif choice == '2':
            list_type = 'folders'
        elif choice == '3':
            list_type = 'both'
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # 4. Get Path Format
    path_format = ""
    while path_format not in ['name_only', 'full_path']:
        choice = input("How to format paths? (1: Name only, 2: Full path, default: 1): ").strip()
        if choice == '1' or not choice:
            path_format = 'name_only'
        elif choice == '2':
            path_format = 'full_path'
        else:
            print("Invalid choice. Please enter 1 or 2.")
            
    # 5. Get Output Filename
    output_file = input("Enter the name for the output file (default: file_list.txt): ").strip()
    if not output_file:
        output_file = "file_list.txt"
    if not output_file.lower().endswith('.txt'):
        output_file += '.txt'

    # --- Summary and Confirmation ---
    print("\n--- Summary ---")
    print(f"Directory:    {target_dir}")
    print(f"Recursive:    {'Yes' if is_recursive else 'No'}")
    print(f"List Content: {list_type.capitalize()}")
    print(f"Path Format:  {path_format.replace('_', ' ').capitalize()}")
    print(f"Output File:  {output_file}")
    print("-----------------\n")

    try:
        input("Press Enter to start, or Ctrl+C to cancel.")
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
        sys.exit()

    generate_file_list(target_dir, is_recursive, list_type, path_format, output_file)

if __name__ == "__main__":
    main()