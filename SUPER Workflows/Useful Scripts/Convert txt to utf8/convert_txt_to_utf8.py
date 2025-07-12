"""
A script to convert text files from various common encodings to UTF-8.

This script is fully interactive and prioritizes safety:
- It asks the user to specify a directory to scan.
- It asks for the file extensions to target (e.g., txt, srt, vtt).
- It asks the user if they want to create a backup (.bak) of each file before
  converting it, which is the recommended and default behavior.
- It provides a summary of actions before starting and requires confirmation.
"""

import os
import sys
import shutil

def convert_files_to_utf8(directory, extensions, create_backup=True):
    """
    Scans a directory for specified file types and converts them to UTF-8.
    
    Args:
        directory (str): The path to the directory to scan.
        extensions (tuple): A tuple of file extensions to process (e.g., ('.txt', '.srt')).
        create_backup (bool): If True, creates a .bak copy of each file before converting.
    """
    # A list of common encodings to try if UTF-8 fails.
    # For many Western languages, 'cp1252' (Windows) or 'latin-1' are common.
    # For more complex scenarios, a library like 'chardet' would be needed,
    # but we will try a few common ones here.
    fallback_encodings = ['cp1252', 'latin-1', 'iso-8859-1']
    
    files_checked = 0
    files_converted = 0

    print("\nStarting scan...")
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for filename in files:
            # Check if the file has one of the desired extensions
            if filename.lower().endswith(extensions):
                file_path = os.path.join(root, filename)
                files_checked += 1
                
                try:
                    # First, try to open as UTF-8. If this works, the file is already compliant.
                    with open(file_path, 'r', encoding='utf-8') as f:
                        f.read() # Try to read the whole file to trigger an error if it's not UTF-8
                    print(f"[✓] Already UTF-8: {file_path}")
                    continue
                except UnicodeDecodeError:
                    # This is what we expect for files that need conversion.
                    print(f"[!] Needs conversion: {file_path}")
                except Exception as e:
                    print(f"[✗] Error reading {file_path}: {e}")
                    continue
                
                # If we get here, the file is not UTF-8. Try to read it with fallback encodings.
                content = None
                original_encoding = None
                for encoding in fallback_encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        original_encoding = encoding
                        break # Stop on the first one that works
                    except (UnicodeDecodeError, TypeError):
                        continue # Try the next encoding

                if content is None:
                    print(f"  [✗] FAILED: Could not decode {filename} with any fallback encoding. Skipping.")
                    continue

                # We have the content, now let's convert it.
                try:
                    # Create a backup if requested
                    if create_backup:
                        shutil.copy2(file_path, file_path + ".bak")
                        print(f"  [i] Backup created: {filename}.bak")
                    
                    # Write the content back in UTF-8, overwriting the original file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"  [✓] CONVERTED from '{original_encoding}' to 'UTF-8'")
                    files_converted += 1

                except Exception as e:
                    print(f"  [✗] FAILED to write new file: {e}")

    print("\n--- Conversion Report ---")
    print(f"Total files checked: {files_checked}")
    print(f"Total files converted: {files_converted}")
    print("-------------------------\n")


def main():
    """Main function to handle user interaction."""
    print("--- Text File to UTF-8 Converter ---")
    print("This script will find and convert text files to UTF-8 encoding.")
    
    # 1. Ask for the directory to scan
    target_dir = ""
    while not target_dir:
        default_dir = os.getcwd()
        user_input = input(f"Enter the directory to scan (press Enter for current: {default_dir}): ").strip()
        target_dir = user_input if user_input else default_dir
        if not os.path.isdir(target_dir):
            print(f"Error: Directory '{target_dir}' not found. Please try again.")
            target_dir = ""

    # 2. Ask for file extensions
    extensions_str = input("Enter file extensions to convert, separated by commas (e.g., txt,srt,vtt): ").strip().lower()
    if not extensions_str:
        extensions_tuple = ('.txt',) # Default to .txt if empty
    else:
        # Convert "txt, srt" to ('.txt', '.srt')
        extensions_tuple = tuple('.' + ext.strip() for ext in extensions_str.split(',') if ext.strip())

    # 3. Ask about creating backups
    backup_choice = input("Create a backup (.bak) of each original file before converting? (Y/n): ").strip().lower()
    create_backup = backup_choice != 'n' # Default to Yes

    # 4. Final confirmation
    print("\n--- Summary ---")
    print(f"Directory to Scan:    {target_dir}")
    print(f"File Extensions:      {', '.join(extensions_tuple)}")
    print(f"Create Backups:       {'Yes' if create_backup else 'No'}")
    if not create_backup:
        print("\nWARNING: You have chosen not to create backups. File modifications will be permanent.")
    print("-----------------\n")

    try:
        confirm = input("Press Enter to begin the conversion, or Ctrl+C to cancel.")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit()

    convert_files_to_utf8(target_dir, extensions_tuple, create_backup)


if __name__ == "__main__":
    main()