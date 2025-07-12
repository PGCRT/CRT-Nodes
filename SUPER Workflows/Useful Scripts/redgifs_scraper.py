"""
A script to download videos from RedGifs.com based on a search term.

This script now correctly handles all user requests:
- It forces the search term to lowercase to prevent API errors.
- It asks the user to try another term if a search yields no results.
- It loops the entire process, allowing for multiple scrapes without restarting.
- It uses 'cloudscraper' to bypass Cloudflare protection.
"""

import os
import sys
import json
import re
import subprocess
import traceback

# --- Dependency Checker ---
def check_and_install_dependencies():
    """Checks for the 'cloudscraper' library and offers to install it."""
    try:
        __import__('cloudscraper')
        print("[✓] 'cloudscraper' library is installed.")
        return True
    except ImportError:
        print("[!] The 'cloudscraper' library is required but not found.")
        answer = input("Do you want to try and install it now? (y/n): ").strip().lower()
        if answer == 'y':
            try:
                print("Installing cloudscraper...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudscraper"])
                print("[✓] 'cloudscraper' installed successfully. Please run the script again to use it.")
            except Exception as e:
                print(f"[✗] Error installing: {e}\nPlease install manually: pip install cloudscraper")
        else:
            print("Installation declined. The script cannot continue.")
        return False

if not check_and_install_dependencies():
    input("\nDependency issue found. Press Enter to close...")
    sys.exit()

import cloudscraper

def sanitize_filename(name):
    """Removes invalid characters and shortens a string to make a valid filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'\s+', '_', name)
    return name[:150]

def get_auth_token(scraper):
    """Performs the initial authentication to get a temporary guest token."""
    auth_url = "https://api.redgifs.com/v2/auth/temporary"
    try:
        print("Step 1: Authenticating with RedGifs API...")
        response = scraper.get(auth_url, timeout=15)
        response.raise_for_status()
        data = response.json()
        token = data.get('token')
        if token:
            print("Authentication successful.")
            return token
        else:
            print("[CRITICAL ERROR] Authentication token not found in API response.")
            return None
    except Exception as e:
        print(f"[CRITICAL ERROR] Could not get authentication token: {e}")
        return None

def download_file(scraper, url, filepath):
    """Downloads a single file from a URL, used for the MP4 videos."""
    try:
        response = scraper.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    [ERROR] Failed to download file {os.path.basename(filepath)}: {e}")
        return False

def scrape_redgifs(scraper, search_term, num_to_download, output_folder):
    """Main function to scrape RedGifs, using a pre-authenticated scraper."""
    os.makedirs(output_folder, exist_ok=True)
    
    api_url = "https://api.redgifs.com/v2/gifs/search"
    downloaded_count = 0
    page = 1
    
    while downloaded_count < num_to_download:
        params = {
            'search_text': search_term.lower(), # <<< THE FIX IS HERE: Force to lowercase
            'count': 80,
            'page': page,
            'order': 'trending',
            'type': 'g'
        }
        
        print(f"\nRequesting page {page} for '{search_term}'...")
        try:
            response = scraper.get(api_url, params=params, timeout=15)
            response.raise_for_status()
        except Exception as e:
            print(f"[CRITICAL ERROR] API request failed: {e}")
            return False # Indicate failure
        
        data = response.json()
        gifs = data.get('gifs', [])
        
        if not gifs:
            print("No results found for this search term.");
            # If it's the very first page and we got nothing, it's a failed search.
            if page == 1:
                return False
            break
            
        print(f"Found {len(gifs)} potential files on this page.")
        for gif_data in gifs:
            if downloaded_count >= num_to_download: break
            
            hd_url = gif_data.get('urls', {}).get('hd')
            if not hd_url: continue
            
            username = gif_data.get('userName', 'anonymous')
            gif_id = gif_data.get('id', 'unknown')
            
            filename = f"{sanitize_filename(username)}_{gif_id}.mp4"
            filepath = os.path.join(output_folder, filename)
            
            if os.path.exists(filepath):
                print(f"  [SKIPPED] {filename}"); continue
                
            print(f"  Downloading video {downloaded_count + 1}/{num_to_download}: {filename}")
            if download_file(scraper, hd_url, filepath):
                downloaded_count += 1
                
        page += 1
    
    print(f"\n--- Scrape Complete ---")
    print(f"Successfully downloaded {downloaded_count} video(s) to '{output_folder}'.")
    return True # Indicate success

def main():
    """Handles all user interaction."""
    print("--- RedGifs.com Scraper ---")
    
    scraper = cloudscraper.create_scraper()
    token = get_auth_token(scraper)
    if not token:
        print("Aborting due to authentication failure."); return
    scraper.headers.update({"Authorization": f"Bearer {token}"})
    
    while True: # Loop for entering a search term until it works
        search_term = input("\nEnter the search term (e.g., 'car exhaust'): ").strip()
        if not search_term:
            print("Search term cannot be empty."); continue
            
        while True:
            try:
                num_str = input("How many files do you want to download? (e.g., 10): ").strip()
                num_to_download = int(num_str)
                if num_to_download > 0: break
                print("Please enter a positive number.")
            except ValueError:
                print("Invalid input. Please enter a whole number.");
                
        default_folder_name = sanitize_filename(search_term)
        output_folder = input(f"Enter output folder name (default: '{default_folder_name}'): ").strip() or default_folder_name
        
        print("\n--- Summary ---")
        print(f"Search Term: '{search_term}'")
        print(f"Files to Download: {num_to_download}")
        print(f"Output Folder: './{output_folder}/'")
        print("-----------------\n")

        input("Press Enter to begin, or Ctrl+C to cancel.")

        success = scrape_redgifs(scraper, search_term, num_to_download, output_folder)
        
        if not success:
            print("\nThe last search failed or returned no results.")
            # This directly addresses your request to ask for another word
            retry_choice = input("Would you like to try a different search term? (y/n): ").strip().lower()
            if retry_choice == 'y':
                continue # Go to the top of the search term loop
            else:
                break # Exit the main loop
        else:
            break # Exit the main loop if scrape was successful

if __name__ == "__main__":
    while True: # Main loop for the entire script process
        try:
            main()
        except Exception:
            print("\n--- AN UNEXPECTED ERROR OCCURRED ---")
            traceback.print_exc()
        
        # This directly addresses your request to loop the script
        print("\n-------------------------------------------")
        rerun_choice = input("Scrape again with a new search? (y/n): ").strip().lower()
        if rerun_choice != 'y':
            break # Exit the program loop
            
    print("\nScript has finished.")
    input("Press Enter to close the console...")