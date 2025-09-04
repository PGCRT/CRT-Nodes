"""
A script to download images from 500px.com based on a search term OR a
photographer's page URL.

The broken gallery feature has been removed for stability. This script now
focuses on the two reliable search methods.
"""

import os
import sys
import json
import re
import subprocess
import traceback

# --- Dependency Checker ---
def check_and_install_dependencies():
    """Checks for the 'requests' library and offers to install it if missing."""
    try:
        __import__('requests')
        print("[✓] 'requests' library is installed.")
        return True
    except ImportError:
        print("[!] The 'requests' library is required but not found.")
        answer = input("Do you want to try and install it now? (y/n): ").strip().lower()
        if answer == 'y':
            try:
                print("Installing requests...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
                print("[✓] 'requests' installed successfully. Please run the script again to use it.")
            except Exception as e:
                print(f"[✗] Error installing: {e}\nPlease install manually: pip install requests")
        else:
            print("Installation declined. The script cannot continue.")
        return False

if not check_and_install_dependencies():
    input("\nDependency issue found. Press Enter to close...")
    sys.exit()

import requests

def sanitize_filename(name):
    """Removes invalid characters from a string to make it a valid filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'\s+', '_', name)
    return name

def get_image_url(photo_data, quality_level):
    """Parses the photo data to find the URL for the desired image quality."""
    quality_map = {'high': 6, 'medium': 4, 'low': 3}
    target_size_id = quality_map.get(quality_level, 4)
    for image in photo_data.get('images', []):
        if image.get('size') == target_size_id:
            return image.get('https_url')
    if photo_data.get('images'):
        return photo_data['images'][0].get('https_url')
    return None

def get_initial_data_from_url(url, session):
    """
    Fetches a page's HTML and extracts the '__NEXT_DATA__' JSON block.
    This is the key to getting the internal IDs needed for API calls.
    """
    try:
        print(f"Step 1: Fetching initial page data from {url}...")
        response = session.get(url, timeout=15)
        response.raise_for_status()
        
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text)
        if match:
            json_data = json.loads(match.group(1))
            print("Successfully extracted page data block.")
            return json_data.get('props', {}).get('pageProps', {})
        else:
            print("[CRITICAL ERROR] Could not find the '__NEXT_DATA__' block on the page.")
            print("The website may be blocking script access or its structure has changed.")
            return None
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to fetch the page: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Failed to parse data from the page: {e}")
    return None

def scrape_500px(session, base_params, num_to_download, quality, output_folder):
    """
    Main function to scrape 500px, handle pagination, and download images.
    It uses a pre-configured session object.
    """
    os.makedirs(output_folder, exist_ok=True)
    api_url = "https://api.500px.com/v1/photos/search"
    downloaded_count = 0
    page = 1
    
    while downloaded_count < num_to_download:
        params = base_params.copy()
        params.update({
            'page': page,
            'rpp': min(50, num_to_download - downloaded_count),
            'image_size[]': [3, 4, 6]
        })
        
        try:
            print(f"Requesting page {page} from API...")
            response = session.get(api_url, params=params, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"\n[CRITICAL ERROR] Network request failed: {e}"); break

        data = response.json()
        photos = data.get('photos', [])
        
        if not photos:
            print("No more photos found."); break
            
        print(f"Found {len(photos)} photos on this page.")
        for photo in photos:
            if downloaded_count >= num_to_download: break
            
            image_url = get_image_url(photo, quality)
            if not image_url: continue

            photo_id = photo.get('id', 'unknown')
            photo_username = photo.get('user', {}).get('username', 'anonymous')
            filename = f"{sanitize_filename(photo_username)}_{photo_id}.jpg"
            filepath = os.path.join(output_folder, filename)
            
            if os.path.exists(filepath):
                print(f"  [SKIPPED] {filename}"); continue

            print(f"  Downloading image {downloaded_count + 1}/{num_to_download}: {filename}")
            try:
                img_response = session.get(image_url, stream=True, timeout=15)
                img_response.raise_for_status()
                with open(filepath, 'wb') as f:
                    for chunk in img_response.iter_content(chunk_size=8192): f.write(chunk)
                downloaded_count += 1
            except requests.exceptions.RequestException as e:
                print(f"    [ERROR] Failed to download {filename}: {e}")
        page += 1

    print(f"\n--- Scrape Complete ---")
    print(f"Successfully downloaded {downloaded_count} image(s) to '{output_folder}'.")

def main():
    """Handles all user interaction."""
    print("--- 500px.com Image Scraper ---")
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })
    
    search_mode, base_params, default_folder_name = None, None, ""

    while search_mode not in ['term', 'photographer']:
        print("\nHow would you like to search?")
        print("  1. By a search term")
        print("  2. By a photographer's page URL")
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice == '1': search_mode = 'term'
        elif choice == '2': search_mode = 'photographer'
        else: print("Invalid choice.")

    if search_mode == 'term':
        query = input("Enter the search term: ").strip()
        if not query: print("Search term cannot be empty."); return
        base_params = {'type': 'search', 'term': query, 'sort': 'highest_rating'}
        default_folder_name = query
    
    elif search_mode == 'photographer':
        url = input("Enter the photographer's full page URL: ").strip()
        initial_data = get_initial_data_from_url(url, session)
        if not initial_data:
            print("[ERROR] Could not process the provided URL."); return
        
        user = initial_data.get('user', {})
        user_id, username = user.get('id'), user.get('username')
        if not user_id or not username:
            print("[ERROR] Could not find user ID and username in page data."); return
        
        print(f"Found User ID: {user_id} for {username}")
        base_params = {'type': 'user_profile_photos', 'term': username, 'sort': 'created_at'}
        default_folder_name = username

    if not base_params:
        print("Failed to configure API parameters."); return

    while True:
        try:
            num_str = input("How many images do you want to download? (e.g., 20): ").strip()
            num_to_download = int(num_str)
            if num_to_download > 0: break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a whole number.");
            
    quality = ""
    while quality not in ['high', 'medium', 'low']:
        choice = input("Choose image quality (1: High, 2: Medium, 3: Low, default: 2): ").strip() or '2'
        if choice == '1': quality = 'high'
        elif choice == '2': quality = 'medium'
        elif choice == '3': quality = 'low'
        
    output_folder = input(f"Enter output folder name (default: '{sanitize_filename(default_folder_name)}'): ").strip() or sanitize_filename(default_folder_name)
    
    print("\n--- Summary ---")
    print(f"Search Mode: {search_mode.replace('_', ' ').capitalize()}")
    print(f"Output Folder: './{output_folder}/'")
    print("-----------------\n")

    input("Press Enter to begin, or Ctrl+C to cancel.")

    scrape_500px(session, base_params, num_to_download, quality, output_folder)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n--- AN UNEXPECTED ERROR OCCURRED ---")
        traceback.print_exc()
    finally:
        print("\nScript has finished or crashed.")
        input("Press Enter to close the console...")