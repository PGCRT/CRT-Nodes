"""
A script to download media (images and videos) from Reddit.

This script can scrape by subreddit, user, or keyword search and can filter
by media type. It is designed to be robust and handle various post formats.

Fix: Correctly constructs gallery image URLs to download full-resolution
originals instead of previews, preventing 403 Forbidden errors.
"""

import os
import sys
import json
import re
import subprocess
import traceback
import shutil

# --- Dependency Checker ---
def check_and_install_dependencies():
    """Checks for required dependencies and offers to install them."""
    try:
        __import__('requests')
        print("[✓] 'requests' library is installed.")
    except ImportError:
        print("[!] The 'requests' library is required but not found.")
        answer = input("Do you want to try and install it now? (y/n): ").strip().lower()
        if answer == 'y':
            try:
                print("Installing requests...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
                print("[✓] 'requests' installed successfully. Please run the script again.")
            except Exception as e:
                print(f"[✗] Error installing: {e}\nPlease install manually: pip install requests")
            return False
        else:
            print("Installation declined. Script cannot continue.")
            return False
            
    if not shutil.which("yt-dlp"):
        print("\n[!] CRITICAL: 'yt-dlp' is not installed or not in your system's PATH.")
        print("    This script requires yt-dlp to download videos.")
        print("    Please download it from: https://github.com/yt-dlp/yt-dlp")
        print("    And place the executable in a folder that is in your system's PATH.")
        return False
    else:
        print("[✓] 'yt-dlp' is installed and accessible.")

    return True

if not check_and_install_dependencies():
    input("\nDependency issue found. Press Enter to close...")
    sys.exit()

import requests

def sanitize_filename(name):
    """Removes invalid characters and shortens a string to make a valid filename."""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    name = re.sub(r'\s+', '_', name)
    return name[:150]

def download_image(session, url, filepath):
    """Downloads a single image from a URL."""
    try:
        response = session.get(url, stream=True, timeout=15)
        response.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.exceptions.RequestException as e:
        print(f"    [ERROR] Failed to download image {url}: {e}")
        return False

def download_video_with_yt_dlp(post_url, filepath):
    """Uses yt-dlp to download a video, which handles audio merging."""
    try:
        command = [
            'yt-dlp',
            '--quiet', '--no-warnings',
            '-o', filepath,
            post_url
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        error_message = e.stderr.decode('utf-8', errors='ignore').strip().splitlines()[-1]
        print(f"    [ERROR] yt-dlp failed: {error_message}")
        return False
    except Exception as e:
        print(f"    [ERROR] An unexpected error occurred with yt-dlp: {e}")
        return False

def scrape_reddit(session, base_url, base_params, num_to_download, media_type, output_folder):
    """Main function to scrape Reddit, handling pagination and different media types."""
    os.makedirs(output_folder, exist_ok=True)
    downloaded_count = 0
    after = None

    while downloaded_count < num_to_download:
        params = base_params.copy()
        params['limit'] = min(100, (num_to_download - downloaded_count) * 2 + 10)
        if after:
            params['after'] = after
        
        print(f"\nRequesting data from Reddit (after={after})...")
        try:
            response = session.get(base_url, params=params, timeout=15)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[CRITICAL ERROR] Network request failed: {e}"); break
        
        data = response.json()
        posts = data.get('data', {}).get('children', [])
        
        if not posts:
            print("No more posts found."); break

        for post in posts:
            if downloaded_count >= num_to_download: break
            
            post_data = post['data']
            title = post_data.get('title', '')
            post_id = post_data.get('id', '')
            permalink = f"https://www.reddit.com{post_data.get('permalink')}"

            if post_data.get('is_gallery'):
                if media_type in ('images', 'both'):
                    gallery_items = post_data.get('media_metadata', {})
                    for i, (item_id, item_data) in enumerate(gallery_items.items()):
                        if downloaded_count >= num_to_download: break
                        
                        # --- THIS IS THE FIX ---
                        # Instead of using the preview URL, we construct the original URL.
                        # The original image is hosted on i.redd.it, not preview.redd.it.
                        file_ext = item_data.get('m', 'image/jpeg').split('/')[-1]
                        img_url = f"https://i.redd.it/{item_id}.{file_ext}"
                        # -----------------------
                        
                        filename = f"{sanitize_filename(title)}_{post_id}_{i+1}.{file_ext}"
                        filepath = os.path.join(output_folder, filename)
                        
                        if os.path.exists(filepath):
                            print(f"[SKIPPED] {filename}"); continue
                            
                        print(f"  Downloading gallery image {downloaded_count+1}/{num_to_download}: {filename}")
                        if download_image(session, img_url, filepath):
                            downloaded_count += 1
                else: continue

            elif post_data.get('is_video'):
                if media_type in ('videos', 'both'):
                    filename = f"{sanitize_filename(title)}_{post_id}.mp4"
                    filepath = os.path.join(output_folder, filename)
                    
                    if os.path.exists(filepath):
                        print(f"[SKIPPED] {filename}"); continue
                        
                    print(f"  Downloading video {downloaded_count+1}/{num_to_download}: {filename}")
                    if download_video_with_yt_dlp(permalink, filepath):
                        downloaded_count += 1
                else: continue

            else:
                if media_type in ('images', 'both'):
                    url = post_data.get('url_overridden_by_dest')
                    if not url or not any(url.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                        continue
                        
                    file_ext = url.split('.')[-1]
                    filename = f"{sanitize_filename(title)}_{post_id}.{file_ext}"
                    filepath = os.path.join(output_folder, filename)
                    
                    if os.path.exists(filepath):
                        print(f"[SKIPPED] {filename}"); continue
                        
                    print(f"  Downloading image {downloaded_count+1}/{num_to_download}: {filename}")
                    if download_image(session, url, filepath):
                        downloaded_count += 1
                else: continue

        after = data.get('data', {}).get('after')
        if not after:
            print("Reached the end of the listing."); break
    
    print(f"\n--- Scrape Complete ---")
    print(f"Successfully downloaded {downloaded_count} media file(s) to '{output_folder}'.")

def main():
    """Handles all user interaction."""
    print("--- Reddit Media Scraper ---")
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
    
    search_mode, base_url, base_params, default_folder_name = None, None, {}, ""

    while search_mode not in ['subreddit', 'user', 'keyword']:
        print("\nHow would you like to search?")
        print("  1. By Subreddit (e.g., r/itookapicture)")
        print("  2. By User (e.g., u/spez)")
        print("  3. By Keyword search")
        choice = input("Enter your choice (1, 2, or 3): ").strip()
        if choice == '1': search_mode = 'subreddit'
        elif choice == '2': search_mode = 'user'
        elif choice == '3': search_mode = 'keyword'
        else: print("Invalid choice.")

    if search_mode == 'subreddit':
        query = input("Enter subreddit name (without r/): ").strip()
        if not query: print("Name cannot be empty."); return
        base_url = f"https://www.reddit.com/r/{query}.json"
        default_folder_name = f"r_{query}"
    
    elif search_mode == 'user':
        query = input("Enter username (without u/): ").strip()
        if not query: print("Name cannot be empty."); return
        base_url = f"https://www.reddit.com/user/{query}.json"
        default_folder_name = f"u_{query}"

    elif search_mode == 'keyword':
        query = input("Enter search keyword(s): ").strip()
        if not query: print("Keyword cannot be empty."); return
        base_url = "https://www.reddit.com/search.json"
        base_params['q'] = query
        default_folder_name = sanitize_filename(query)

    media_type = ""
    while media_type not in ['images', 'videos', 'both']:
        choice = input("What do you want to download? (1: Images, 2: Videos, 3: Both, default: 3): ").strip() or '3'
        if choice == '1': media_type = 'images'
        elif choice == '2': media_type = 'videos'
        elif choice == '3': media_type = 'both'

    while True:
        try:
            num_str = input("How many files do you want to download? (e.g., 25): ").strip()
            num_to_download = int(num_str)
            if num_to_download > 0: break
            print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a whole number.");
            
    output_folder = input(f"Enter output folder name (default: '{default_folder_name}'): ").strip() or default_folder_name
    
    print("\n--- Summary ---")
    print(f"Search Mode: {search_mode.capitalize()}")
    if search_mode != 'keyword':
        print(f"Target: {default_folder_name}")
    else:
        print(f"Keyword: {base_params['q']}")
    print(f"Media Type: {media_type.capitalize()}")
    print(f"Number to Download: {num_to_download}")
    print(f"Output Folder: './{output_folder}/'")
    print("-----------------\n")

    input("Press Enter to begin, or Ctrl+C to cancel.")

    scrape_reddit(session, base_url, base_params, num_to_download, media_type, output_folder)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n--- AN UNEXPECTED ERROR OCCURRED ---")
        traceback.print_exc()
    finally:
        print("\nScript has finished or crashed.")
        input("Press Enter to close the console...")