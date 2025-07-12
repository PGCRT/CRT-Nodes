#!/usr/bin/env python3
"""
YouTube Shorts Downloader Script
Downloads YouTube Shorts videos to the current directory.
"""

import os
import sys
import subprocess
import re

def check_dependencies():
    """Check if yt-dlp and ffmpeg are installed."""
    dependencies_ok = True
    
    # Check yt-dlp
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        print("✅ yt-dlp found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("yt-dlp not found. Installing...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'yt-dlp'], check=True)
            print("✅ yt-dlp installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install yt-dlp. Please install it manually: pip install yt-dlp")
            dependencies_ok = False
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        print("✅ ffmpeg found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  ffmpeg not found. This is needed for format conversion.")
        print("Please install ffmpeg:")
        print("- Windows: Download from https://ffmpeg.org/download.html")
        print("- Mac: brew install ffmpeg")
        print("- Linux: sudo apt install ffmpeg (Ubuntu/Debian) or equivalent")
        print("The script will still work but may not convert formats properly.")
    
    return dependencies_ok

def validate_youtube_url(url):
    """Validate if the URL is a valid YouTube URL."""
    youtube_patterns = [
        r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'https?://(?:www\.)?youtube\.com/shorts/[\w-]+',
        r'https?://youtu\.be/[\w-]+',
        r'https?://(?:www\.)?youtube\.com/embed/[\w-]+',
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False

def download_video(url, output_path='.'):
    """Download the YouTube video in highest quality and convert to MP4 if needed."""
    try:
        print(f"Downloading video from: {url}")
        print("Fetching highest quality available...")
        
        # Command to download best quality and convert to MP4
        cmd = [
            'yt-dlp',
            '--format', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/bestvideo+bestaudio/best[ext=mp4]/best',
            '--merge-output-format', 'mp4',  # Ensure final output is MP4
            '--output', os.path.join(output_path, '%(title)s.%(ext)s'),
            '--no-playlist',
            '--embed-thumbnail',  # Embed thumbnail in video
            '--add-metadata',     # Add metadata to file
            '--write-description', # Save description as .description file
            '--write-info-json',  # Save video info as .info.json file
            url
        ]
        
        print("Please wait... (This may take a while for high quality videos)")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Download completed successfully!")
            print("📱 Video saved in MP4 format with highest available quality")
            
            # Extract filename from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Destination:' in line or 'Merging formats into' in line:
                    print(f"📁 File saved in current directory")
                    break
                elif 'has already been downloaded' in line:
                    print(f"📁 File was already downloaded")
                    break
        else:
            print("❌ High quality download failed, trying fallback options...")
            
            # Fallback format options for problematic videos
            fallback_formats = [
                'best[ext=mp4]/best',
                'bestvideo[height<=1080]+bestaudio/best[height<=1080]',
                'bestvideo[height<=720]+bestaudio/best[height<=720]',
                'worst[ext=mp4]/worst'
            ]
            
            success = False
            for format_option in fallback_formats:
                fallback_cmd = [
                    'yt-dlp',
                    '--format', format_option,
                    '--merge-output-format', 'mp4',
                    '--output', os.path.join(output_path, '%(title)s.%(ext)s'),
                    '--no-playlist',
                    url
                ]
                
                print(f"Trying format: {format_option}")
                fallback_result = subprocess.run(fallback_cmd, capture_output=True, text=True)
                
                if fallback_result.returncode == 0:
                    print("✅ Fallback download completed successfully!")
                    print("📱 Video saved in MP4 format")
                    success = True
                    break
            
            if not success:
                print("❌ All download attempts failed!")
                print("Trying to get more information about the video...")
                
                # Get video info for debugging
                info_cmd = ['yt-dlp', '--dump-json', url]
                info_result = subprocess.run(info_cmd, capture_output=True, text=True)
                
                if info_result.returncode == 0:
                    import json
                    try:
                        video_info = json.loads(info_result.stdout)
                        print(f"Video Title: {video_info.get('title', 'Unknown')}")
                        print(f"Duration: {video_info.get('duration', 'Unknown')} seconds")
                        print(f"View Count: {video_info.get('view_count', 'Unknown')}")
                        
                        # Try one more basic download
                        print("\nTrying basic download without format specification...")
                        basic_cmd = ['yt-dlp', '--output', os.path.join(output_path, '%(title)s.%(ext)s'), url]
                        basic_result = subprocess.run(basic_cmd, capture_output=True, text=True)
                        
                        if basic_result.returncode == 0:
                            print("✅ Basic download successful!")
                        else:
                            print("❌ All attempts failed!")
                            print(f"Error: {basic_result.stderr}")
                    except json.JSONDecodeError:
                        print("❌ Could not parse video information")
                        print(f"Raw error: {result.stderr}")
                else:
                    print(f"❌ Could not get video information. Error: {info_result.stderr}")
            
    except Exception as e:
        print(f"❌ An error occurred: {e}")

def main():
    """Main function to run the YouTube Shorts downloader."""
    print("🎬 YouTube Shorts Downloader - Highest Quality MP4")
    print("=" * 50)
    
    # Check if dependencies are available
    if not check_dependencies():
        print("\n⚠️  Some dependencies are missing but the script will still try to work.")
        input("Press Enter to continue...")
    
    while True:
        # Get URL from user
        url = input("\n📹 Enter YouTube Shorts URL (or 'quit' to exit): ").strip()
        
        if url.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not url:
            print("❌ Please enter a valid URL.")
            continue
        
        # Validate URL
        if not validate_youtube_url(url):
            print("❌ Invalid YouTube URL. Please enter a valid YouTube or YouTube Shorts link.")
            continue
        
        # Download the video in highest quality
        download_video(url)
        
        # Ask if user wants to download another video
        continue_choice = input("\n🔄 Download another video? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("👋 Goodbye!")
            break

if __name__ == "__main__":
    main()