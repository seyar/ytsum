import urllib.parse
import subprocess
import re

def clean_youtube_url(url):
    """Clean and validate YouTube URL or video ID"""
    # Extract video ID from various URL formats
    video_id = None

    # Unescape URL first
    url = urllib.parse.unquote(url.replace('\\', ''))

    # Handle full URLs
    if url.startswith(('http://', 'https://')):
        try:
            parsed = urllib.parse.urlparse(url)
            if 'youtu.be' in parsed.netloc.lower():
                video_id = parsed.path.strip('/')
            else:
                params = urllib.parse.parse_qs(parsed.query)
                video_id = params['v'][0]
        except:
            pass

    # Handle partial URLs
    elif 'youtube.com' in url.lower() or 'youtu.be' in url.lower():
        try:
            if 'youtu.be' in url.lower():
                video_id = url.split('youtu.be/')[-1].split('?')[0]
            else:
                video_id = url.split('v=')[1].split('&')[0]
        except:
            pass

    # Handle direct video ID
    else:
        video_id = url.strip('/')

    # Validate video ID format (11 characters, alphanumeric and -_)
    if not video_id or not re.match(r'^[A-Za-z0-9_-]{11}$', video_id):
        raise ValueError(f"Invalid YouTube video ID: {video_id}")

    # Check if video exists
    try:
        result = subprocess.run([
            'yt-dlp',
            '--simulate',
            '--cookies', '/app/data/www.youtube.com_cookies.txt',
            '--no-warnings',
            '--no-playlist',
            f'https://www.youtube.com/watch?v={video_id}'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if "Video unavailable" in error_msg:
                raise ValueError(f"Video {video_id} is unavailable or has been removed")
            elif "Private video" in error_msg:
                raise ValueError(f"Video {video_id} is private")
            else:
                raise ValueError(f"Error accessing video: {error_msg}")
    except subprocess.CalledProcessError:
        raise ValueError(f"Could not verify video availability")

    return f"https://www.youtube.com/watch?v={video_id}"

def get_video_id(clean_url):
    return clean_url.split('v=')[1].split('&')[0]
