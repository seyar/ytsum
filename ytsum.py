#!/usr/bin/env python3
import sys
import os
import subprocess
import tempfile
from pathlib import Path
import json
import argparse
import urllib.parse
import ell
from anthropic import Anthropic
from colorama import init, Fore, Style
import replicate
import time

# Initialize colorama
init()

# Initialize Anthropic client and register with Ell
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print_error("ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)

claude_client = Anthropic()
ell.config.register_model("claude-3-5-sonnet-20241022", claude_client)

# Emoji constants
EMOJI_DOWNLOAD = "⬇️ "
EMOJI_TRANSCRIBE = "🎯 "
EMOJI_SUMMARY = "📝 "
EMOJI_SUCCESS = "✅ "
EMOJI_ERROR = "❌ "
EMOJI_SEARCH = "🔍 "
EMOJI_SAVE = "💾 "

def print_step(emoji, message, color=Fore.BLUE):
    """Print a step with emoji and color"""
    print(f"{color}{emoji}{message}{Style.RESET_ALL}")

def print_error(message):
    """Print error message in red with emoji"""
    print(f"{Fore.RED}{EMOJI_ERROR}{message}{Style.RESET_ALL}")

def print_success(message):
    """Print success message in green with emoji"""
    print(f"{Fore.GREEN}{EMOJI_SUCCESS}{message}{Style.RESET_ALL}")

def to_shorthand(text):
    """Convert text to shorthand format"""
    replacements = {
        'you': 'u',
        'are': 'r',
        'see': 'c',
        'for': '4',
        'to': '2',
        'too': '2',
        'two': '2',
        'four': '4',
        'be': 'b',
        'before': 'b4',
        'great': 'gr8',
        'thanks': 'thx',
        'thank you': 'ty',
        'because': 'bc',
        'people': 'ppl',
        'want': 'wnt',
        'love': 'luv',
        'okay': 'k',
        'yes': 'y',
        'no': 'n',
        'please': 'plz',
        'sorry': 'sry',
        'see you': 'cya',
        'I am': 'im',
        'i am': 'im',
        'good': 'gd',
        'right': 'rt',
        'later': 'l8r',
        'have': 'hv',
        'see you later': 'cul8r',
        'laughing': 'lol',
        'message': 'msg',
        'information': 'info',
        'about': 'abt',
        'awesome': 'awsm',
        'quickly': 'quick',
        'first': '1st',
        'second': '2nd',
        'third': '3rd',
    }
    
    # Convert to lowercase first
    result = text.lower()
    
    # Split into words, remove articles, and rejoin
    words = result.split()
    words = [w for w in words if w not in ['the', 'a', 'an']]
    result = ' '.join(words)
    
    # Apply other replacements
    for old, new in replacements.items():
        result = result.replace(old.lower(), new)
    
    return result

def clean_youtube_url(url_or_id):
    """Clean and normalize YouTube URL or video ID"""
    # Check if input is just a video ID
    if not any(domain in url_or_id for domain in ['youtube.com', 'youtu.be']):
        # Assume it's a video ID, construct full URL
        return f"https://www.youtube.com/watch?v={url_or_id}"
    
    # Handle full URL
    url = urllib.parse.unquote(url_or_id)
    url = url.replace('\\', '')
    try:
        if 'youtu.be' in url:
            video_id = url.split('/')[-1].split('?')[0]
        else:
            # Extract v parameter
            query = urllib.parse.urlparse(url).query
            params = urllib.parse.parse_qs(query)
            video_id = params['v'][0]
        
        return f"https://www.youtube.com/watch?v={video_id}"
    except:
        return url

def download_video(url, output_path):
    """Download audio using yt-dlp"""
    try:
        clean_url = clean_youtube_url(url)
        print_step(EMOJI_DOWNLOAD, "Downloading audio...")
        
        subprocess.run([
            'yt-dlp',
            '--output', output_path,
            '--format', 'ba[ext=m4a]',
            '--extract-audio',
            '--force-overwrites',
            clean_url
        ], check=True)
        return True
    except subprocess.CalledProcessError:
        print_error("Failed to download audio")
        return False

def get_language_code(language_name: str) -> str:
    """Convert language name to ISO 639-1 code using Claude"""
    
    @ell.simple(model="claude-3-5-sonnet-20241022", temperature=0.0, max_tokens=2)
    def get_code(lang: str) -> str:
        """You are an expert in language codes. Return only the ISO 639-1 code (2 letters) for the given language name.
        For example:
        - English -> en
        - Russian -> ru
        - Spanish -> es
        - Chinese -> zh
        - Japanese -> ja
        If unsure, return 'en' as fallback."""
        return f"Convert this language name to ISO 639-1 code: {lang}. No \`\`\` or \`\`\`python, no intro, no commentaries, only the code."
    
    try:
        code = get_code(language_name).strip().lower()
        # Validate it's a 2-letter code
        if len(code) == 2 and code.isalpha():
            return code
        return 'en'
    except:
        return 'en'

def get_youtube_subtitles(url, output_path, language="en"):
    """Try to download subtitles from YouTube using yt-dlp"""
    try:
        # Convert language name to code
        language_code = get_language_code(language)
        print_step(EMOJI_SEARCH, f"Searching for YouTube subtitles in {language} ({language_code})...")
        clean_url = clean_youtube_url(url)
        
        # Try to download subtitles directly with basic command
        result = subprocess.run([
            'yt-dlp',
            '--write-subs',
            '--sub-langs', language_code,
            '--skip-download',
            clean_url
        ], capture_output=True, text=True)
        
        # Look for the downloaded subtitle file in current directory
        if "Writing video subtitles to:" in result.stdout:
            # Extract the filename from yt-dlp output
            for line in result.stdout.splitlines():
                if "Writing video subtitles to:" in line:
                    subtitle_file = line.split("Writing video subtitles to:", 1)[1].strip()
                    if os.path.exists(subtitle_file):
                        print_success(f"Found subtitles!")
                        # Convert VTT to plain text
                        text = convert_vtt_to_text(subtitle_file)
                        txt_file = subtitle_file.replace('.vtt', '.txt')
                        with open(txt_file, 'w', encoding='utf-8') as f:
                            f.write(text)
                        return txt_file
        
        print_step(EMOJI_SEARCH, "No subtitles found, will transcribe audio...")
        return None
        
    except Exception as e:
        print_error(f"Failed to get subtitles: {e}")
        return None

def convert_vtt_to_text(vtt_file):
    """Convert VTT subtitles to plain text"""
    text = []
    with open(vtt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Skip VTT header
    start = 0
    for i, line in enumerate(lines):
        if line.strip() == "WEBVTT":
            start = i + 1
            break
    
    # Process subtitle content
    for line in lines[start:]:
        # Skip timing lines and empty lines
        if '-->' in line or not line.strip():
            continue
        # Add non-empty lines to text
        if line.strip():
            text.append(line.strip())
    
    return ' '.join(text)

def transcribe_with_fast_whisper(video_path):
    """Transcribe video using Faster Whisper"""
    try:
        from faster_whisper import WhisperModel
        
        print_step(EMOJI_TRANSCRIBE, "Transcribing with Fast Whisper...")
        model = WhisperModel("base", device="auto", compute_type="auto")
        
        segments, _ = model.transcribe(video_path)
        transcript = " ".join([segment.text for segment in segments])
        
        transcript_path = str(Path(video_path).with_suffix('.txt'))
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        return True
        
    except ImportError:
        print_error("Faster Whisper not found. Please install it with:")
        print(f"{Fore.YELLOW}pip install faster-whisper{Style.RESET_ALL}")
        return False
    except Exception as e:
        print_error(f"Fast transcription error: {e}")
        return False

def transcribe_with_replicate(video_path, language=None):
    """Transcribe video using Replicate's Incredibly Fast Whisper"""
    try:
        print_step(EMOJI_TRANSCRIBE, "Transcribing with Incredibly Fast Whisper...")
        
        # Convert audio to MP3 format
        mp3_path = convert_audio_format(video_path, 'mp3')
        if not mp3_path:
            print_error("Failed to convert audio to MP3")
            return False
        
        # Prepare input parameters
        input_params = {
            "audio": open(mp3_path, 'rb'),  # Send file directly
            "batch_size": 64,
        }
        
        if language:
            input_params["language"] = language.lower()
        
        # Run transcription
        output = replicate.run(
            "vaibhavs10/incredibly-fast-whisper:3ab86df6c8f54c11309d4d1f930ac292bad43ace52d10c80d87eb258b3c9f79c",
            input=input_params
        )
        
        if not output or "text" not in output:
            print_error("Invalid response from Replicate")
            return False
        
        # Write transcript to file
        transcript_path = os.path.splitext(video_path)[0] + '.txt'
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(output["text"])
        
        return True
        
    except Exception as e:
        print_error(f"Replicate transcription error: {e}")
        return False

def split_audio_into_chunks(input_path, chunk_size_mb=20):
    """Split audio file into chunks under specified size"""
    try:
        # Get file size in MB
        file_size = os.path.getsize(input_path) / (1024 * 1024)
        if file_size <= chunk_size_mb:
            return [input_path]
        
        # Calculate duration of each chunk
        duration_info = subprocess.run([
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ], capture_output=True, text=True)
        
        total_duration = float(duration_info.stdout.strip())  # Strip whitespace
        if total_duration <= 0:
            print_error("Invalid audio duration")
            return None
            
        # Calculate chunk duration (ensure it's at least 1 second)
        chunk_duration = max(1, int((chunk_size_mb / file_size) * total_duration))
        
        # Create chunks directory
        chunks_dir = os.path.join(os.path.dirname(input_path), "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        
        chunk_paths = []
        for i in range(0, int(total_duration), chunk_duration):
            chunk_path = os.path.join(chunks_dir, f"chunk_{i}.mp3")
            subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-y',  # Overwrite output
                '-ss', str(i),  # Start time
                '-t', str(chunk_duration),  # Duration
                '-acodec', 'libmp3lame',
                '-ar', '44100',
                '-ac', '2',
                '-b:a', '192k',
                chunk_path
            ], check=True, capture_output=True)
            chunk_paths.append(chunk_path)
        
        return chunk_paths
        
    except Exception as e:
        print_error(f"Error splitting audio: {e}")
        return None

def transcribe_with_openai_whisper(video_path):
    """Transcribe video using OpenAI's Whisper API"""
    try:
        from openai import OpenAI
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print_error("OPENAI_API_KEY environment variable not set")
            return False
        
        print_step(EMOJI_TRANSCRIBE, "Transcribing with OpenAI Whisper...")
        client = OpenAI()
        
        # Check if input format is supported
        supported_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
        input_ext = Path(video_path).suffix.lower()
        
        # Convert only if needed
        audio_path = video_path
        if input_ext not in supported_formats:
            print_step(EMOJI_TRANSCRIBE, "Converting to supported format...")
            audio_path = convert_audio_format(video_path, 'mp3', bitrate='32k', mono=True)
            if not audio_path:
                return False
        
        # Check file size (25MB limit)
        MAX_SIZE_MB = 25
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        
        if file_size_mb > MAX_SIZE_MB:
            print_step(EMOJI_TRANSCRIBE, f"File too large ({file_size_mb:.1f}MB), optimizing...")
            
            # Try aggressive compression first
            compressed_path = convert_audio_format(audio_path, 'mp3', bitrate='32k', mono=True)
            if not compressed_path:
                return False
            
            # Check if compression was enough
            compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
            if compressed_size_mb > MAX_SIZE_MB:
                print_step(EMOJI_TRANSCRIBE, "Still too large, splitting into chunks...")
                chunk_paths = split_audio_into_chunks(compressed_path, chunk_size_mb=20)
            else:
                chunk_paths = [compressed_path]
        else:
            chunk_paths = [audio_path]
        
        if not chunk_paths:
            return False
        
        # Transcribe each chunk
        transcripts = []
        for chunk_path in chunk_paths:
            # Verify chunk size
            chunk_size_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            if chunk_size_mb > MAX_SIZE_MB:
                print_error(f"Chunk too large: {chunk_size_mb:.1f}MB")
                continue
                
            with open(chunk_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                transcripts.append(transcription.text)
        
        if not transcripts:
            print_error("No successful transcriptions")
            return False
        
        # Combine transcripts
        full_transcript = " ".join(transcripts)
        
        # Write transcript to file
        transcript_path = os.path.splitext(video_path)[0] + '.txt'
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(full_transcript)
        
        # Clean up chunks if we created them
        if len(chunk_paths) > 1:
            chunks_dir = os.path.dirname(chunk_paths[0])
            for chunk in chunk_paths:
                os.remove(chunk)
            os.rmdir(chunks_dir)
        
        return True
        
    except ImportError:
        print_error("OpenAI package not found. Please install it with:")
        print(f"{Fore.YELLOW}pip install openai{Style.RESET_ALL}")
        return False
    except Exception as e:
        print_error(f"OpenAI Whisper error: {e}")
        return False

def transcribe_video(video_path, use_fast_whisper=False, use_replicate=False, language=None):
    """Transcribe video using chosen transcription method"""
    if use_replicate:
        return transcribe_with_replicate(video_path, language)
    elif use_fast_whisper:
        return transcribe_with_fast_whisper(video_path)
    else:
        return transcribe_with_openai_whisper(video_path)  # Default to OpenAI API

def summarize_with_claude(text, language="english"):
    """Get summary using Claude 3.5 Sonnet"""
    print_step(EMOJI_SUMMARY, f"Generating summary in {language}...")
    
    # Read the prompt from file
    prompt_file = Path(__file__).parent / "prompt.txt"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read()
    
    # Format prompt with specified language
    prompt = prompt_template.format(language=language)
    
    @ell.simple(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=8192)
    def get_summary(content: str) -> str:
        return (
            f"Here is the transcript you need to analyze:\n"
            f"<transcript>\n"
            f"{content}\n"
            f"</transcript>\n\n"
            f"{prompt}"
        )

    try:
        summary = get_summary(text)
        return summary
    except Exception as e:
        print_error(f"Error generating summary with Claude: {e}")
        return None

def get_video_metadata(url):
    """Get video metadata using yt-dlp"""
    try:
        clean_url = clean_youtube_url(url)
        print_step(EMOJI_SEARCH, "Fetching video metadata...")
        
        result = subprocess.run([
            'yt-dlp',
            '--dump-json',
            '--no-playlist',
            clean_url
        ], check=True, capture_output=True, text=True)
        
        metadata = json.loads(result.stdout)
        header_parts = ["---"]
        
        # Add metadata fields only if they exist
        if title := metadata.get('title'):
            header_parts.append(f"Title: {title}")
        
        if channel := metadata.get('channel'):
            header_parts.append(f"Channel: {channel}")
        
        if upload_date := metadata.get('upload_date'):
            header_parts.append(f"Upload Date: {upload_date}")
        
        if duration := metadata.get('duration_string'):
            header_parts.append(f"Duration: {duration}")
        
        if views := metadata.get('view_count'):
            header_parts.append(f"Views: {views:,}")
        
        if description := metadata.get('description'):
            # Process description with Ell
            processed = process_metadata_description(description)
            header_parts.append(f"Description: {processed}")
        
        if tags := metadata.get('tags'):
            # Process tags with Ell
            processed_tags = process_metadata_description(' '.join(tags))
            header_parts.append(f"Tags: {processed_tags}")
        
        header_parts.extend(["---", ""])
        
        return '\n'.join(header_parts)
    except Exception as e:
        print_error(f"Failed to fetch metadata: {e}")
        return ""

def convert_audio_format(input_path, output_format='mp3', bitrate='192k', mono=False):
    """Convert audio to specified format using FFmpeg"""
    try:
        print_step(EMOJI_TRANSCRIBE, f"Converting audio to {output_format} ({bitrate}{'mono' if mono else ''})...")
        output_path = str(Path(input_path).with_suffix(f'.{output_format}'))
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-y',  # Overwrite output file if exists
            '-vn',  # No video
            '-acodec', 'libmp3lame' if output_format == 'mp3' else output_format,
            '-ar', '44100',  # Sample rate
            '-ac', '1' if mono else '2',  # Mono/Stereo
            '-b:a', bitrate,  # Bitrate
            output_path
        ]
        
        # Run FFmpeg with error output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Verify file exists and is not empty
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            print_error("FFmpeg output file is missing or empty")
            return None
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        print_error(f"FFmpeg conversion failed: {e.stderr}")
        return None
    except Exception as e:
        print_error(f"Audio conversion error: {e}")
        return None

def process_metadata_description(metadata):
    """Process metadata description using Ell"""
    
    @ell.simple(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=1000)
    def summarize_metadata(content: str) -> str:
        """You are a metadata processor that creates concise video descriptions.
        Rules:
        1. Description must be a single line, max 3 semicolon-separated points
        2. Tags must be grouped by theme with parentheses, max 5 groups
        3. Remove all URLs, social media links, and promotional text
        4. Focus only on plot/content-relevant information
        5. Use semicolons to separate multiple plot points
        6. Group related tags inside parentheses
        7. Exclude generic/redundant tags"""
        
        return f"""Process this video metadata into a concise format:
1. Extract main plot points (max 3, separated by semicolons)
2. Group related tags (max 5 groups, use parentheses)

Metadata:
{content}

Format output as:
Description: [plot point 1]; [plot point 2]; [plot point 3]
Tags: [group1], [group2 (item1, item2)], [group3], [group4 (items...)]"""

    try:
        result = summarize_metadata(metadata)
        return result
    except Exception as e:
        print_error(f"Error processing metadata: {e}")
        return metadata

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Summarize YouTube videos')
    parser.add_argument('url', help='YouTube video URL or video ID')
    parser.add_argument('--language', default='english',
                       help='Output language for the summary (default: english)')
    
    # Add transcription method group
    trans_group = parser.add_mutually_exclusive_group()
    trans_group.add_argument('--fast-whisper', action='store_true',
                           help='Use Fast Whisper for transcription (faster)')
    trans_group.add_argument('--whisper', action='store_true',
                           help='Use OpenAI Whisper for transcription (slower but may be more accurate)')
    trans_group.add_argument('--replicate', action='store_true',
                           help='Use Replicate Incredibly Fast Whisper (fastest, requires API key)')
    args = parser.parse_args()
    
    # Get video metadata first
    try:
        metadata = get_video_metadata(args.url)
    except Exception as e:
        print_error(f"Error processing metadata: {e}")
        metadata = ""  # Continue without metadata
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        audio_path = temp_dir / "audio.m4a"
        base_path = temp_dir / "audio"
        
        # Try YouTube subtitles first (in requested language)
        subtitle_language = args.language.lower() if args.language else "en"
        subtitle_file = get_youtube_subtitles(args.url, str(base_path), subtitle_language)
        
        if subtitle_file:
            # Read the subtitle file
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                transcript = f.read()
            # Clean up the downloaded subtitle file
            os.remove(subtitle_file)
        elif args.whisper or args.fast_whisper or args.replicate:
            # Only download and transcribe if no subtitles found
            method = ('Fast Whisper' if args.fast_whisper 
                     else 'OpenAI Whisper' if args.whisper 
                     else 'Incredibly Fast Whisper')
            print_step(EMOJI_TRANSCRIBE, f"Using {method} for transcription...")
            if not download_video(args.url, str(audio_path)):
                sys.exit(1)
            
            if not transcribe_video(str(audio_path), 
                                  use_fast_whisper=args.fast_whisper,
                                  use_replicate=args.replicate,
                                  language=args.language):
                sys.exit(1)
                
            transcript = (temp_dir / "audio.txt").read_text()
        else:
            # No subtitles and no specific transcriber, fall back to Fast Whisper
            print_step(EMOJI_TRANSCRIBE, "No subtitles found, falling back to Fast Whisper transcription...")
            if not download_video(args.url, str(audio_path)):
                sys.exit(1)
            
            if not transcribe_video(str(audio_path), True):  # Default to Fast Whisper
                sys.exit(1)
            
            transcript = (temp_dir / "audio.txt").read_text()
        
        # Convert to shorthand
        shorthand = to_shorthand(transcript)
        
        # Get summary
        summary = summarize_with_claude(shorthand, language=args.language)
        if not summary:
            sys.exit(1)
        
        # Combine metadata and summary
        output = metadata + summary
        
        # Save output
        output_file = f"summary-{args.url.split('/')[-1]}.txt"
        Path(output_file).write_text(output)
        print_success(f"Summary saved to {output_file}")

if __name__ == "__main__":
    main()
