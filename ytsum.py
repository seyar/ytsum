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
from openai import OpenAI
import shutil
import re
from lumaai import LumaAI
import requests
import ffmpeg
from runwayml import RunwayML
from youtube_url import clean_youtube_url, get_video_id
from PIL import Image, ImageDraw
import base64
import io
import math

# Initialize colorama
init()

# Initialize Anthropic client and register with Ell
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print_error("ANTHROPIC_API_KEY environment variable not set")
    sys.exit(1)

claude_client = Anthropic()
ell.config.register_model("claude-3-5-sonnet-20241022", claude_client)

# Initialize LumaAI client
luma_api_key = os.getenv("LUMAAI_API_KEY")
if luma_api_key:
    luma_client = LumaAI(auth_token=luma_api_key)
else:
    luma_client = None

# Initialize RunwayML client
runway_api_key = os.getenv("RUNWAYML_API_SECRET")
if runway_api_key:
    runway_client = RunwayML()
else:
    runway_client = None

# Emoji constants
EMOJI_DOWNLOAD = "⬇️ "
EMOJI_TRANSCRIBE = "🎯 "
EMOJI_SUMMARY = "📝 "
EMOJI_SUCCESS = "✅ "
EMOJI_ERROR = "❌ "
EMOJI_SEARCH = "🔍 "
EMOJI_SAVE = "💾 "
EMOJI_PODCAST = "🎙️ "
EMOJI_AUDIO = "🔊 "
EMOJI_VIDEO = "🎥 "

# Add after other constants
DEFAULT_HOST_VOICES = {
    "host1": {"voice": "alloy", "name": "Alex"},
    "host2": {"voice": "nova", "name": "Sarah"}
}

# Update constants
AVAILABLE_VOICES = {
    "alloy": "Neutral voice",
    "echo": "Male voice",
    "fable": "Male voice",
    "onyx": "Male voice",
    "nova": "Female voice",
    "shimmer": "Female voice"
}

# Add after OpenAI client initialization
# Create output directory if it doesn't exist
OUTPUT_DIR = Path("/app/out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create subdirectories
(OUTPUT_DIR / "temp_videos").mkdir(exist_ok=True)

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
            '--cookies', '/app/data/www.youtube.com_cookies.txt',
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

def summarize_with_claude(transcript, metadata="", language="english"):
    """Generate summary using Claude"""
    # Get video duration from metadata or use default
    try:
        duration = float(re.search(r'Duration: (\d+\.\d+)', metadata).group(1))
    except:
        duration = 600  # Default to 10 minutes

    targets = calculate_target_length(duration)

    # Read the prompt template
    try:
        with open('prompt.txt', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except Exception as e:
        print_error(f"Error reading prompt template: {e}")
        return None

    @ell.simple(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=8192)
    def get_summary(content: str, target_words: int) -> str:
        # Format the prompt template with the target language
        formatted_prompt = prompt_template.format(language=language)

        return f"""{formatted_prompt}

        Target length: {target_words} words.

        Transcript:
        {content}"""

    try:
        return get_summary(f"{transcript}\n\nMetadata:\n{metadata}", targets['summary'])
    except Exception as e:
        print_error(f"Error generating summary: {e}")
        return None

def get_video_metadata(url):
    """Get video metadata using yt-dlp"""
    try:
        clean_url = clean_youtube_url(url)
        print_step(EMOJI_SEARCH, "Fetching video metadata...")

        result = subprocess.run([
            'yt-dlp',
            '--dump-json',
            '--cookies', '/app/data/www.youtube.com_cookies.txt',
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

def convert_to_podcast_script(summary, language="english", duration=None):
    """Convert summary to podcast script using Claude"""
    if duration is None:
        # Estimate duration from summary length
        duration = len(summary.split()) * 0.5  # rough estimate: 0.5 seconds per word

    targets = calculate_target_length(duration)
    @ell.simple(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=4096)
    def get_podcast(content: str, voice1: str, voice2: str, target_lang: str) -> str:
        return f"""Convert this summary into an engaging podcast script with two hosts.
        Target length: {targets['podcast']} words total.
        Output language: {target_lang}
        Use these voice names for the hosts: {voice1.upper()} and {voice2.upper()}.

        Rules:
        1. Format each line as: "VOICE_NAME: <dialogue>"
           Example: "{voice1.upper()}: That's an interesting point!"
        2. Use only {voice1.upper()} and {voice2.upper()} consistently
        3. Make it conversational but informative
        4. Keep all dialogue in {target_lang} language
        5. Include brief reactions and interactions between hosts
        6. Start with one host introducing the topic
        7. End with the other host wrapping up
        8. Keep the original insights and information
        9. Avoid meta-commentary or introductions
        10. Do NOT use typical AI buzzwords: dive in, delve into, fascinating,etc.
        10. Come up with original beginning (use ending for that). Do NOT start with "Today we are..."

        Available voices:
        {json.dumps(AVAILABLE_VOICES, indent=2)}

        Summary to convert:
        {content}"""

    try:
        # Randomly select two different voices
        import random
        available_voices = list(AVAILABLE_VOICES.keys())
        host1_voice = random.choice(available_voices)
        available_voices.remove(host1_voice)
        host2_voice = random.choice(available_voices)

        return get_podcast(summary, host1_voice, host2_voice, language)
    except Exception as e:
        print_error(f"Error converting to podcast script: {e}")
        return None

def generate_host_audio(text, host_config, output_path):
    """Generate audio for a specific host"""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            print_error("OPENAI_API_KEY environment variable not set")
            return False

        client = OpenAI()
        print_step(EMOJI_AUDIO, f"Generating audio for {host_config['name']}...")

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=host_config['voice'],
            input=text
        ) as response:
            response.stream_to_file(output_path)
        return True
    except Exception as e:
        print_error(f"Error generating audio: {e}")
        return False

def combine_audio_files(audio_files, output_file):
    """Combine multiple audio files with crossfade"""
    try:
        print_step(EMOJI_AUDIO, "Combining audio files...")

        if len(audio_files) < 2:
            print_error("Need at least two audio files to combine.")
            return False

        # Ensure output directory exists
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Build filter complex for crossfade
        filter_parts = []
        n = len(audio_files)

        # Label all inputs
        labels = [f'[{i}:a]' for i in range(n)]

        # Build the filter chain
        current_label = 0
        next_tmp = n  # Start temporary labels after input labels

        for i in range(n-1):
            if i == 0:
                # First merge
                filter_parts.append(f'{labels[i]}{labels[i+1]}acrossfade=d=0.5:c1=tri:c2=tri[tmp{next_tmp}]')
                current_label = next_tmp
                next_tmp += 1
            else:
                # Merge result with next input
                filter_parts.append(f'[tmp{current_label}]{labels[i+1]}acrossfade=d=0.5:c1=tri:c2=tri[tmp{next_tmp}]')
                current_label = next_tmp
                next_tmp += 1

        # Create input arguments
        inputs = []
        for audio_file in audio_files:
            inputs.extend(['-i', str(audio_file)])

        # Build final command
        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex',
            ';'.join(filter_parts),
            '-map', f'[tmp{current_label}]',
            '-ac', '2',  # Convert to stereo
            '-ar', '44100',  # Standard sample rate
            str(output_file)
        ]

        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print_error(f"FFmpeg error: {result.stderr}")
            return False

        return True

    except Exception as e:
        print_error(f"Error combining audio files: {e}")
        return False

def generate_podcast_audio(script, output_file):
    """Generate podcast audio with detected voices"""
    temp_files = []
    voice_configs = {}  # Will store voice configs as we discover them

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Process each line of the script
            for i, line in enumerate(script.split('\n')):
                if not line.strip():
                    continue

                # Parse voice and text
                try:
                    voice_name, text = line.split(':', 1)
                    voice_name = voice_name.strip().lower()
                    text = text.strip()
                except ValueError:
                    continue

                # Skip if not a valid voice
                if voice_name not in AVAILABLE_VOICES:
                    continue

                # Create voice config if not seen before
                if voice_name not in voice_configs:
                    voice_configs[voice_name] = {
                        "voice": voice_name,
                        "name": voice_name.capitalize()
                    }

                # Generate audio for this line
                temp_file = os.path.join(temp_dir, f"part_{i:03d}.mp3")
                if generate_host_audio(text, voice_configs[voice_name], temp_file):
                    temp_files.append(temp_file)

            # Combine all audio files
            if temp_files:
                return combine_audio_files(temp_files, output_file)

        return False
    except Exception as e:
        print_error(f"Error generating podcast: {e}")
        return False

def sanitize_filename(filename):
    """Convert URL or video ID to safe filename"""
    # Extract video ID from URL if present
    if 'youtube.com' in filename or 'youtu.be' in filename:
        try:
            if 'youtu.be' in filename:
                video_id = filename.split('/')[-1].split('?')[0]
            else:
                query = urllib.parse.urlparse(filename).query
                params = urllib.parse.parse_qs(query)
                video_id = params['v'][0]
            return video_id
        except:
            pass

    # Handle query parameters
    if '?' in filename:
        parts = filename.split('?')
        filename = parts[0]
        params = parts[1].replace('=', '_').replace('&', '_')
        filename = f"{filename}_{params}"

    # Count trailing special characters
    trailing_specials = len(filename) - len(filename.rstrip(r'\\/:*"<>|!'))

    # First replace special characters with underscores
    clean = re.sub(r'[\\/:*"<>|]', '_', filename)  # Replace invalid chars with underscore

    # Replace spaces and other non-alphanumeric chars (except dashes) with underscore
    clean = re.sub(r'[^\w\-]', '_', clean)

    # Replace multiple consecutive underscores with a single one
    clean = re.sub(r'_+', '_', clean)

    # Remove leading underscores
    clean = clean.lstrip('_')

    # Add single trailing underscore if original had special chars at the end
    if trailing_specials > 0:
        clean = clean.rstrip('_') + '_'

    # Preserve casing from original filename
    if filename.isupper():
        clean = clean.upper()
    elif not filename.islower():  # If mixed case or title case
        parts = clean.split('_')
        clean = '_'.join(p.capitalize() for p in parts)

    return clean

def generate_video_segments(podcast_script, num_segments=5, seed=42):
    """Generate video prompts that match podcast content and flow"""

    @ell.simple(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=2048)
    def get_video_prompts(script: str, num: int) -> str:
        return f"""Create {num} detailed video prompts that directly visualize the key moments from this podcast conversation.
        Each prompt must be under 500 characters long and create a clear, engaging scene.

        Podcast Script:
        {script}

        Guidelines for Each Prompt:
        1. Scene Content:
           - Focus on the specific topic being discussed
           - Show real environments and objects
           - Include relevant details mentioned by hosts
           - Keep descriptions concise but clear

        2. Visual Style:
           - Professional documentary style
           - Clean, high-quality visuals
           - Natural lighting
           - Clear focal points

        3. Required Structure (keep under 500 chars):
           "A [brief location] shows [main subject/action]. [Supporting details]. [Human elements] [interact with] [key concept]. [Lighting] highlights [focus]. [Camera angle]."

        4. Key Points:
           - Be specific but concise
           - Use concrete imagery
           - Match the conversation
           - Stay under length limit

        Instructions:
        1. Read the section
        2. Identify key concept
        3. Create concise scene
        4. Check character count
        5. Trim if needed

        Return a properly formatted JSON array of strings like this:
        [
            "First scene (under 500 chars)...",
            "Second scene (under 500 chars)..."
        ]

        Important: Use double quotes and ensure valid JSON format."""

    try:
        # Generate prompts and ensure valid JSON
        response = get_video_prompts(podcast_script, num_segments)

        # Parse JSON
        prompts = json.loads(response)

        if not isinstance(prompts, list) or len(prompts) != num_segments:
            raise ValueError(f"Invalid prompt format - must be array of exactly {num_segments} strings")

        # Validate and truncate prompts
        MAX_LENGTH = 500  # Keep some buffer below 512
        processed_prompts = []

        for i, prompt in enumerate(prompts, 1):
            if not isinstance(prompt, str):
                raise ValueError(f"Prompt {i} must be a string")

            # Ensure minimum detail
            if len(prompt.split()) < 20:
                raise ValueError(f"Prompt {i} is too short - needs more detail")

            # Truncate if too long
            if len(prompt) > MAX_LENGTH:
                # Find last complete sentence that fits
                sentences = prompt.split('.')
                truncated = ''
                for sentence in sentences:
                    if len(truncated + sentence + '.') <= MAX_LENGTH:
                        truncated += sentence + '.'
                    else:
                        break
                prompt = truncated.strip()

            processed_prompts.append(prompt)

        return processed_prompts

    except json.JSONDecodeError as e:
        print_error(f"Error parsing JSON response: {e}")
        print_error(f"Raw response: {response[:200]}...")
        return None
    except Exception as e:
        print_error(f"Error generating video prompts: {e}")
        return None

def upload_image_to_uguu(image_path, max_retries=3):
    """Upload image to uguu.se and get URL"""
    try:
        url = "https://uguu.se/upload"

        # Prepare the file with proper format
        with open(image_path, 'rb') as f:
            files = {
                'files[]': (
                    Path(image_path).name,
                    f,
                    'image/jpeg'
                )
            }

            # Try upload with retries
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        url,
                        files=files,
                        timeout=30
                    )

                    if response.status_code != 200:
                        print_error(f"Upload failed with status {response.status_code}")
                        if attempt < max_retries - 1:
                            time.sleep(1)
                            continue
                        return None

                    # Parse JSON response
                    try:
                        result = response.json()
                        if (result.get('success') and
                            isinstance(result.get('files'), list) and
                            result['files'] and
                            'url' in result['files'][0]):
                            return result['files'][0]['url']
                    except (ValueError, KeyError, AttributeError):
                        # If JSON parsing fails or format is unexpected, try text response
                        text = response.text.strip()
                        if text.startswith('http'):
                            return text

                    print_error(f"Invalid response format: {response.text[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None

                except requests.exceptions.RequestException as e:
                    print_error(f"Upload attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(1)
                        continue
                    return None

            return None

    except Exception as e:
        print_error(f"Error uploading image: {e}")
        return None

def generate_video_segments_with_luma(prompts, output_dir, base_images=None, podcast_script=None):
    """Generate video segments using LumaAI with optional base images"""
    if not luma_client:
        print_error("LUMA_API_KEY environment variable not set")
        return None

    video_paths = []
    for i, prompt in enumerate(prompts):
        try:
            print_step(EMOJI_VIDEO, f"Generating video segment {i+1}/{len(prompts)}...")

            # Set up generation parameters
            generation_params = {
                "prompt": prompt,
                "aspect_ratio": "16:9",
                "loop": False
            }

            # Add base image if available
            if base_images and i < len(base_images):
                image_url = upload_image_to_uguu(base_images[i])
                if not image_url:
                    print_error(f"Failed to upload image {i+1}, continuing without image")
                else:
                    generation_params["keyframes"] = {
                        "frame0": {
                            "type": "image",
                            "url": image_url
                        }
                    }

            # Try generation with retries and prompt regeneration
            max_retries = 3
            max_prompt_retries = 3
            generation = None

            for prompt_attempt in range(max_prompt_retries):
                try:
                    # Create generation with retries
                    for attempt in range(max_retries):
                        try:
                            generation = luma_client.generations.create(**generation_params)
                            break
                        except Exception as e:
                            if attempt < max_retries - 1:
                                print_error(f"Generation attempt {attempt + 1} failed: {e}, retrying...")
                                time.sleep(2)
                            else:
                                raise

                    if not generation:
                        raise Exception("Failed to create generation after retries")

                    # Poll for completion with timeout
                    start_time = time.time()
                    timeout = 300
                    completed = False
                    moderation_failed = False

                    while not completed and time.time() - start_time < timeout:
                        try:
                            generation = luma_client.generations.get(id=generation.id)

                            if generation.state == "completed":
                                completed = True
                                break
                            elif generation.state == "failed":
                                error_msg = getattr(generation, 'failure_reason', 'Unknown error')
                                if "moderation failed" in error_msg.lower():
                                    moderation_failed = True
                                    break
                                # Add regeneration for any failure
                                if prompt_attempt < max_prompt_retries - 1:
                                    print_error(f"Generation failed: {error_msg}, regenerating prompt...")
                                    new_prompts = generate_video_segments(podcast_script, num_segments=1)
                                    if new_prompts and len(new_prompts) > 0:
                                        generation_params["prompt"] = new_prompts[0]
                                        break
                                raise Exception(f"Video generation failed: {error_msg}")
                            elif generation.state == "canceled":
                                raise Exception("Video generation was cancelled")
                            else:
                                print_step(EMOJI_VIDEO, f"Generating segment {i+1}...", color=Fore.YELLOW)
                                time.sleep(3)

                        except Exception as e:
                            print_error(f"Error checking generation status: {e}")
                            time.sleep(3)

                    if moderation_failed:
                        if prompt_attempt < max_prompt_retries - 1:
                            print_error("Moderation failed, regenerating prompt...")
                            # Regenerate prompt for this segment
                            new_prompts = generate_video_segments(podcast_script, num_segments=1)
                            if new_prompts and len(new_prompts) > 0:
                                generation_params["prompt"] = new_prompts[0]
                                continue
                        raise Exception("Failed to generate acceptable prompt after retries")

                    if not completed:
                        raise Exception(f"Generation timed out after {timeout} seconds")

                    # If we get here, generation was successful
                    break

                except Exception as e:
                    if prompt_attempt < max_prompt_retries - 1:
                        print_error(f"Prompt attempt {prompt_attempt + 1} failed: {e}, trying new prompt...")
                        continue
                    raise

            # Download video with retries
            max_download_retries = 3
            for attempt in range(max_download_retries):
                try:
                    output_path = output_dir / f"segment_{i:02d}.mp4"
                    response = requests.get(generation.assets.video, stream=True, timeout=30)
                    response.raise_for_status()

                    with open(output_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)

                    if output_path.stat().st_size == 0:
                        raise Exception("Downloaded file is empty")

                    video_paths.append(output_path)
                    break

                except Exception as e:
                    if attempt < max_download_retries - 1:
                        print_error(f"Download attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(2)
                    else:
                        raise

        except Exception as e:
            print_error(f"Error generating video segment {i+1}: {e}")
            return None

    return video_paths

def combine_video_segments(video_paths, target_duration, output_path):
    """Combine video segments and adjust to match target duration"""
    try:
        print_step(EMOJI_VIDEO, "Combining video segments...")

        # Create temporary file for concatenation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write input files list with absolute paths
            for video_path in video_paths:
                # Convert to absolute path
                abs_path = Path(video_path).resolve()
                if not abs_path.exists():
                    raise FileNotFoundError(f"Video file not found: {abs_path}")
                f.write(f"file '{abs_path}'\n")
            temp_list = f.name

        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Concatenate videos
            temp_concat = output_path.parent / 'temp_concat.mp4'
            subprocess.run([
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', temp_list, '-c', 'copy', str(temp_concat)
            ], check=True, capture_output=True)

            # Get concatenated video duration
            probe = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(temp_concat)
            ], capture_output=True, text=True)
            current_duration = float(probe.stdout.strip())

            # Calculate speed factor to stretch video to match target duration
            # If current_duration is 30s and target is 60s, we want speed_factor = 2
            # to make the video twice as slow
            speed_factor = target_duration / current_duration

            subprocess.run([
                'ffmpeg', '-y', '-i', str(temp_concat),
                '-filter:v', f'setpts={speed_factor}*PTS',
                '-an', str(output_path)
            ], check=True, capture_output=True)

            return True

        finally:
            # Clean up temporary files
            os.unlink(temp_list)
            if temp_concat.exists():
                os.unlink(temp_concat)

    except subprocess.CalledProcessError as e:
        print_error(f"FFmpeg error: {e.stderr.decode()}")
        return False
    except Exception as e:
        print_error(f"Error combining videos: {e}")
        return False

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        probe = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', audio_path
        ], capture_output=True, text=True)
        return float(probe.stdout.strip())
    except:
        return None

def combine_audio_video(video_path, audio_path, output_path):
    """Combine video with audio track using ffmpeg and add fade out"""
    try:
        print_step(EMOJI_VIDEO, "Combining video and audio...")

        # Get video duration
        probe = ffmpeg.probe(video_path)
        duration = float(probe['streams'][0]['duration'])
        fade_start = duration - 1  # Start fade 1 second before end

        # Create filter complex for fade out
        # Apply fade filter directly to video stream
        stream = (
            ffmpeg
            .input(video_path)
            .filter('fade', type='out', start_time=fade_start, duration=1)
            .output(
                ffmpeg.input(audio_path),
                str(output_path),
                acodec='aac',
                strict='experimental',
                **{
                    'filter_complex_threads': 1,
                    'max_muxing_queue_size': 1024
                }
            )
        )

        # Run ffmpeg with overwrite and error handling
        try:
            ffmpeg.run(
                stream,
                overwrite_output=True,
                capture_stdout=True,
                capture_stderr=True
            )
            return True

        except ffmpeg.Error as e:
            if e.stderr:
                print_error(f"FFmpeg error: {e.stderr.decode()}")
            if e.stdout:
                print_error(f"FFmpeg output: {e.stdout.decode()}")
            return False

    except Exception as e:
        print_error(f"Error combining audio and video: {e}")
        return False

def generate_video_segments_with_runway(prompts, output_dir, base_images=None, timeout=900, podcast_script=None):
    """Generate video segments using RunwayML with optional base images"""
    if not runway_client:
        print_error("RUNWAYML_API_SECRET environment variable not set")
        return None

    video_paths = []
    for i, prompt in enumerate(prompts):
        try:
            print_step(EMOJI_VIDEO, f"Generating video segment {i+1}/{len(prompts)}...")

            # Try generation with retries and prompt regeneration
            max_retries = 3
            max_prompt_retries = 3

            for prompt_attempt in range(max_prompt_retries):
                try:
                    # Use base image if available, otherwise create gradient
                    if base_images and i < len(base_images):
                        with open(base_images[i], 'rb') as f:
                            image_bytes = f.read()
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        image_uri = f"data:image/jpeg;base64,{image_b64}"
                    else:
                        temp_image = output_dir / f"input_{i:02d}.png"
                        gradient = create_gradient_image()
                        gradient.save(temp_image)
                        with open(temp_image, 'rb') as f:
                            image_bytes = f.read()
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
                        image_uri = f"data:image/png;base64,{image_b64}"
                        temp_image.unlink()

                    # Create task with current prompt
                    task = runway_client.image_to_video.create(
                        model='gen3a_turbo',
                        prompt_text=prompt,
                        prompt_image=image_uri,
                        duration=10,
                        ratio="1280:768"
                    )

                    # Poll for completion with timeout
                    start_time = time.time()
                    max_retries = 180
                    retries = 0
                    moderation_failed = False

                    while retries < max_retries:
                        if time.time() - start_time > timeout:
                            print_error(f"Timeout after {timeout} seconds")
                            try:
                                runway_client.tasks.cancel(id=task.id)
                            except:
                                pass
                            return None

                        try:
                            task_status = runway_client.tasks.retrieve(id=task.id)
                        except Exception as e:
                            print_error(f"Error retrieving task status: {e}")
                            time.sleep(5)
                            retries += 1
                            continue

                        if task_status.status == "SUCCEEDED":
                            if not hasattr(task_status, 'output') or not task_status.output:
                                print_error("No output in completed task")
                                return None

                            video_urls = task_status.output
                            if not video_urls or not isinstance(video_urls, list):
                                print_error("Invalid output format in task")
                                return None

                            video_url = video_urls[0]
                            break

                        elif task_status.status == "FAILED":
                            error_msg = getattr(task_status, 'failure', '') or getattr(task_status, 'failureCode', 'Unknown error')
                            if "moderation" in error_msg.lower():
                                moderation_failed = True
                                break
                            # Add regeneration for any failure
                            if prompt_attempt < max_prompt_retries - 1:
                                print_error(f"Generation failed: {error_msg}, regenerating prompt...")
                                new_prompts = generate_video_segments(podcast_script, num_segments=1)
                                if new_prompts and len(new_prompts) > 0:
                                    prompt = new_prompts[0]  # Update prompt for next attempt
                                    break
                            print_error(f"Video generation failed: {error_msg}")
                            return None

                        elif task_status.status == "CANCELLED":
                            print_error("Video generation was cancelled")
                            return None

                        elif task_status.status == "THROTTLED":
                            print_step(EMOJI_VIDEO, f"Generation queued (throttled)... Attempt {retries+1}/{max_retries}", color=Fore.YELLOW)

                        elif task_status.status == "PENDING":
                            print_step(EMOJI_VIDEO, f"Generation pending... Attempt {retries+1}/{max_retries}", color=Fore.YELLOW)

                        elif task_status.status == "RUNNING":
                            progress = float(getattr(task_status, 'progress', 0) or 0) * 100
                            elapsed = int(time.time() - start_time)
                            print_step(EMOJI_VIDEO,
                                     f"Generating segment {i+1}... ({progress:.0f}%) - {elapsed}s elapsed",
                                     color=Fore.YELLOW)

                        time.sleep(5)
                        retries += 1
                        continue

                    if moderation_failed:
                        if prompt_attempt < max_prompt_retries - 1:
                            print_error("Moderation failed, regenerating prompt...")
                            # Regenerate prompt for this segment
                            new_prompts = generate_video_segments(podcast_script, num_segments=1)
                            if new_prompts and len(new_prompts) > 0:
                                prompt = new_prompts[0]  # Update prompt for next attempt
                                break
                            raise Exception("Failed to generate acceptable prompt after retries")

                    # Download video with retries
                    max_download_retries = 3
                    for download_attempt in range(max_download_retries):
                        try:
                            output_path = output_dir / f"segment_{i:02d}.mp4"
                            response = requests.get(video_url, stream=True, timeout=30)
                            response.raise_for_status()

                            with open(output_path, 'wb') as file:
                                for chunk in response.iter_content(chunk_size=8192):
                                    file.write(chunk)

                            if output_path.stat().st_size == 0:
                                raise Exception("Downloaded file is empty")

                            video_paths.append(output_path)
                            break

                        except Exception as e:
                            if download_attempt < max_download_retries - 1:
                                print_error(f"Download attempt {download_attempt + 1} failed: {e}, retrying...")
                                time.sleep(2)
                            else:
                                raise

                    # If we get here, generation and download were successful
                    break

                except Exception as e:
                    if prompt_attempt < max_prompt_retries - 1:
                        print_error(f"Prompt attempt {prompt_attempt + 1} failed: {e}, trying new prompt...")
                        continue
                    raise

        except Exception as e:
            print_error(f"Error generating video segment {i+1}: {e}")
            return None

    return video_paths

def calculate_num_segments(audio_duration, provider="luma"):
    """Calculate optimal number of video segments based on audio duration and provider"""
    # Provider-specific segment durations
    SEGMENT_DURATIONS = {
        "luma": 5,    # LumaAI generates 5s videos
        "runway": 10  # RunwayML generates 10s videos
    }

    # Provider-specific maximum segments
    MAX_SEGMENTS = {
        "luma": 10,   # Allow more segments for LumaAI due to shorter duration
        "runway": 5   # Keep RunwayML at 5 segments max
    }

    segment_duration = SEGMENT_DURATIONS.get(provider, 5)  # Default to 5s if provider unknown
    max_segments = MAX_SEGMENTS.get(provider, 5)  # Default to 5 if provider unknown

    # Calculate ideal number of segments to cover the audio
    ideal_segments = math.ceil(audio_duration / segment_duration)

    # Keep segments between 2 and max_segments
    if audio_duration <= segment_duration:
        # Very short audio - single segment
        return 1
    elif audio_duration <= 2 * segment_duration:
        # Short audio - two segments
        return 2
    elif audio_duration <= max_segments * segment_duration:
        # Medium audio - scale segments based on duration
        return min(max_segments, max(2, ideal_segments))
    else:
        # Long audio - cap at max_segments
        return max_segments

def calculate_target_length(duration_seconds):
    """Calculate target word counts based on content duration"""
    # Base lengths for a 10-minute video
    BASE_DURATION = 600  # 10 minutes in seconds
    BASE_SUMMARY_WORDS = 300
    BASE_PODCAST_WORDS = 600

    # Calculate scaling factor (with min/max limits)
    scale = min(max(duration_seconds / BASE_DURATION, 0.3), 2.0)

    return {
        'summary': int(BASE_SUMMARY_WORDS * scale),
        'podcast': int(BASE_PODCAST_WORDS * scale)
    }

def generate_image_prompts(video_prompts):
    """Generate relevant, concrete image prompts that match podcast content"""

    @ell.simple(model="claude-3-5-sonnet-20241022", temperature=0.3, max_tokens=2048)
    def get_image_prompts(prompts: list, summary: str, podcast: str) -> str:
        return f"""Create {len(prompts)} detailed image prompts for Stable Diffusion that illustrate the key topics being discussed.
        Each prompt should create a clear, realistic visualization of the concepts, using concrete imagery.

        Content Summary:
        {summary}

        Podcast Script:
        {podcast}

        Required Elements for Each Prompt:
        1. Base Quality:
           - Start with: "masterpiece, highly detailed, 8k uhd, photorealistic"
           - End with: "professional lighting, cinematic composition"

        2. Scene Components:
           - Main Subject: Primary topic or concept being discussed
           - Environment: Relevant setting or location
           - Supporting Elements: Objects, tools, or items that relate to the topic
           - Human Element: People, hands, or human presence when relevant
           - Scale: Show size and scope of the subject matter

        3. Visual Guidelines:
           - Create documentary-style scenes
           - Show real objects and environments
           - Include relevant details from the discussion
           - Use appropriate lighting for the setting
           - Choose engaging camera angles
           - Keep scenes grounded and realistic

        4. Scene Types:
           - Process/Action: Show something being done or created
           - Location/Setting: Establish where something happens
           - Object/Detail: Focus on specific items being discussed
           - Interaction: Show how things or people work together
           - Result/Impact: Visualize outcomes or effects

        Instructions:
        1. Read the current section of discussion
        2. Identify the main concept or point
        3. Choose the most appropriate scene type
        4. Include specific details mentioned in the content
        5. Make it concrete and photorealistic
        6. Ensure it matches the topic being discussed

        Example Structure:
        "masterpiece, highly detailed, 8k uhd, photorealistic, [main subject in action/setting], [environment details], [supporting elements], [human presence if relevant], [lighting and atmosphere], professional lighting, cinematic composition"

        Return a JSON array of {len(prompts)} strings.
        No code blocks, only the JSON array."""

    try:
        # Read the summary and podcast files for context
        summary_file = next(Path("out").glob("summary-*.txt"))
        podcast_file = next(Path("out").glob("podcast-*.txt"))
        summary = summary_file.read_text()
        podcast = podcast_file.read_text()

        # Generate prompts with content context
        prompts = json.loads(get_image_prompts(video_prompts, summary, podcast))

        # Validate prompts
        if not isinstance(prompts, list) or len(prompts) != len(video_prompts):
            raise ValueError(f"Invalid prompt format - must be array of exactly {len(video_prompts)} strings")

        # Ensure all prompts are strings and have required elements
        prompts = [str(p) for p in prompts]

        # Validate prompt structure
        for i, prompt in enumerate(prompts):
            if not isinstance(prompt, str):
                raise ValueError(f"Prompt {i} must be a string")
            if not prompt.startswith("masterpiece, highly detailed, 8k uhd, photorealistic"):
                raise ValueError(f"Prompt {i} must start with the required quality elements")
            if not prompt.endswith("professional lighting, cinematic composition"):
                raise ValueError(f"Prompt {i} must end with the required composition elements")

        return prompts

    except Exception as e:
        print_error(f"Error generating image prompts: {e}")
        return None

def generate_flux_images(prompts, output_dir):
    """Generate images using Flux Pro Ultra for each prompt"""
    if not os.getenv("REPLICATE_API_TOKEN"):
        print_error("REPLICATE_API_TOKEN environment variable not set")
        return None

    image_paths = []  # Store local paths
    for i, prompt in enumerate(prompts):
        try:
            print_step(EMOJI_VIDEO, f"Generating base image {i+1}/{len(prompts)}...")

            # Get image URL from Replicate
            output_url = replicate.run(
                "black-forest-labs/flux-1.1-pro-ultra",
                input={
                    "raw": False,
                    "prompt": prompt,
                    "aspect_ratio": "16:9",
                    "output_format": "jpg",
                    "safety_tolerance": 2,
                    "image_prompt_strength": 0.1
                }
            )

            # Download and save image
            output_path = output_dir / f"base_{i:02d}.jpg"
            response = requests.get(output_url, stream=True)
            with open(output_path, 'wb') as file:
                file.write(response.content)

            image_paths.append(output_path)

        except Exception as e:
            print_error(f"Error generating base image {i+1}: {e}")
            return None

    return image_paths

def create_gradient_image(width=1280, height=768):
    """Create a simple gradient image for video generation"""
    image = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(image)

    # Create a vertical gradient from dark to light blue
    for y in range(height):
        # Calculate color components
        r = int(20 * y / height)  # Dark to slightly red
        g = int(50 * y / height)  # Dark to medium green
        b = int(255 * y / height)  # Dark to bright blue

        # Draw horizontal line with current color
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return image

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Summarize YouTube videos')
    parser.add_argument('url', help='YouTube video URL or video ID')
    parser.add_argument('--language', default='english',
                       help='Output language for the summary (default: english)')
    parser.add_argument('--podcast', action='store_true',
                       help='Generate podcast version with audio')
    parser.add_argument('--ignore-subs', action='store_true',
                       help='Ignore YouTube subtitles and force transcription')

    # Add transcription method group
    trans_group = parser.add_mutually_exclusive_group()
    trans_group.add_argument('--fast-whisper', action='store_true',
                           help='Use Fast Whisper for transcription (faster)')
    trans_group.add_argument('--whisper', action='store_true',
                           help='Use OpenAI Whisper for transcription (slower but may be more accurate)')
    trans_group.add_argument('--replicate', action='store_true',
                           help='Use Replicate Incredibly Fast Whisper (fastest, requires API key)')

    # Video generation group
    video_group = parser.add_mutually_exclusive_group()
    video_group.add_argument('--lumaai', action='store_true',
                          help='Generate video using Luma AI (requires --podcast)')
    video_group.add_argument('--runwayml', action='store_true',
                          help='Generate video using RunwayML (requires --podcast)')

    args = parser.parse_args()

    try:
        # Clean and validate URL
        clean_url = clean_youtube_url(args.url)

        # Get video ID for filenames
        try:
            video_id = get_video_id(clean_url)
        except:
            print_error("Could not extract video ID from URL")
            sys.exit(1)

        # Check for existing files
        summary_file = OUTPUT_DIR / f"summary-{video_id}.txt"
        podcast_script_file = OUTPUT_DIR / f"podcast-{video_id}.txt"
        podcast_audio_file = OUTPUT_DIR / f"podcast-{video_id}.mp3"
        final_video_file = OUTPUT_DIR / f"video-{video_id}.mp4"

        # Get video metadata first (always do this to verify video exists)
        try:
            metadata = get_video_metadata(clean_url)
            # Get video duration from metadata
            duration = None
            if metadata:
                try:
                    duration = float(re.search(r'Duration: (\d+\.\d+)', metadata).group(1))
                except:
                    pass
        except Exception as e:
            print_error(f"Error processing metadata: {e}")
            metadata = ""  # Continue without metadata
            duration = None

        # Check if we need to generate summary
        if summary_file.exists():
            print_step(EMOJI_SUCCESS, f"Summary already exists at {summary_file}")
            summary = summary_file.read_text()
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_dir = Path(temp_dir)
                audio_path = temp_dir / "audio.m4a"
                base_path = temp_dir / "audio"

                # Try YouTube subtitles first (unless --ignore-subs is used)
                transcript = None
                if not args.ignore_subs:
                    subtitle_language = args.language.lower() if args.language else "en"
                    subtitle_file = get_youtube_subtitles(clean_url, str(base_path), subtitle_language)

                    if subtitle_file:
                        # Read the subtitle file
                        with open(subtitle_file, 'r', encoding='utf-8') as f:
                            transcript = f.read()
                        # Clean up the downloaded subtitle file
                        os.remove(subtitle_file)

                # If no transcript yet (no subs or --ignore-subs), transcribe audio
                if not transcript:
                    method = ('Fast Whisper' if args.fast_whisper
                             else 'OpenAI Whisper' if args.whisper
                             else 'Incredibly Fast Whisper' if args.replicate
                             else 'Fast Whisper')  # Default
                    print_step(EMOJI_TRANSCRIBE, f"Using {method} for transcription...")

                    if not download_video(clean_url, str(audio_path)):
                        sys.exit(1)

                    if not transcribe_video(str(audio_path),
                                          use_fast_whisper=args.fast_whisper or (not args.whisper and not args.replicate),
                                          use_replicate=args.replicate,
                                          language=args.language):
                        sys.exit(1)

                    transcript = (temp_dir / "audio.txt").read_text()

                # Convert to shorthand
                shorthand = to_shorthand(transcript)

                # Generate summary with appropriate length
                summary = summarize_with_claude(shorthand, metadata, args.language)
                if not summary:
                    sys.exit(1)

                # Save summary
                Path(summary_file).write_text(metadata + summary)
                print_success(f"Summary saved to {summary_file}")

        # If podcast option is enabled
        if args.podcast:
            # Check if podcast files exist
            if podcast_script_file.exists() and podcast_audio_file.exists():
                print_step(EMOJI_SUCCESS, f"Podcast script already exists at {podcast_script_file}")
                print_step(EMOJI_SUCCESS, f"Podcast audio already exists at {podcast_audio_file}")
                podcast_script = podcast_script_file.read_text()
            else:
                # Convert to podcast script and generate audio
                podcast_script = convert_to_podcast_script(summary, args.language, duration)
                if not podcast_script:
                    sys.exit(1)

                # Save podcast script
                podcast_script_file.write_text(podcast_script)

                # Generate audio file
                if not generate_podcast_audio(podcast_script, podcast_audio_file):
                    sys.exit(1)

                print_success(f"Podcast script saved to {podcast_script_file}")
                print_success(f"Podcast audio saved to {podcast_audio_file}")

            # If video generation is enabled
            if args.lumaai or args.runwayml:
                # Check if final video exists
                if final_video_file.exists():
                    print_step(EMOJI_SUCCESS, f"Final video already exists at {final_video_file}")
                    return

                # Create temporary directory for video segments
                video_temp_dir = OUTPUT_DIR / "temp_videos"
                video_temp_dir.mkdir(exist_ok=True)

                temp_video = None
                try:
                    # Get podcast audio duration
                    audio_duration = get_audio_duration(podcast_audio_file)
                    if not audio_duration:
                        print_error("Could not determine podcast duration")
                        sys.exit(1)

                    # Calculate number of segments needed
                    num_segments = calculate_num_segments(
                        audio_duration,
                        provider="luma" if args.lumaai else "runway"
                    )

                    # Generate video prompts
                    prompts = generate_video_segments(podcast_script, num_segments=num_segments)
                    if not prompts:
                        sys.exit(1)

                    # Generate base images with Flux
                    image_prompts = generate_image_prompts(prompts)
                    if not image_prompts:
                        sys.exit(1)

                    base_images = generate_flux_images(image_prompts, video_temp_dir)
                    if not base_images:
                        sys.exit(1)

                    # Generate video segments with selected provider
                    if args.lumaai:
                        video_paths = generate_video_segments_with_luma(
                            prompts,
                            video_temp_dir,
                            base_images,
                            podcast_script=podcast_script
                        )
                    else:  # args.runwayml
                        video_paths = generate_video_segments_with_runway(
                            prompts,
                            video_temp_dir,
                            base_images,
                            podcast_script=podcast_script
                        )

                    if not video_paths:
                        sys.exit(1)

                    # Combine videos and match audio duration
                    temp_video = OUTPUT_DIR / f"temp-video-{video_id}.mp4"

                    if not combine_video_segments(video_paths, audio_duration, temp_video):
                        sys.exit(1)

                    # Combine with podcast audio
                    if not combine_audio_video(temp_video, podcast_audio_file, final_video_file):
                        sys.exit(1)

                    print_success(f"Final video saved to {final_video_file}")

                finally:
                    # Clean up temporary files
                    if video_temp_dir.exists():
                        shutil.rmtree(video_temp_dir)
                    if temp_video and temp_video.exists():
                        os.remove(temp_video)

    except KeyboardInterrupt:
        print_error("\nOperation cancelled by user")
        sys.exit(1)

if __name__ == "__main__":
    main()
