#!/bin/bash

# Colors and emojis
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'
EMOJI_DOWNLOAD="⬇️ "
EMOJI_TRANSCRIBE="🎯 "
EMOJI_SUMMARY="📝 "
EMOJI_SUCCESS="✅ "
EMOJI_ERROR="❌ "
EMOJI_SEARCH="🔍 "

# Print functions
print_step() { printf "${BLUE}${2} ${1}${NC}\n"; }
print_error() { printf "${RED}${EMOJI_ERROR} ${1}${NC}\n"; }
print_success() { printf "${GREEN}${EMOJI_SUCCESS} ${1}${NC}\n"; }

# Check dependencies
command -v yt-dlp >/dev/null 2>&1 || { print_error "yt-dlp is required"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { print_error "ffmpeg is required"; exit 1; }

# Check API keys
[ -z "$ANTHROPIC_API_KEY" ] && { print_error "ANTHROPIC_API_KEY not set"; exit 1; }

# Parse arguments
VIDEO_URL=""
LANGUAGE="english"
TRANSCRIBER="fast-whisper"

while [[ $# -gt 0 ]]; do
    case $1 in
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --whisper)
            TRANSCRIBER="whisper"
            shift
            ;;
        --replicate)
            TRANSCRIBER="replicate"
            shift
            ;;
        *)
            VIDEO_URL="$1"
            shift
            ;;
    esac
done

[ -z "$VIDEO_URL" ] && { print_error "Video URL required"; exit 1; }

# Clean YouTube URL
clean_url() {
    local url="$1"
    if [[ ! "$url" =~ (youtube\.com|youtu\.be) ]]; then
        url="https://www.youtube.com/watch?v=$url"
    fi
    echo "$url"
}

VIDEO_URL=$(clean_url "$VIDEO_URL")

# Create temp directory
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Try to get subtitles first
print_step "Searching for YouTube subtitles..." "$EMOJI_SEARCH"
LANG_CODE=$(python3 -c "from ytsum import get_language_code; print(get_language_code('$LANGUAGE'))")

yt-dlp \
    --write-subs \
    --sub-langs "$LANG_CODE" \
    --skip-download \
    --output "$TEMP_DIR/video" \
    "$VIDEO_URL"

# Check if subtitles were downloaded
if [ -f "$TEMP_DIR/video.$LANG_CODE.vtt" ]; then
    print_success "Found subtitles!"
    # Convert VTT to plain text
    sed '1,/^$/d' "$TEMP_DIR/video.$LANG_CODE.vtt" | \
    sed '/-->/d' | \
    sed '/^$/d' | \
    tr '\n' ' ' > "$TEMP_DIR/transcript.txt"
else
    print_step "No subtitles found, transcribing audio..." "$EMOJI_SEARCH"
    
    # Download audio
    print_step "Downloading audio..." "$EMOJI_DOWNLOAD"
    yt-dlp \
        --extract-audio \
        --audio-format m4a \
        --output "$TEMP_DIR/audio.%(ext)s" \
        "$VIDEO_URL"
    
    # Transcribe based on selected method
    case $TRANSCRIBER in
        "whisper")
            [ -z "$OPENAI_API_KEY" ] && { print_error "OPENAI_API_KEY not set"; exit 1; }
            print_step "Using OpenAI Whisper..." "$EMOJI_TRANSCRIBE"
            python3 -c "from ytsum import transcribe_with_openai_whisper; transcribe_with_openai_whisper('$TEMP_DIR/audio.m4a')"
            ;;
        "replicate")
            [ -z "$REPLICATE_API_TOKEN" ] && { print_error "REPLICATE_API_TOKEN not set"; exit 1; }
            print_step "Using Replicate..." "$EMOJI_TRANSCRIBE"
            python3 -c "from ytsum import transcribe_with_replicate; transcribe_with_replicate('$TEMP_DIR/audio.m4a', '$LANGUAGE')"
            ;;
        *)
            print_step "Using Fast Whisper..." "$EMOJI_TRANSCRIBE"
            python3 -c "from ytsum import transcribe_with_fast_whisper; transcribe_with_fast_whisper('$TEMP_DIR/audio.m4a')"
            ;;
    esac
    
    mv "$TEMP_DIR/audio.txt" "$TEMP_DIR/transcript.txt"
fi

# Get metadata
print_step "Fetching metadata..." "$EMOJI_SEARCH"
python3 -c "from ytsum import get_video_metadata; print(get_video_metadata('$VIDEO_URL'))" > "$TEMP_DIR/metadata.txt"

# Convert to shorthand
print_step "Converting to shorthand..." "$EMOJI_SUMMARY"
python3 -c "from ytsum import to_shorthand; print(to_shorthand(open('$TEMP_DIR/transcript.txt').read()))" > "$TEMP_DIR/shorthand.txt"

# Generate summary
print_step "Generating summary..." "$EMOJI_SUMMARY"
python3 -c "
from ytsum import summarize_with_claude
with open('$TEMP_DIR/shorthand.txt') as f:
    summary = summarize_with_claude(f.read(), '$LANGUAGE')
print(summary)
" > "$TEMP_DIR/summary.txt"

# Combine output
cat "$TEMP_DIR/metadata.txt" "$TEMP_DIR/summary.txt" > "summary-${VIDEO_URL##*=}.txt"
print_success "Summary saved to summary-${VIDEO_URL##*=}.txt"