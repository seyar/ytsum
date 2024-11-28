#!/bin/bash

# Colors and emojis
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

EMOJI_DOWNLOAD="⬇️ "
EMOJI_TRANSCRIBE="🎯 "
EMOJI_SUMMARY="📝 "
EMOJI_SUCCESS="✅ "
EMOJI_ERROR="❌ "
EMOJI_SEARCH="🔍 "
EMOJI_SAVE="💾 "

# Print functions
print_step() {
    echo -e "${BLUE}${1}${2}${NC}"
}

print_error() {
    echo -e "${RED}${EMOJI_ERROR} ${1}${NC}"
}

print_success() {
    echo -e "${GREEN}${EMOJI_SUCCESS} ${1}${NC}"
}

# Convert text to shorthand to save tokens
toShorthand() {
    local input=$1
    local shorthand="$input"

    # Common word replacements
    declare -A replacements=(
        ["you"]="u"
        ["are"]="r"
        ["see"]="c"
        ["for"]="4"
        ["to"]="2"
        ["too"]="2"
        ["two"]="2"
        ["four"]="4"
        ["be"]="b"
        ["before"]="b4"
        ["great"]="gr8"
        ["thanks"]="thx"
        ["thank you"]="ty"
        ["because"]="bc"
        ["people"]="ppl"
        ["want"]="wnt"
        ["love"]="luv"
        ["okay"]="k"
        ["yes"]="y"
        ["no"]="n"
        ["please"]="plz"
        ["sorry"]="sry"
        ["see you"]="cya"
        ["I am"]="Im"
        ["good"]="gd"
        ["right"]="rt"
        ["later"]="l8r"
        ["have"]="hv"
        ["see you later"]="cul8r"
        ["laughing"]="lol"
        ["message"]="msg"
        ["information"]="info"
        ["about"]="abt"
        ["awesome"]="awsm"
        ["quickly"]="quick"
        ["first"]="1st"
        ["second"]="2nd"
        ["third"]="3rd"
        [" the "]=" "
        [" a "]=" "
        ["would"]="wd"
        ["could"]="cd"
        ["should"]="shd"
        ["with"]="w/"
        ["without"]="w/o"
        ["through"]="thru"
        ["think"]="thk"
        ["something"]="smth"
        ["someone"]="sm1"
        ["everyone"]="evry1"
        ["anyone"]="any1"
        ["nobody"]="no1"
        ["tomorrow"]="tmrw"
        ["tonight"]="2nite"
        ["today"]="2day"
        ["yesterday"]="yday"
        ["please"]="pls"
        ["probably"]="prob"
        ["definitely"]="def"
        ["really"]="rly"
        ["whatever"]="wtv"
        ["what"]="wut"
        ["why"]="y"
        ["where"]="whr"
        ["when"]="whn"
        ["who"]="hu"
        ["how"]="hw"
    )

    # Apply replacements
    for word in "${!replacements[@]}"; do
        shorthand="${shorthand//$word/${replacements[$word]}}"
    done

    # Remove extra spaces
    shorthand=$(echo "$shorthand" | tr -s ' ' | sed 's/^ *//g' | sed 's/ *$//g')

    echo "$shorthand"
}

# Clean and normalize YouTube URL or video ID
clean_youtube_url() {
    local url_or_id="$1"
    
    # Check if input is just a video ID
    if [[ ! "$url_or_id" =~ "youtube.com" ]] && [[ ! "$url_or_id" =~ "youtu.be" ]]; then
        # Assume it's a video ID, construct full URL
        echo "https://www.youtube.com/watch?v=$url_or_id"
        return
    fi
    
    # Extract video ID from full URL
    local video_id
    if [[ "$url_or_id" =~ youtu\.be ]]; then
        video_id=$(echo "$url_or_id" | sed -E 's/.*youtu.be\/([^?]*).*/\1/')
    else
        video_id=$(echo "$url_or_id" | sed -E 's/.*[?&]v=([^&]*).*/\1/')
    fi
    
    echo "https://www.youtube.com/watch?v=$video_id"
}

# Add metadata function
get_video_metadata() {
    local url="$1"
    local clean_url=$(clean_youtube_url "$url")
    
    print_step "$EMOJI_SEARCH" "Fetching video metadata..."
    
    # Get metadata in JSON format
    local metadata
    metadata=$(yt-dlp --dump-json --no-playlist "$clean_url")
    
    # Extract and format metadata using jq
    local title=$(echo "$metadata" | jq -r '.title')
    local channel=$(echo "$metadata" | jq -r '.channel')
    local upload_date=$(echo "$metadata" | jq -r '.upload_date')
    local duration=$(echo "$metadata" | jq -r '.duration_string')
    local views=$(echo "$metadata" | jq -r '.view_count')
    local description=$(echo "$metadata" | jq -r '.description')
    local tags=$(echo "$metadata" | jq -r '.tags | join(", ")')
    
    # Create metadata header
    cat << EOF
---
Title: $title
Channel: $channel
Upload Date: $upload_date
Duration: $duration
Views: $views
Description: |
    ${description//$'\n'/$$'\n    '}
Tags: $tags
---

EOF
}

# Update argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --fast-whisper)
            USE_FAST_WHISPER=1
            shift
            ;;
        --whisper)
            USE_WHISPER=1
            shift
            ;;
        *)
            URL="$1"
            shift
            ;;
    esac
done

# Update transcription logic
if [ -n "$USE_FAST_WHISPER" ]; then
    # ... existing Fast Whisper code ...
elif [ -n "$USE_WHISPER" ]; then
    # ... existing Whisper code ...
fi

# Get metadata
METADATA=$(get_video_metadata "$URL")

# Save output with metadata
echo "$METADATA$SUMMARY" > "$OUTPUT_FILE"