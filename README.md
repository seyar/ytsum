# YouTube Summary Tool (ytsum)

A command-line tool that creates summaries and podcasts from YouTube videos using AI.

## Features

- Downloads and transcribes YouTube videos
- Generates concise summaries using Claude AI
- Creates podcast versions with OpenAI Text-to-Speech
- Supports multiple languages
- Multiple transcription options (Fast Whisper, OpenAI Whisper, Replicate)
- Automatic subtitle detection and download
- Combines podcast audio files for seamless playback

## Requirements

- Python 3.8+
- FFmpeg
- yt-dlp
- OpenAI API key (for podcast feature)
- Anthropic API key (for summaries)
- Replicate API key (optional, for faster transcription)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-summarizer.git
cd youtube-summarizer
pip install -r requirements.txt
```

2. Set up API keys:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   export OPENAI_API_KEY='your-api-key-here'  # Required for podcast feature
   export REPLICATE_API_TOKEN='your-api-key-here'  # Optional, for Replicate
   ```

3. Install system dependencies:
   ```bash
   # macOS
   brew install ffmpeg yt-dlp

   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   pip install yt-dlp
   ```

## Usage

### Basic Summary

```bash
python ytsum.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Generate Podcast

```bash
python ytsum.py "https://www.youtube.com/watch?v=VIDEO_ID" --podcast
```

The script will use two different voices from the following options:

- `alloy`: Neutral voice
- `echo`: Male voice
- `fable`: Male voice
- `onyx`: Male voice
- `nova`: Female voice
- `shimmer`: Female voice

Example podcast script:

```
NOVA: Welcome to our summary of this fascinating video!
ECHO: That's right, Nova. Let's break down the key points...
```

### Transcription Methods

```bash
# Use Fast Whisper (faster, local)
python ytsum.py "VIDEO_URL" --fast-whisper

# Use OpenAI Whisper (more accurate)
python ytsum.py "VIDEO_URL" --whisper

# Use Replicate (fastest, requires API key)
python ytsum.py "VIDEO_URL" --replicate
```

### Language Options

```bash
# Generate summary in Spanish
python ytsum.py "VIDEO_URL" --language spanish

# Generate podcast in Spanish
python ytsum.py "VIDEO_URL" --podcast --language spanish
```

## Output Files

All generated files are saved in the `out/` directory:

- `out/summary-{video_id}.txt`: Text summary of the video
- `out/podcast-{video_id}.txt`: Podcast script (when using `--podcast`)
- `out/podcast-{video_id}.mp3`: Audio podcast file (when using `--podcast`)

The output directory is automatically created if it doesn't exist.

## Dependencies

- `yt-dlp`: For downloading YouTube videos
- `ffmpeg`: For audio processing
- `openai`: For Whisper API and text-to-speech
- `anthropic`: For Claude AI summaries
- `replicate`: For cloud transcription
- `colorama`: For colored terminal output
- `ell`: For AI model management

## Notes on Audio Processing

- The script combines audio files without crossfade for better compatibility.
- All audio files are converted to a consistent format to ensure proper concatenation.
- FFmpeg's `concat` demuxer is used for combining audio files.

## Error Handling

The tool includes comprehensive error handling for:

- Failed video downloads
- Missing subtitles
- Transcription errors
- API issues
- Missing dependencies
- File system operations

## Development

When contributing, please:

1. Add tests for new functionality
2. Ensure all tests pass
3. Follow the existing code style
4. Update documentation as needed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
