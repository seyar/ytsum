# Awesome YouTube Video Summary/Podcast/Video

A Python script to generate summaries (Claude), podcasts (Whisper), and videos (RunwayML or Luma AI) from annoyingly long YouTube content.

![CleanShot 2024-12-02 at 16 25 26@2x](https://github.com/user-attachments/assets/f1881131-b645-4ecb-a2ad-966d81a95451)
![CleanShot 2024-12-02 at 16 26 49@2x](https://github.com/user-attachments/assets/6aff0f22-3da3-488e-8e4e-27bb442ece86)

## Example
- Original video: https://www.youtube.com/watch?v=_K-L9uhsBLM
- Summary: https://dl.dropbox.com/scl/fi/mdkbglfbs4m9ydeo9a2k7/video-_K-L9uhsBLM.mp4?rlkey=3wrowryg9gio1walaxhdbp2is&dl=0

## Features

- Generate concise summaries of YouTube videos
- Create engaging podcast scripts with multiple voices
- Generate AI-powered videos with synchronized podcast audio
- Support for multiple languages
- Multiple transcription options
- Multiple video generation providers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sliday/ytsum.git
cd ytsum
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (required for audio/video processing):
   - macOS: `brew install ffmpeg`
   - Ubuntu/Debian: `sudo apt-get install ffmpeg`
   - Windows: Download from [FFmpeg website](https://ffmpeg.org/download.html)

## Environment Setup

Create a `.env` file with your API keys:
```
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_api_key
LUMAAI_API_KEY=your_lumaai_api_key
RUNWAYML_API_SECRET=your_runwayml_api_key
REPLICATE_API_TOKEN=your_replicate_api_key
```

## Release
```
git tag -m "v0.0.6" v0.0.6
git push --follow-tags origin main
```

## Image
https://ghcr.io/seyar/ytsum

## Usage

### Basic Summary
```bash
python ytsum.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Generate Podcast
```bash
python ytsum.py --podcast "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Generate Video with Podcast
```bash
# Using Luma AI (faster, recommended)
python ytsum.py --podcast --lumaai "https://www.youtube.com/watch?v=VIDEO_ID"

# Using RunwayML
python ytsum.py --podcast --runwayml "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Additional Options
- `--language`: Specify output language (default: english)
- `--ignore-subs`: Force transcription even when subtitles exist
- `--fast-whisper`: Use Fast Whisper for transcription (faster)
- `--whisper`: Use OpenAI Whisper for transcription (more accurate)
- `--replicate`: Use Replicate's Incredibly Fast Whisper

## Output Files

All output files are saved in the `out` directory:
- `summary-{video_id}.txt`: Text summary
- `podcast-{video_id}.txt`: Podcast script
- `podcast-{video_id}.mp3`: Podcast audio
- `video-{video_id}.mp4`: Final video with podcast audio

## Video Generation

The tool supports two AI video generation providers:

### Luma AI (Recommended)
- Faster generation times
- High-quality cinematic videos
- Supports camera movements and scene transitions
- Maintains visual consistency
- Optional image input for style reference

### RunwayML
- High-quality video generation
- Requires input image
- Longer processing times
- Professional-grade output

Both providers:
1. Generate base images using Flux AI
2. Create video segments based on podcast content
3. Combine segments with audio
4. Support custom duration and aspect ratio

## Transcription Options

1. Fast Whisper (Default)
   - Quick transcription
   - Good accuracy
   - No API key required

2. OpenAI Whisper
   - High accuracy
   - Slower processing
   - Requires OpenAI API key

3. Replicate Whisper
   - Fastest option
   - Good accuracy
   - Requires Replicate API key

## Testing

Run the test suite:
```bash
python test_ytsum.py
```

Run specific test groups:
```bash
# Run Luma AI tests only
pytest -v -m luma

# Run RunwayML tests only
pytest -v -m runway
```

## Dependencies

- `anthropic`: Claude API for text generation
- `openai`: Whisper API for transcription and TTS
- `lumaai`: Luma AI for video generation (recommended)
- `runwayml`: RunwayML for video generation
- `replicate`: Flux AI for image generation
- `ffmpeg-python`: Audio/video processing
- `colorama`: Terminal output formatting
- `pytest`: Testing framework

## Docker

```bash
docker build -t ytsum .
docker run -it ytsum
```

```bash
docker compose run ytsum python ytsum.py --podcast --lumaai 'https://www.youtube.com/watch?v=VIDEO_ID'
```

## Webserver

```bash
curl -v -X POST 'http://ytsum:5050/process'  -H "Content-Type: application/json"  -d '{"url": "https://www.youtube.com/watch?v=E6qJ1QYscEo", "language": "russian",    "podcast": true, "video_type": "lumaai"}'
```

```bash
curl -v 'http://ytsum:5050/status/<job_id>'  -H "Content-Type: application/json"
```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
