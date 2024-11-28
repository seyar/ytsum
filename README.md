# Awesome Video Summarizer (AI of course)

A Python/Shell tool that creates concise, intelligent summaries of YouTube videos using AI transcription and Claude 3.5 Sonnet for analysis.

![CleanShot 2024-11-29 at 00 06 15@2x](https://github.com/user-attachments/assets/3ba0dc0e-ceb9-49cc-8484-54b8196af79d)
![CleanShot 2024-11-29 at 00 13 21@2x](https://github.com/user-attachments/assets/1393f1ad-7e02-4996-a37e-bbf91513c777)

## Features

- Downloads audio from YouTube videos
- Multiple transcription options:
  - YouTube subtitles (when available)
  - Local processing:
    - Fast Whisper (default, fast local processing)
  - Cloud options:
    - OpenAI Whisper API (slower but potentially more accurate, pay per minute, 25MB limit)
    - Replicate Incredibly Fast Whisper (fastest cloud option, $0.0070/run)
- Multi-language support:
  - Transcription in 90+ languages with Replicate
  - Summaries in any language with Claude
- Token-efficient shorthand conversion
- AI-powered analysis using Claude 3.5 Sonnet
- Detailed content breakdown and concise summaries
- Colorful terminal output with emoji indicators

## Prerequisites

- Python 3.11+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for YouTube video downloading
- [FFmpeg](https://ffmpeg.org/) for audio conversion
- An Anthropic API key for Claude access
- An OpenAI API key for Whisper transcription (optional)
- A Replicate API key for Incredibly Fast Whisper (optional)

## Installation

1. Clone and setup:
   ```bash
   git clone https://github.com/yourusername/youtube-summarizer.git
   cd youtube-summarizer
   pip install -r requirements.txt
   ```

2. Set up API keys:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   export OPENAI_API_KEY='your-api-key-here'  # If using OpenAI Whisper
   export REPLICATE_API_TOKEN='your-api-key-here'  # If using Replicate
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

Basic usage with video URL or ID:
```bash
# Using video URL (uses Fast Whisper locally)
python ytsum.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Using just video ID (uses Fast Whisper locally)
python ytsum.py VIDEO_ID
```

Language options:
```bash
# Default: English summary
python ytsum.py VIDEO_ID

# Non-English summary
python ytsum.py VIDEO_ID --language "Russian"

# Transcribe and summarize in specific language
python ytsum.py --replicate VIDEO_ID --language "russian"
```

Transcription options:
```bash
# Default: Use Fast Whisper (local)
python ytsum.py VIDEO_ID

# Cloud options:
python ytsum.py --whisper VIDEO_ID     # OpenAI Whisper API (25MB limit)
python ytsum.py --replicate VIDEO_ID   # Replicate (no size limit)
```

## Language Support

### Transcription Languages (Replicate)
When using `--replicate`, the following languages are supported (must be lowercase):
- afrikaans, albanian, amharic, arabic, armenian
- azerbaijani, bashkir, basque, belarusian, bengali
- bosnian, breton, bulgarian, cantonese, catalan
- chinese, croatian, czech, danish, dutch, english
- estonian, faroese, finnish, french, galician
- georgian, german, greek, gujarati, haitian creole
- hausa, hawaiian, hebrew, hindi, hungarian
- icelandic, indonesian, italian, japanese, javanese
- kannada, kazakh, khmer, korean, lao, latin
- latvian, lingala, lithuanian, luxembourgish
- macedonian, malagasy, malay, malayalam, maltese
- maori, marathi, mongolian, myanmar, nepali
- norwegian, nynorsk, occitan, pashto, persian
- polish, portuguese, punjabi, romanian, russian
- sanskrit, serbian, shona, sindhi, sinhala
- slovak, slovenian, somali, spanish, sundanese
- swahili, swedish, tagalog, tajik, tamil
- tatar, telugu, thai, tibetan, turkish
- turkmen, ukrainian, urdu, uzbek, vietnamese
- welsh, yiddish, yoruba

### Summary Languages (Claude)
The `--language` parameter accepts any language name for the summary output. Claude will:
1. Analyze the transcript in its original language
2. Generate a detailed breakdown in English
3. Translate the final summary into the requested language

## Performance & Costs

| Method | Processing | Speed (150min) | Cost | Size Limit |
|--------|------------|----------------|------|------------|
| Fast Whisper | Local | ~9 min | Free | None |
| OpenAI Whisper | Cloud | ~31 min | Pay per minute | 25MB |
| Replicate | Cloud | ~2 min | $0.0070/run | None |

Notes:
- OpenAI Whisper API automatically splits files over 25MB
- Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm
- Fast Whisper works with any format FFmpeg can handle
- Replicate works best with MP3 files

## Example Output

Here's an example using a clip from Seinfeld titled "George Starts Thinking About The Future":

```
🔍 Searching for YouTube subtitles...
✅ Found YouTube subtitles!
📝 Converting to shorthand...
📝 Generating summary with Claude...
✅ Summary saved to summary-ggLvk7547_w.txt
```

The generated summary includes:
- Detailed breakdown of key moments and themes
- Important quotes and their significance
- Main topics discussed
- Concise 1-2 sentence summary in British English

## Testing

Run the test suite:
```bash
pytest test_ytsum.py -v
```

The tests cover:
- URL cleaning and normalization
- Shorthand text conversion
- Claude summarization (mocked)
- Error handling

## Dependencies

Core:
- yt-dlp: YouTube video downloading
- anthropic: Claude AI access
- ell: Claude API wrapper
- colorama: Colored terminal output

Optional:
- faster-whisper: Fast transcription option
- openai-whisper: Alternative transcription option
- replicate: Replicate Incredibly Fast Whisper

Test:
- pytest: Testing framework
- pytest-asyncio: Async test support
- pytest-mock: Mocking support

## Project Structure

```
.
├── ytsum.py      # Python implementation
├── ytsum.sh      # Shell implementation
��── prompt.txt    # Shared prompt for Claude
├── test_ytsum.py # Test suite
└── README.md
```

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

This project is licensed under the MIT License - see the LICENSE file for details.

## Optional Dependencies

```bash
# Local transcription (default):
pip install faster-whisper

# Cloud transcription:
pip install openai      # For OpenAI Whisper API
pip install replicate   # For Replicate
```

## API Keys

```bash
# Required:
export ANTHROPIC_API_KEY='your-key'  # For Claude summaries

# Optional (for cloud transcription):
export OPENAI_API_KEY='your-key'     # For OpenAI Whisper
export REPLICATE_API_TOKEN='your-key' # For Replicate
```
