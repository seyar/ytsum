import pytest
import json
from pathlib import Path
from ytsum import (
    clean_youtube_url,
    to_shorthand,
    summarize_with_claude,
    convert_audio_format,
    get_video_metadata,
    transcribe_with_replicate,
    transcribe_with_openai_whisper,
    process_metadata_description,
    split_audio_into_chunks,
    get_youtube_subtitles,
    get_language_code
)

def test_clean_youtube_url():
    # Test video ID only
    assert clean_youtube_url("ggLvk7547_w") == "https://www.youtube.com/watch?v=ggLvk7547_w"
    
    # Test full URLs
    assert clean_youtube_url("https://www.youtube.com/watch?v=ggLvk7547_w") == "https://www.youtube.com/watch?v=ggLvk7547_w"
    assert clean_youtube_url("https://youtu.be/ggLvk7547_w") == "https://www.youtube.com/watch?v=ggLvk7547_w"
    
    # Test with extra parameters
    assert clean_youtube_url("https://www.youtube.com/watch?v=ggLvk7547_w&t=123") == "https://www.youtube.com/watch?v=ggLvk7547_w"

def test_to_shorthand():
    # Basic replacements
    assert to_shorthand("you are") == "u r"
    
    # Case-insensitive test
    assert to_shorthand("I am going to see you later") == "im going 2 c u l8r"
    assert to_shorthand("i am going to see you later") == "im going 2 c u l8r"
    
    # Article removal
    assert to_shorthand("the cat and a dog") == "cat and dog"

@pytest.mark.asyncio
async def test_summarize_with_claude(mocker):
    # Mock prompt file
    mock_prompt = "Test prompt with {language}"
    mocker.patch("pathlib.Path.open", mocker.mock_open(read_data=mock_prompt))
    
    # Mock Claude response
    mock_summary = "<detailed_breakdown>Test breakdown</detailed_breakdown>\n<summary>Test summary</summary>"
    
    # Create mock decorator
    def mock_decorator(*args, **kwargs):
        def mock_function(func):
            return lambda x: mock_summary
        return mock_function
    
    # Patch ell.simple
    mocker.patch("ell.simple", mock_decorator)
    
    # Test with default language
    result = summarize_with_claude("test transcript")
    assert result == mock_summary
    
    # Test with specific language
    result = summarize_with_claude("test transcript", language="russian")
    assert result == mock_summary

def test_convert_audio_format(mocker):
    # Mock FFmpeg subprocess call
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0
    
    # Test basic MP3 conversion
    result = convert_audio_format("test.m4a", "mp3")
    assert result == "test.mp3"
    
    # Verify FFmpeg was called with correct parameters
    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "ffmpeg"
    assert "-acodec" in args
    assert "libmp3lame" in args
    assert "-ac" in args
    assert args[args.index("-ac") + 1] == "2"  # Stereo by default
    assert "-b:a" in args
    assert args[args.index("-b:a") + 1] == "192k"  # Default bitrate
    
    # Test mono conversion with custom bitrate
    result = convert_audio_format("test.m4a", "mp3", bitrate="32k", mono=True)
    assert result == "test.mp3"
    
    args = mock_run.call_args[0][0]
    assert "-ac" in args
    assert args[args.index("-ac") + 1] == "1"  # Mono
    assert "-b:a" in args
    assert args[args.index("-b:a") + 1] == "32k"  # Custom bitrate

def test_get_video_metadata(mocker):
    # Mock yt-dlp JSON output
    mock_metadata = {
        "title": "Test Video",
        "channel": "Test Channel",
        "upload_date": "20240315",
        "duration_string": "1:23",
        "view_count": 12345,
        "description": "Test description with promotional content",
        "tags": ["tag1", "tag2", "tag3"]
    }
    
    # Mock subprocess run
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.stdout = json.dumps(mock_metadata)
    
    # Mock metadata processing
    mock_process = mocker.patch("ytsum.process_metadata_description")
    mock_process.return_value = "Processed description"
    
    result = get_video_metadata("test_id")
    
    # Verify metadata formatting
    assert "Title: Test Video" in result
    assert "Channel: Test Channel" in result
    assert "Views: 12,345" in result
    assert "Description: Processed description" in result
    
    # Verify processing was called
    mock_process.assert_any_call(mock_metadata["description"])
    mock_process.assert_any_call(" ".join(mock_metadata["tags"]))

def test_transcribe_with_replicate(mocker):
    # Mock FFmpeg conversion
    mock_convert = mocker.patch("ytsum.convert_audio_format")
    mock_convert.return_value = "test.mp3"
    
    # Mock file operations
    mock_file = mocker.mock_open(read_data=b"test audio data")
    mocker.patch("builtins.open", mock_file)
    
    # Mock os.path instead of pathlib.Path
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.getsize", return_value=1024)
    
    # Mock Replicate API call
    mock_replicate = mocker.patch("replicate.run")
    mock_replicate.return_value = {"text": "test transcript"}
    
    # Test transcription
    result = transcribe_with_replicate("test.m4a")
    assert result is True
    
    # Verify basic flow
    mock_convert.assert_called_once()
    
    # Verify Replicate was called correctly
    mock_replicate.assert_called_once()
    call_args = mock_replicate.call_args[1]
    assert "input" in call_args
    assert call_args["input"]["batch_size"] == 64
    
    # Verify file operations
    mock_file.assert_any_call("test.mp3", "rb")  # Check file was opened for reading
    mock_file.assert_any_call("test.txt", "w", encoding="utf-8")  # Check transcript was written

def test_process_metadata_description(mocker):
    # Mock Ell response
    mock_response = "Test summary"
    
    # Create mock decorator
    def mock_decorator(*args, **kwargs):
        def mock_function(func):
            return lambda x: mock_response
        return mock_function
    
    # Patch ell.simple
    mocker.patch("ell.simple", mock_decorator)
    
    # Test sample metadata
    test_metadata = {
        "description": """
        From Seinfeld Season 8 Episode 12 'The Money': Jerry buys back a car his parents sold.
        Watch all episodes on Netflix!
        """,
        "tags": ["seinfeld", "jerry", "george", "kramer"]
    }
    
    # Test description processing
    result = process_metadata_description(test_metadata["description"])
    assert result == mock_response
    
    # Test tags processing
    result = process_metadata_description(" ".join(test_metadata["tags"]))
    assert result == mock_response

def test_split_audio_into_chunks(mocker):
    # Mock file size (30MB)
    mocker.patch("os.path.getsize", return_value=30 * 1024 * 1024)
    
    # Mock ffprobe duration check
    mock_probe = mocker.MagicMock()
    mock_probe.stdout = "300.0\n"  # 5 minutes with newline
    mock_run = mocker.patch("subprocess.run", return_value=mock_probe)
    
    # Mock directory operations
    mocker.patch("os.makedirs")
    mocker.patch("os.path.dirname", return_value="/tmp")
    mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))
    
    # Test splitting
    chunks = split_audio_into_chunks("test.mp3")
    assert chunks is not None
    assert len(chunks) == 2  # Should split into 2 chunks for 30MB file
    
    # Verify FFmpeg calls
    ffmpeg_calls = [
        call for call in mock_run.call_args_list 
        if 'ffmpeg' in call.args[0][0]
    ]
    assert len(ffmpeg_calls) == 2  # Two chunks
    
    # Verify chunk paths
    assert all('chunk_' in path for path in chunks)
    assert all(path.endswith('.mp3') for path in chunks)

def test_transcribe_with_openai_whisper(mocker):
    # Mock file operations
    file_size_mock = mocker.patch("os.path.getsize")
    file_size_mock.side_effect = [
        20 * 1024 * 1024,  # Initial file size for supported format test
        20 * 1024 * 1024,  # Size check for transcription
        30 * 1024 * 1024,  # Initial size for unsupported format
        20 * 1024 * 1024,  # Size after compression
        20 * 1024 * 1024,  # Size check for transcription
        30 * 1024 * 1024,  # Initial size for large file test
        20 * 1024 * 1024,  # Size after compression
        20 * 1024 * 1024,  # Size check for transcription
    ]
    
    # Mock file paths and operations
    mocker.patch("pathlib.Path.suffix", ".m4a")  # Supported format
    mocker.patch("os.path.exists", return_value=True)
    mocker.patch("os.path.dirname", return_value="/tmp")
    mocker.patch("os.path.join", side_effect=lambda *args: "/".join(args))
    mocker.patch("os.makedirs")
    mock_remove = mocker.patch("os.remove")
    mock_rmdir = mocker.patch("os.rmdir")
    
    # Mock OpenAI client and response
    mock_client = mocker.MagicMock()
    mock_transcription = mocker.MagicMock()
    mock_transcription.text = "test transcript"
    mock_client.audio.transcriptions.create.return_value = mock_transcription
    mock_openai = mocker.patch("openai.OpenAI", return_value=mock_client)
    
    # Mock file operations
    mock_file = mocker.mock_open(read_data=b"test audio data")
    mocker.patch("builtins.open", mock_file)
    
    # Mock environment variable
    mocker.patch("os.getenv", return_value="test-api-key")
    
    # Mock audio conversion
    mock_convert = mocker.patch("ytsum.convert_audio_format")
    mock_convert.return_value = "test.mp3"
    
    # Test 1: Transcription with supported format
    result = transcribe_with_openai_whisper("test.m4a")
    assert result is True
    assert not mock_convert.called  # No conversion needed
    
    # Test 2: Unsupported format
    mocker.patch("pathlib.Path.suffix", ".aac")
    result = transcribe_with_openai_whisper("test.aac")
    assert result is True
    assert mock_convert.called
    
    # Verify compression settings
    args = mock_convert.call_args[1]
    assert args["bitrate"] == "32k"
    assert args["mono"] is True
    
    # Test 3: Large file that compresses successfully
    mock_convert.reset_mock()
    result = transcribe_with_openai_whisper("test.m4a")
    assert result is True
    
    # Verify compression was used
    assert mock_convert.called
    assert mock_client.audio.transcriptions.create.called
    
    # Verify compression settings
    args = mock_convert.call_args[1]
    assert args["bitrate"] == "32k"
    assert args["mono"] is True

def test_get_language_code(mocker):
    # Mock Ell response
    def mock_decorator(*args, **kwargs):
        def mock_function(func):
            def wrapper(lang: str):
                # Simple mapping for testing
                codes = {
                    "english": "en",
                    "russian": "ru",
                    "spanish": "es",
                    "invalid": "xyz",  # Should fallback to en
                }
                return codes.get(lang.lower(), "en")
            return wrapper
        return mock_function
    
    # Patch ell.simple
    mocker.patch("ell.simple", mock_decorator)
    
    # Test valid languages
    assert get_language_code("English") == "en"
    assert get_language_code("Russian") == "ru"
    assert get_language_code("Spanish") == "es"
    
    # Test fallbacks
    assert get_language_code("Invalid") == "en"
    assert get_language_code("") == "en"

def test_get_youtube_subtitles(mocker):
    # Mock language code conversion
    mock_get_code = mocker.patch("ytsum.get_language_code")
    mock_get_code.side_effect = lambda x: {
        "Russian": "ru",
        "English": "en"
    }.get(x, "en")
    
    # Mock subprocess for yt-dlp
    mock_run = mocker.patch("subprocess.run")
    mock_run.return_value.returncode = 0  # Ensure subprocess succeeds
    
    # Mock file existence checks
    mock_exists = mocker.patch("os.path.exists")
    
    # Mock file operations
    mock_file = mocker.mock_open(read_data="Test subtitles")
    mocker.patch("builtins.open", mock_file)
    
    # Test 1: Found subtitles in requested language
    mock_run.return_value.stdout = """
    [info] Writing video subtitles to: test_path.ru.vtt
    [download] 100% of 15.00KiB
    """
    mock_exists.side_effect = lambda x: "test_path.ru.vtt" in x  # Match exact file
    result = get_youtube_subtitles("test_url", "test_path", "Russian")
    assert result == "test_path.ru.txt"  # We return the converted txt file
    assert mock_get_code.called_with("Russian")
    
    # Test 2: Found English subtitles as fallback
    mock_run.return_value.stdout = """
    [info] Writing video subtitles to: test_path.en.vtt
    [download] 100% of 15.00KiB
    """
    mock_exists.side_effect = lambda x: "test_path.en.vtt" in x  # Match exact file
    result = get_youtube_subtitles("test_url", "test_path", "Russian")
    assert result == "test_path.en.txt"  # We return the converted txt file
    
    # Test 3: No subtitles available
    mock_run.return_value.stdout = "No subtitles available"
    mock_exists.side_effect = lambda x: False  # No files exist
    result = get_youtube_subtitles("test_url", "test_path", "Russian")
    assert result is None
    
    # Verify yt-dlp was called correctly
    calls = mock_run.call_args_list
    assert any("--write-subs" in str(call) for call in calls)
    assert any("--sub-langs" in str(call) for call in calls)
    assert any("ru" in str(call) for call in calls)
    
    # Verify file existence checks
    assert mock_exists.call_count >= 2  # At least one check per test
    assert any("test_path.ru.vtt" in str(call) for call in mock_exists.call_args_list)
    assert any("test_path.en.vtt" in str(call) for call in mock_exists.call_args_list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 