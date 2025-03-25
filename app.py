from flask import Flask, request, jsonify, render_template
import subprocess
import os
import time
from pathlib import Path
from uuid import uuid4
import threading
from collections import defaultdict
from youtube_url import clean_youtube_url, get_video_id
import re
from pathlib import Path

app = Flask(__name__)

# Create output directory if it doesn't exist python -m unittest test_app.py -v
OUTPUT_DIR = Path("/app/out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# {
#     method: 'POST',
#     headers: {
#         'Content-Type': 'application/json',
#     },
#     body: JSON.stringify({
#         url: 'https://www.youtube.com/watch?v=VIDEO_ID',
#         language: 'russian',
#         podcast: true,
#         lumaai:
#         runwayml:
#         ignore-subs:
#         fast-whisper:
#         whisper:
#         replicate:
#     }),
# }
# Add after other imports


def get_metadata_description(url):
    """Extract metadata description from summary file if available"""
    try:
        video_id = get_video_id(clean_youtube_url(url))
        description_file = OUTPUT_DIR / f"description-{video_id}.txt"

        if not description_file.exists():
            return None

        with open(description_file, 'r', encoding='utf-8') as f:
            content = f.read()

        if content:
            return content

        return None
    except Exception as e:
        print(f"Error getting metadata: {e}", flush=True)
        return None

def get_podcast_audio(url):
    """Get podcast audio file path if available"""
    video_id = get_video_id(clean_youtube_url(url))
    audio_file = OUTPUT_DIR / f"podcast-{video_id}.mp3"

    if audio_file.exists():
        return str(audio_file)

    return None

jobs = defaultdict(dict)

def start_background_task(cmd, job_id):
    def run_task():
        try:
            error_text = "ERROR:"
            result = subprocess.run(cmd, capture_output=True, text=True)
            if error_text in result.stdout or error_text in result.stderr:
                jobs[job_id] = {
                    'status': 'failed',
                    'result': result.stdout,
                    'error': str(e),
                    'cmd': cmd
                }

            jobs[job_id] = {
                'status': 'completed',
                'result': result,
                'cmd': cmd
            }
        except Exception as e:
            jobs[job_id] = {
                'status': 'failed',
                'error': str(e),
                'cmd': cmd
            }

    # Store initial job status
    jobs[job_id] = {
        'status': 'processing',
        'cmd': cmd
    }

    # Start the background thread
    thread = threading.Thread(target=run_task)
    thread.daemon = True
    thread.start()

    return job_id

def get_summary_text(url):
    # Validate input
    if not url or not isinstance(url, str):
        return None

    try:
        # Extract video ID from the command
        video_id = get_video_id(clean_youtube_url(url))
        if not video_id:
            return None

        # Check if summary file exists
        summary_file = OUTPUT_DIR / f"summary-{video_id}.txt"
        if not summary_file.exists():
            return None

        # Read and validate file content
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_text = f.read().strip()
            return summary_text if summary_text else None

    except (OSError, IOError) as e:
        print(f"Can not reading summary file: {e}", flush=True)
        return None
    except Exception as e:
        print(f"Unexpected error in get_summary_text: {e}", flush=True)
        return None

# {
#     "status": "processing",
#     "metadata": "...",  // Optional, included if available
#     "audio_file": "...", // Optional, included if available
#     // When completed:
#     "success": true,
#     "text": "...",
#     "stdout": "..."
#     }
# Update the status endpoint to use the jobs dictionary
@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    print(f"Check status for job={job_id}", flush=True)
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = jobs[job_id]

        if job['status'] == 'failed':
            return jsonify({
                'status': 'failed',
                'error': job['error']
            }), 500

        url = job['cmd'][-1]  # Last argument is the URL
        response = {'status': 'processing'}

        # Check for metadata first
        metadata = get_metadata_description(url)
        if metadata:
            response['metadata'] = metadata

        # Check for audio file next
        audio_file = get_podcast_audio(url)
        if audio_file:
            response['audio_file'] = audio_file

        # Check for final summary
        if job['status'] == 'completed':
            result = job['result']

            if "ERROR:" in result.stdout:
                print(f"Processing failed. Maybe no cookie {result.stderr} {result.stdout}")
                return jsonify({
                    'error': 'Processing failed',
                    'details': {
                        'return_code': result.returncode,
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'command': ' '.join(job['cmd'])
                    }
                }), 500

            summary_text = get_summary_text(url)
            if summary_text:
                response['status'] = 'completed'
                response['success'] = True
                response['text'] = summary_text
                response['stdout'] = result.stdout

        return jsonify(response)

    except Exception as e:
        print(f"Error job={str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video():
    try:
        data = request.get_json()
        url = data.get('url')
        print(f"Process video url {url}", flush=True)

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        language = data.get('language', 'english')
        podcast = data.get('podcast', False)
        lumaai = data.get('lumaai', False)
        runwayml = data.get('runwayml', False)
        ignore_subs = data.get('ignore-subs', False)
        fast_whisper = data.get('fast-whisper', False)
        whisper = data.get('whisper', False)
        replicate = data.get('replicate', False)

        # Build command
        cmd = ['python3', 'ytsum.py', '--language', language]

        if podcast:
            cmd.append('--podcast')
        if lumaai:
            cmd.append('--lumaai')
        if runwayml:
            cmd.append('--runwayml')
        if ignore_subs:
            cmd.append('--ignore-subs')
        if fast_whisper:
            cmd.append('--fast-whisper')
        if whisper:
            cmd.append('--whisper')
        if replicate:
            cmd.append('--replicate')

        cmd.append(url)
        print(f"Process cmd: {cmd}", flush=True)

        job_id = str(uuid4())
        summary_text = get_summary_text(url)
        if summary_text is None:
            start_background_task(cmd, job_id)
            return jsonify({
                'status': 'processing',
                'job_id': job_id,
                'check_status_url': f'/status/{job_id}'
            })
        else:
            return jsonify({
                'status': 'completed',
                'job_id': job_id,
                'text': summary_text
            })

    except Exception as e:
        print(f"Process cmd: {str(e)}", flush=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=os.getenv("PORT"), debug=True)
