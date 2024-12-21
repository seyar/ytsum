from flask import Flask, request, jsonify, render_template
import subprocess
import os
import time
from pathlib import Path
from uuid import uuid4
import threading
from collections import defaultdict
from youtube_url import clean_youtube_url, get_video_id

app = Flask(__name__)

# Create output directory if it doesn't exist
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
#         podcast:,
#         lumaai:
#         runwayml:
#         ignore-subs:
#         fast-whisper:
#         whisper:
#         replicate:
#     }),
# }

jobs = defaultdict(dict)

def start_background_task(cmd):
    job_id = str(uuid4())

    def run_task():
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
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

# Update the status endpoint to use the jobs dictionary
@app.route('/status/<job_id>', methods=['GET'])
def check_status(job_id):
    try:
        if job_id not in jobs:
            return jsonify({'error': 'Job not found'}), 404

        job = jobs[job_id]

        if job['status'] == 'failed':
            return jsonify({
                'status': 'failed',
                'error': job['error']
            }), 500

        if job['status'] == 'processing':
            return jsonify({
                'status': 'processing'
            })

        # Job completed
        result = job['result']
        if "ERROR:" in result.stdout:
            return jsonify({
                'error': 'Processing failed',
                'details': {
                    'return_code': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'command': ' '.join(job['cmd'])
                }
            }), 500

        # Extract video ID from the command
        url = job['cmd'][-1]  # Last argument is the URL
        video_id = get_video_id(clean_youtube_url(url))

        # Check if summary file exists
        summary_file = OUTPUT_DIR / f"summary-{video_id}.txt"
        summary_text = ""
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_text = f.read()

        return jsonify({
            'status': 'completed',
            'success': True,
            'text': summary_text,
            'summary': result.stdout,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/process', methods=['POST'])
def process_video():
    try:
        data = request.get_json()
        url = data.get('url')

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

        job_id = start_background_task(cmd)

        return jsonify({
            'status': 'processing',
            'job_id': job_id,
            'check_status_url': f'/status/{job_id}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=os.getenv("PORT"), debug=True)
