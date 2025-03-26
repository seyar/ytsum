import unittest
from unittest.mock import patch, MagicMock
import json
from app import app, jobs, get_summary_text
from pathlib import Path

class TestApp(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()
        # Clear jobs before each test
        jobs.clear()

    def tearDown(self):
        # Clean up after each test
        jobs.clear()

    def test_process_video_missing_url(self):
        response = self.client.post('/process',
                                  data=json.dumps({}),
                                  content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'URL is required')

    @patch('app.start_background_task')
    @patch('app.get_summary_text')
    def test_process_new_video(self, mock_get_summary, mock_start_task):
        mock_get_summary.return_value = None
        test_url = 'https://youtube.com/watch?v=test123'

        response = self.client.post('/process',
                                  data=json.dumps({'url': test_url}),
                                  content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'processing')
        self.assertTrue('job_id' in data)
        self.assertTrue('check_status_url' in data)
        mock_start_task.assert_called_once()

    # @patch('app.get_summary_text')
    # def test_process_existing_summary(self, mock_get_summary):
    #     # Clear any cached jobs from previous tests
    #     jobs.clear()

    #     test_url = 'https://youtube.com/watch?v=test123'
    #     test_summary = "Existing summary"
    #     mock_get_summary.return_value = test_summary

    #     response = self.client.post('/process',
    #                               data=json.dumps({'url': test_url}),
    #                               content_type='application/json')

    #     self.assertEqual(response.status_code, 200)
    #     data = json.loads(response.data)
    #     self.assertEqual(data['status'], 'processing')

    #     # Verify the job was created correctly
    #     job_id = data['job_id']
    #     self.assertIn(job_id, jobs)
    #     self.assertEqual(jobs[job_id]['status'], 'completed')
    #     self.assertEqual(jobs[job_id]['text'], test_summary)

    #     # Verify the check_status_url is correct
    #     self.assertEqual(data['check_status_url'], f'/status/{job_id}')

    #     # Verify get_summary_text was called with the correct URL
    #     mock_get_summary.assert_called_once_with(test_url)

    def test_check_status_invalid_job(self):
        response = self.client.get('/status/invalid-job-id')
        self.assertEqual(response.status_code, 404)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Job not found')

    # def test_check_status_processing(self):
    #     job_id = 'test-job'
    #     jobs[job_id] = {
    #         'status': 'processing',
    #         'cmd': ['test', 'command']
    #     }

    #     response = self.client.get(f'/status/{job_id}')
    #     self.assertEqual(response.status_code, 200)
    #     data = json.loads(response.data)
    #     self.assertEqual(data['status'], 'processing')

    def test_check_status_failed(self):
        job_id = 'test-job'
        error_message = 'Test error'
        jobs[job_id] = {
            'status': 'failed',
            'error': error_message,
            'cmd': ['test', 'command']
        }

        response = self.client.get(f'/status/{job_id}')
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'failed')
        self.assertEqual(data['error'], error_message)

    def test_check_status_completed_with_error(self):
        job_id = 'test-job'
        jobs[job_id] = {
            'status': 'completed',
            'result': MagicMock(
                stdout='ERROR: Something went wrong',
                stderr='Error details',
                returncode=1
            ),
            'cmd': ['test', 'command', 'http://test.url']
        }

        response = self.client.get(f'/status/{job_id}')
        self.assertEqual(response.status_code, 500)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'Invalid YouTube video ID: None')

    # @patch('app.get_summary_text')
    # @patch('app.get_cookie_path')
    # def test_check_status_completed_success(self, mock_get_cookie_path, mock_get_summary):
    #     job_id = 'test-job'
    #     test_summary = "Test summary"
    #     mock_get_summary.return_value = test_summary
    #     mock_get_cookie_path.return_value = '/app/data/www.youtube.com_cookies.txt'

    #     jobs[job_id] = {
    #         'status': 'completed',
    #         'result': MagicMock(stdout='Success'),
    #         'cmd': ['test', 'command', 'https://www.youtube.com/watch?v=UHCbb-Nl78I']
    #     }

    #     response = self.client.get(f'/status/{job_id}')
    #     self.assertEqual(response.status_code, 200)
    #     data = json.loads(response.data)
    #     self.assertEqual(data['status'], 'completed')
    #     self.assertTrue(data['success'])
    #     self.assertEqual(data['text'], test_summary)

    def test_get_summary_text_invalid_input(self):
        self.assertIsNone(get_summary_text(None))
        self.assertIsNone(get_summary_text(''))
        self.assertIsNone(get_summary_text(123))

    @patch('app.get_video_id')
    def test_get_summary_text_invalid_video_id(self, mock_get_video_id):
        mock_get_video_id.return_value = None
        self.assertIsNone(get_summary_text('https://youtube.com/invalid'))

    @patch('app.get_video_id')
    def test_get_summary_text_file_not_found(self, mock_get_video_id):
        mock_get_video_id.return_value = 'test123'
        self.assertIsNone(get_summary_text('https://youtube.com/watch?v=test123'))

    @patch('app.get_video_id')
    def test_get_summary_text_empty_file(self, mock_get_video_id):
        mock_get_video_id.return_value = 'test123'
        test_file = Path('/app/out/summary-test123.txt')

        # Create empty file
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text('')

        self.assertIsNone(get_summary_text('https://youtube.com/watch?v=test123'))

        # Cleanup
        test_file.unlink()

if __name__ == '__main__':
    unittest.main()
