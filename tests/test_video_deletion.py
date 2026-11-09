import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.app import app
from backend.store import Store


class VideoDeletionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for folder in ('videos', 'evidence', 'photos'):
            (self.root / folder).mkdir()
        self.store = Store(self.root)
        self.client = TestClient(app)
        self.patches = [patch.object(app.state, 'store', self.store, create=True),
                        patch.object(app.state, 'worker', None, create=True)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        self.client.close()
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def add_video(self, status='completed', has_input=True):
        jid, eid, input_id = uuid4().hex, uuid4().hex, uuid4().hex
        output = f'videos/{jid}_annotated.mp4'
        evidence = f'evidence/{eid}.jpg'
        job = {'id': jid, 'kind': 'video', 'name': 'source.mp4', 'status': status, 'output': output}
        if has_input:
            job['input'] = f'videos/{input_id}.input'
            (self.root / job['input']).write_bytes(b'input')
        (self.root / output).write_bytes(b'output')
        (self.root / evidence).write_bytes(b'evidence')
        self.store.put_job(job)
        self.store.put_event({'id': eid, 'job_id': jid, 'person_id': 'person', 'evidence': evidence})
        return job, evidence

    def test_delete_one_removes_owned_files_and_events_only(self):
        job, evidence = self.add_video()
        other, _ = self.add_video()
        unrelated = self.root / 'photos' / 'keep.jpg'
        unrelated.write_bytes(b'photo')
        response = self.client.delete('/api/videos/' + job['id'])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'deleted': 1, 'untracked_inputs': 0})
        self.assertEqual(self.store.events(job['id']), [])
        for rel in (job['input'], job['output'], evidence):
            self.assertFalse((self.root / rel).exists())
        self.assertTrue((self.root / other['output']).exists())
        self.assertTrue(unrelated.exists())
        self.assertEqual(self.client.delete('/api/videos/' + job['id']).status_code, 404)

    def test_delete_all_blocks_running_and_preserves_live_jobs(self):
        old, _ = self.add_video(has_input=False)
        running, _ = self.add_video('running')
        live = {'id': uuid4().hex, 'kind': 'live', 'status': 'completed'}
        self.store.put_job(live)
        self.assertEqual(self.client.delete('/api/videos').status_code, 409)
        self.assertEqual(self.client.delete('/api/videos/' + running['id']).status_code, 409)
        self.assertTrue((self.root / old['output']).exists())
        running['status'] = 'completed'
        # Updating this fixture directly avoids SQLite REPLACE on a referenced row.
        import json
        with self.store.connection() as c:
            c.execute('UPDATE jobs SET payload=? WHERE id=?', (json.dumps(running), running['id']))
        response = self.client.delete('/api/videos')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'deleted': 2, 'untracked_inputs': 1})
        self.assertEqual([j['id'] for j in self.store.jobs()], [live['id']])

    def test_finished_status_is_blocked_until_worker_thread_exits(self):
        job, _ = self.add_video()
        worker = Mock()
        import threading
        worker.lock = threading.RLock()
        worker.current = job
        worker.thread.is_alive.return_value = True
        with patch.object(app.state, 'worker', worker):
            self.assertEqual(self.client.delete('/api/videos/' + job['id']).status_code, 409)
        self.assertTrue((self.root / job['output']).exists())

    def test_unsafe_path_is_rejected_without_deleting(self):
        job, _ = self.add_video()
        job['input'] = '../keep.input'
        import json
        with self.store.connection() as c:
            c.execute('UPDATE jobs SET payload=? WHERE id=?', (json.dumps(job), job['id']))
        self.assertEqual(self.client.delete('/api/videos/' + job['id']).status_code, 422)
        self.assertTrue((self.root / job['output']).exists())

    def test_completed_video_can_be_downloaded_without_changing_playback(self):
        job, _ = self.add_video()
        with patch('backend.app.DATA', self.root):
            playback = self.client.get('/api/jobs/' + job['id'] + '/video')
            download = self.client.get('/api/jobs/' + job['id'] + '/video?download=1')
        self.assertEqual(playback.status_code, 200)
        self.assertEqual(download.status_code, 200)
        self.assertEqual(playback.content, b'output')
        self.assertEqual(download.content, playback.content)
        self.assertNotIn('content-disposition', playback.headers)
        self.assertEqual(download.headers['content-disposition'],
                         'attachment; filename="source_annotated.mp4"')
        self.assertTrue((self.root / job['output']).exists())

    def test_download_requires_completed_job_and_owned_output(self):
        running, _ = self.add_video('running')
        cancelled, _ = self.add_video('cancelled')
        with patch('backend.app.DATA', self.root):
            self.assertEqual(self.client.get('/api/jobs/' + running['id'] + '/video?download=1').status_code, 404)
            self.assertEqual(self.client.get('/api/jobs/' + cancelled['id'] + '/video').status_code, 200)
            self.assertEqual(self.client.get('/api/jobs/' + cancelled['id'] + '/video?download=1').status_code, 404)
            cancelled['output'] = 'photos/unrelated.jpg'
            import json
            with self.store.connection() as connection:
                connection.execute('UPDATE jobs SET payload=? WHERE id=?', (json.dumps(cancelled), cancelled['id']))
            self.assertEqual(self.client.get('/api/jobs/' + cancelled['id'] + '/video').status_code, 404)


if __name__ == '__main__':
    unittest.main()
