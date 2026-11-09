"""Saved video evidence must remain readable beyond the recent event feed."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from backend.app import app
from backend.store import Store


class VideoHistoryTests(unittest.TestCase):
    def test_old_evidence_is_readable_after_500_newer_events(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'evidence').mkdir()
            store=Store(root);jid=uuid4().hex;eid=uuid4().hex
            store.put_job({'id':jid,'kind':'video','status':'completed'})
            path=f'evidence/{eid}.jpg';(root/path).write_bytes(b'old evidence fixture')
            store.put_event({'id':eid,'job_id':jid,'person_id':'anonymous','confirmed_at':.5,'evidence':path})
            for _ in range(501):
                store.put_event({'id':uuid4().hex,'job_id':jid,'person_id':'anonymous','confirmed_at':1})
            self.assertNotIn(eid,{e['id'] for e in store.events()})
            with patch.object(app.state,'store',store,create=True),patch('backend.app.DATA',root):
                with patch.object(app.state,'worker',None,create=True):
                    client=TestClient(app)
                    try:
                        response=client.get(f'/api/events/{eid}/image')
                        self.assertEqual(response.status_code,200)
                        self.assertEqual(response.content,b'old evidence fixture')
                        self.assertEqual(client.get('/api/events/'+uuid4().hex+'/image').status_code,404)
                        (root/path).unlink()
                        self.assertEqual(client.get(f'/api/events/{eid}/image').status_code,404)
                    finally:client.close()


if __name__=='__main__':unittest.main()
