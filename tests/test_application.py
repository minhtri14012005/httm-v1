"""API/model integration on upstream sample imagery, never personal attachments."""
import io
import json
import os
from pathlib import Path
import time
import unittest
import uuid
from unittest.mock import Mock, patch

import av
import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT=Path(__file__).resolve().parents[1]
TEST_DATA=ROOT/'results'/('app_test_'+uuid.uuid4().hex)
os.environ['APP_DATA_DIR']=str(TEST_DATA)
from backend.app import app


class ApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(app);cls.client.__enter__()
        for _ in range(180):
            s=cls.client.get('/api/status').json()
            if s['runtime']['state']!='loading':break
            time.sleep(.25)
        if s['runtime']['state']!='ready':raise RuntimeError(s)
        cls.runtime=s['runtime']

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None,None,None)

    def test_full_image_video_lifecycle(self):
        c=self.client
        r=c.post('/api/people',json={'name':'Upstream sample (test)','notes':'Software test only'})
        self.assertEqual(r.status_code,200,r.text);pid=r.json()['id']
        image=ROOT/'samples/img2.jpeg'
        for _ in range(2):
            with image.open('rb') as f:r=c.post(f'/api/people/{pid}/photos',files={'file':('sample.jpeg',f,'image/jpeg')})
            self.assertEqual(r.status_code,200,r.text)
        people=c.get('/api/people').json();self.assertEqual(len(people[0]['photos']),2)
        photo=people[0]['photos'][0]['id']
        self.assertEqual(c.get('/api/photos/'+photo).status_code,200)
        c.delete('/api/photos/'+photo)
        r=c.put('/api/people/'+pid,json={'name':'Sample reference','notes':'Two uploads, one removed','watched':True})
        self.assertEqual(r.status_code,200,r.text)
        bad=c.post(f'/api/people/{pid}/photos',files={'file':('bad.jpg',b'not an image','image/jpeg')})
        self.assertEqual(bad.status_code,422,bad.text)
        # A short fixture repeats upstream sample imagery; not real motion evidence.
        video=TEST_DATA/'fixture.mp4';frame=cv2.imread(str(image))
        h,w=frame.shape[:2];w-=w%2;h-=h%2;frame=cv2.resize(frame,(w,h))
        with av.open(str(video),'w') as out:
            s=out.add_stream('libx264',rate=8);s.width=w;s.height=h;s.pix_fmt='yuv420p'
            for _ in range(16):
                for packet in s.encode(av.VideoFrame.from_ndarray(frame,format='bgr24')):out.mux(packet)
            for packet in s.encode():out.mux(packet)
        with video.open('rb') as f:r=c.post('/api/jobs/video',files={'file':('upstream_fixture.mp4',f,'video/mp4')})
        self.assertEqual(r.status_code,200,r.text);jid=r.json()['id']
        overlap=c.post('/api/people',json={'name':'Should be busy'})
        self.assertEqual(overlap.status_code,409)
        with c.websocket_connect('/ws') as ws:
            messages=[]
            for _ in range(100):
                message=ws.receive_json();messages.append(message['type'])
                if message['type']=='confirmed':
                    stored=c.get('/api/events',params={'job_id':jid}).json()
                    self.assertIn(message['event']['id'],{event['id'] for event in stored})
                if message['type']=='finished':break
        job=next(j for j in c.get('/api/jobs').json() if j['id']==jid)
        self.assertEqual(job['status'],'completed',job)
        self.assertGreater(job['processed_frames'],0)
        self.assertEqual(job['sampling_fps'],8)
        self.assertEqual(job['decoded_frames'],16)
        self.assertEqual(job['skipped_frames'],job['decoded_frames']-job['processed_frames'])
        ev=c.get('/api/events',params={'job_id':jid}).json()
        self.assertEqual(len(ev),1,ev)
        self.assertEqual(messages.count('confirmed'),1)
        self.assertEqual(ev[0]['person_id'],pid)
        self.assertEqual(c.get('/api/events/'+ev[0]['id']+'/image').status_code,200)
        r=c.get('/api/jobs/'+jid+'/video');self.assertEqual(r.status_code,200)
        with av.open(io.BytesIO(r.content)) as out:
            self.assertGreater(len(list(out.decode(video=0))),0)
        # Exercise cancellation without depending on camera or user video.
        with video.open('rb') as f:r=c.post('/api/jobs/video',files={'file':('cancel_fixture.mp4',f,'video/mp4')})
        cancel_id=r.json()['id'];c.post('/api/jobs/stop')
        for _ in range(100):
            job2=next(j for j in c.get('/api/jobs').json() if j['id']==cancel_id)
            if job2['status'] not in ('running','stopping'):break
            time.sleep(.1)
        self.assertEqual(job2['status'],'cancelled',job2)
        self.assertEqual(c.post('/api/live/start',json={'source':'unknown'}).status_code,422)
        camera=c.get('/api/cameras').json();self.assertEqual([x['key'] for x in camera],['laptop','camo'])
        # Test transition ordering without opening any real user's camera.
        fake=Mock();fake.active.return_value=True;fake.current={'kind':'live'}
        fake.start.return_value={'id':'mock_camera_switch','status':'running'}
        with patch.object(app.state,'worker',fake):
            r=c.post('/api/live/start',json={'source':'camo'})
            self.assertEqual(r.status_code,200,r.text)
        ordered=[x[0] for x in fake.mock_calls]
        self.assertLess(ordered.index('stop'),ordered.index('wait'))
        self.assertLess(ordered.index('wait'),ordered.index('start'))
        c.delete('/api/people/'+pid)
        self.assertEqual(c.get('/api/people').json(),[])
        evidence={'runtime':self.runtime,'job':job,'events':ev,'websocket_message_types':messages,
                  'cancel_job':job2,'camera_enumeration':camera,'test_media':'repeated upstream sample, not real-world accuracy test'}
        (TEST_DATA/'verification.json').write_text(json.dumps(evidence,indent=2),encoding='utf8')
        print('APPLICATION_TEST_EVIDENCE',TEST_DATA)


if __name__=='__main__':unittest.main()
