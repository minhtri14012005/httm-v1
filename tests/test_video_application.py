"""Real-model Video Upload lifecycle in an isolated store, never cameras."""
from contextlib import ExitStack
import hashlib,io,json,threading,time,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import av,cv2,numpy as np,torch
from fastapi.testclient import TestClient
from backend import app as api
from backend.store import Store

ROOT=Path(__file__).resolve().parents[1]


class VideoApplicationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stack=ExitStack();cls.addClassCleanup(cls.stack.close)
        tmp=cls.stack.enter_context(TemporaryDirectory(prefix='video_app_',dir=ROOT/'results'))
        cls.root=Path(tmp).resolve()
        cfg=json.loads((ROOT/'config.json').read_text());cfg['mode']='cpu';cfg['device']=None
        config_path=cls.root/'config.json';config_path.write_text(json.dumps(cfg))
        cls.stack.enter_context(patch.object(api,'DATA',cls.root))
        cls.stack.enter_context(patch.object(api,'CONFIG_PATH',config_path))
        cls.stack.enter_context(patch('backend.worker.slots',side_effect=AssertionError('Video tests must not enumerate cameras')))
        cls.stack.enter_context(patch('backend.worker.cv2.VideoCapture',side_effect=AssertionError('Video tests must not open cameras')))
        cls.client=cls.stack.enter_context(TestClient(api.app))
        for _ in range(240):
            status=cls.client.get('/api/status').json()
            if status['runtime']['state']!='loading':break
            time.sleep(.25)
        if status['runtime']['state']!='ready':raise RuntimeError(status)
        cls.video=cls.root/'sample.mp4'
        image=cv2.imread(str(ROOT/'samples/img2.jpeg'));h,w=image.shape[:2]
        cls.encode(cls.video,cv2.resize(image,(w-w%2,h-h%2)))
        # An actual far frame is used only to test rejected detection handling.
        original=Path('C:/Users/Admin/Downloads/1794199254418_427763108401913506_1169167745021074804.mp4')
        with av.open(str(original)) as container:
            frame=next(f for f in container.decode(video=0) if float(f.time)>=36)
            cls.far=cls.root/'far.mp4';cls.encode(cls.far,frame.to_ndarray(format='bgr24'),8)

    @staticmethod
    def encode(path,image,count=16):
        h,w=image.shape[:2]
        with av.open(str(path),'w') as out:
            stream=out.add_stream('libx264',rate=8);stream.width=w;stream.height=h;stream.pix_fmt='yuv420p'
            stream.options={'crf':'18'}
            for _ in range(count):
                for packet in stream.encode(av.VideoFrame.from_ndarray(image,format='bgr24')):out.mux(packet)
            for packet in stream.encode():out.mux(packet)

    def upload(self,path,name=None):
        with path.open('rb') as f:return self.client.post('/api/jobs/video',files={'file':(name or path.name,f,'video/mp4')})

    def complete(self,path):
        with self.client.websocket_connect('/ws') as ws:
            response=self.upload(path);self.assertEqual(response.status_code,200,response.text)
            jid=response.json()['id'];messages=[]
            for _ in range(100):
                message=ws.receive_json();messages.append(message)
                if message['type']=='confirmed':
                    events=self.client.get('/api/events',params={'job_id':jid}).json()
                    self.assertIn(message['event']['id'],{e['id'] for e in events})
                if message['type']=='finished' and message['job']['id']==jid:break
        api.app.state.worker.wait(10)
        job=next(j for j in self.client.get('/api/jobs').json() if j['id']==jid)
        self.assertEqual(job['status'],'completed',job)
        return job,messages

    def test_video_lifecycle_cpu_and_cuda(self):
        c=self.client
        r=c.post('/api/people',json={'name':'Upstream software fixture','notes':'Functional test only'})
        self.assertEqual(r.status_code,200,r.text);pid=r.json()['id']
        with (ROOT/'samples/img2.jpeg').open('rb') as f:
            response=c.post(f'/api/people/{pid}/photos',files={'file':('sample.jpeg',f,'image/jpeg')})
        self.assertEqual(response.status_code,200,response.text);photo=response.json()['id']
        stored=api.app.state.store.photo(photo);vector=np.asarray(json.loads(stored['embedding']))
        self.assertEqual(vector.shape,(512,));self.assertTrue(np.isfinite(vector).all())
        self.assertAlmostEqual(float(np.linalg.norm(vector)),1,places=5)
        self.assertEqual(stored['checkpoint'],api.app.state.engine.face.checkpoint_hash)
        self.assertEqual(c.get('/api/photos/'+photo).status_code,200)
        # Reopening the storage layer retains the exact saved embedding.
        self.assertEqual(Store(self.root).photo(photo)['embedding'],stored['embedding'])
        self.assertFalse(c.get('/api/people').json()[0]['watched'])
        self.assertEqual(api.app.state.store.templates(stored['checkpoint']),[])
        before_inputs=set((self.root/'videos').iterdir())
        self.assertEqual(self.upload(self.video).status_code,422)
        self.assertEqual(set((self.root/'videos').iterdir()),before_inputs)
        for watched in (True,False,True):
            self.assertEqual(c.put('/api/people/'+pid,json={'name':'Upstream software fixture','notes':'','watched':watched}).status_code,200)
            self.assertEqual(bool(api.app.state.store.templates(stored['checkpoint'])),watched)
        self.assertEqual(c.post(f'/api/people/{pid}/photos',files={'file':('bad.jpg',b'not an image','image/jpeg')}).status_code,422)
        self.assertEqual(len(c.get('/api/people').json()[0]['photos']),1)
        results=[]
        modes=['cpu']+(['cuda'] if torch.cuda.is_available() else [])
        for mode in modes:
            with self.subTest(mode=mode):
                cfg=c.get('/api/config').json();cfg['mode']=mode
                self.assertEqual(c.put('/api/config',json=cfg).status_code,200)
                self.assertEqual(c.get('/api/status').json()['runtime']['selected'],mode)
                job,messages=self.complete(self.video)
                self.assertEqual(job['decoded_frames'],16)
                self.assertEqual(job['processed_frames'],16)
                self.assertEqual(job['runtime']['selected'],mode)
                self.assertTrue(job['diagnostics']['faces_detected'])
                self.assertTrue(job['diagnostics']['faces_usable'])
                self.assertEqual(hashlib.sha256((self.root/job['input']).read_bytes()).digest(),hashlib.sha256(self.video.read_bytes()).digest())
                events=c.get('/api/events',params={'job_id':job['id']}).json()
                self.assertEqual(len(events),1,{'events':events,'diagnostics':job['diagnostics']})
                self.assertEqual(sum(m['type']=='confirmed' for m in messages),1)
                self.assertFalse(any(m['type'] in ('lost','face_lost','tracking_error') for m in messages))
                self.assertEqual(c.get('/api/events/'+events[0]['id']+'/image').status_code,200)
                playback=c.get('/api/jobs/'+job['id']+'/video');self.assertEqual(playback.status_code,200)
                with av.open(io.BytesIO(playback.content)) as output:
                    frames=list(output.decode(video=0));self.assertEqual(len(frames),16)
                    self.assertAlmostEqual(float(frames[-1].time),15/8,delta=.002)
                downloaded=c.get('/api/jobs/'+job['id']+'/video?download=1')
                self.assertEqual(downloaded.content,playback.content)
                self.assertIn('attachment',downloaded.headers['content-disposition'])
                ranged=c.get('/api/jobs/'+job['id']+'/video',headers={'Range':'bytes=0-31'})
                self.assertEqual(ranged.status_code,206);self.assertEqual(ranged.content,playback.content[:32])
                self.assertEqual(Store(self.root).events(job['id'])[0]['id'],events[0]['id'])
                far,messages=self.complete(self.far)
                self.assertGreater(far['diagnostics']['faces_detected'],0)
                self.assertGreater(far['diagnostics']['faces_too_small'],0)
                self.assertEqual(far['diagnostics']['faces_usable'],0)
                self.assertEqual(c.get('/api/events',params={'job_id':far['id']}).json(),[])
                self.assertFalse(any(m['type']=='confirmed' for m in messages))
                results.append({'mode':mode,'frames':job['processed_frames'],'confirmed_events':len(events),
                                'filtered_video_frames':far['processed_frames']})
        # Corrupt video must reach a visible failed job, without a false event.
        bad=c.post('/api/jobs/video',files={'file':('broken.mp4',b'not a video','video/mp4')})
        self.assertEqual(bad.status_code,200,bad.text)
        api.app.state.worker.wait(10)
        failed=next(j for j in c.get('/api/jobs').json() if j['id']==bad.json()['id'])
        self.assertEqual(failed['status'],'failed');self.assertTrue(failed['error'])
        self.assertEqual(c.get('/api/events',params={'job_id':failed['id']}).json(),[])
        self.assertEqual(c.post('/api/jobs/video',files={'file':('empty.mp4',b'','video/mp4')}).status_code,422)
        print('VIDEO_FUNCTIONAL_CHECK',json.dumps(results),flush=True)


if __name__=='__main__':unittest.main()
