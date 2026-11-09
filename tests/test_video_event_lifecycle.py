"""Anonymous state sequences through the real video worker and persistence."""
import copy,json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import av,numpy as np
from backend.store import Store
from backend.worker import Worker

ROOT=Path(__file__).resolve().parents[1]


class VideoEventLifecycleTests(unittest.TestCase):
    def test_filtered_faces_revalidation_gap_and_new_track_do_not_duplicate_events(self):
        cfg=json.loads((ROOT/'config.json').read_text())
        with TemporaryDirectory(prefix='video_states_',dir=ROOT/'results') as temporary:
            root=Path(temporary);(root/'videos').mkdir()
            for folder in ['photos','evidence']:(root/folder).mkdir()
            source=root/'videos'/'sequence.mp4'
            with av.open(str(source),'w') as output:
                stream=output.add_stream('libx264',rate=8);stream.width=120;stream.height=200;stream.pix_fmt='yuv420p'
                for _ in range(72):
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((200,120,3),np.uint8),format='bgr24')):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
            store=Store(root);pid=store.create_person('Anonymous state fixture')
            store.edit_person(pid,'Anonymous state fixture','',True)
            vector=np.zeros(512,np.float32);vector[0]=1
            store.add_photo(pid,'unused.jpg',vector,'synthetic')
            index=-1
            def infer(*args):
                nonlocal index
                index+=1
                track={'track_id':1 if index<64 else 2,'bbox':[10,10,100,190],'confidence':.9}
                if index<8 or 50<=index<64:
                    faces=[{'bbox':[20,30,60,70],'quality_status':'usable','embedding':vector}]
                elif index<16:
                    faces=[{'bbox':[20,30,40,50],'quality_status':'too_small'}]
                elif index<24:
                    faces=[{'bbox':[20,30,60,70],'quality_status':'blurred'}]
                else:faces=[]
                return [track],faces
            engine=SimpleNamespace(face=SimpleNamespace(checkpoint_hash='synthetic'),status={'state':'ready','selected':'cpu'},
                tracker=Mock(return_value=object()),infer=infer,last_diagnostics={})
            messages=[];drawn=[];real_draw=Worker.draw
            def draw(image,tracks,faces,**kwargs):
                marked=real_draw(image,tracks,faces,**kwargs)
                drawn.append({'tracks':copy.deepcopy(tracks),'faces':copy.deepcopy(faces),
                              'body_pixel':marked[10,10].tolist(),
                              'face_pixel':marked[30,20].tolist() if faces else None})
                return marked
            worker=Worker(store,cfg,engine,messages.append)
            with patch.object(Worker,'draw',side_effect=draw),patch('backend.worker.slots',side_effect=AssertionError('No camera access')):
                worker.start('video',str(source),'anonymous-sequence.mp4',cfg);worker.wait(10)
            self.assertEqual(worker.current['status'],'completed',worker.current)
            self.assertEqual(worker.current['processed_frames'],72)
            confirmed=[m for m in messages if m['type']=='confirmed']
            self.assertEqual(len(confirmed),1)
            events=store.events(worker.current['id']);self.assertEqual(len(events),1)
            event_id=events[0]['id'];self.assertEqual(confirmed[0]['event']['id'],event_id)
            self.assertEqual(drawn[4]['tracks'][0]['event_id'],event_id)
            self.assertEqual(drawn[10]['tracks'][0]['identity_status'],'tracking_unverified')
            self.assertEqual(drawn[10]['body_pixel'],[70,200,70])
            self.assertEqual(drawn[10]['face_pixel'],[50,50,220])
            self.assertEqual(drawn[18]['face_pixel'],[50,50,220])
            self.assertEqual(drawn[49]['tracks'][0]['identity_status'],'unconfirmed')
            self.assertIsNone(drawn[49]['tracks'][0]['person_id'])
            self.assertEqual(drawn[49]['body_pixel'],[50,50,220])
            self.assertEqual(drawn[55]['tracks'][0]['event_id'],event_id)
            self.assertEqual(drawn[55]['body_pixel'],[70,200,70])
            self.assertEqual(drawn[55]['face_pixel'],[70,200,70])
            self.assertIsNone(drawn[65]['tracks'][0]['person_id'])
            self.assertEqual(drawn[65]['body_pixel'],[50,50,220])
            self.assertTrue(all(m['type'] in ('confirmed','progress','finished') for m in messages))


if __name__=='__main__':unittest.main()
