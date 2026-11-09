"""Video-only decode/sampling/output integration with an anonymous empty fixture."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import av
import numpy as np

from backend.store import Store
from backend.worker import Worker
from backend.recognition import Evidence

ROOT=Path(__file__).resolve().parents[1]


class VideoPipelineTests(unittest.TestCase):
    def test_upload_worker_preserves_timestamps_and_defaults_to_no_roi(self):
        cfg=json.loads((ROOT/'config.json').read_text())
        cfg.update(video_analysis_fps=15)
        cfg.pop('video_face_roi_max_tracks',None)
        results=(ROOT/'results').resolve();results.mkdir(exist_ok=True)
        with TemporaryDirectory(prefix='video_fixture_',dir=results) as temporary:
            root=Path(temporary).resolve()
            self.assertEqual(root.parent,results)
            (root/'videos').mkdir()
            source=root/'videos'/'fixture.mp4'
            with av.open(str(source),'w') as output:
                stream=output.add_stream('libx264',rate=30)
                stream.width=320;stream.height=480;stream.pix_fmt='yuv420p'
                for _ in range(16):
                    frame=av.VideoFrame.from_ndarray(np.zeros((480,320,3),np.uint8),format='bgr24')
                    for packet in stream.encode(frame):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
            store=Store(root)
            pid=store.create_person('Synthetic test fixture')
            store.edit_person(pid,'Synthetic test fixture','',True)
            vector=np.zeros(512,np.float32);vector[0]=1
            store.add_photo(pid,'unused.jpg',vector,'synthetic-checkpoint')
            engine=SimpleNamespace(face=SimpleNamespace(checkpoint_hash='synthetic-checkpoint'),
                status={'state':'ready','selected':'cpu'},tracker=Mock(return_value=object()),
                infer=Mock(return_value=([],[])),last_diagnostics={'faces_detected':0})
            emitted=[]
            worker=Worker(store,cfg,engine,emitted.append)
            with patch('backend.worker.Evidence',wraps=Evidence) as evidence_factory, \
                 patch('backend.worker.slots',side_effect=AssertionError('Camera enumeration is outside this test')), \
                 patch('backend.worker.cv2.VideoCapture',side_effect=AssertionError('Camera access is outside this test')):
                worker.start('video',str(source),'fixture.mp4',cfg)
                worker.wait(10)
            job=worker.current
            self.assertEqual(job['status'],'completed',job)
            self.assertEqual((job['decoded_frames'],job['processed_frames'],job['skipped_frames']),(16,8,8))
            self.assertEqual(job['sampling_fps'],15)
            self.assertEqual(engine.tracker.call_args.args[1],15)
            self.assertTrue(evidence_factory.call_args.kwargs['reset_ambiguous_pending'])
            self.assertTrue(all(call.args[2]['face_roi_max_tracks']==0 for call in engine.infer.call_args_list))
            self.assertTrue(all(call.args[2]['exclude_alignment_padding'] for call in engine.infer.call_args_list))
            self.assertNotIn('face_roi_max_tracks',cfg)
            self.assertNotIn('exclude_alignment_padding',cfg)
            self.assertEqual(store.events(job['id']),[])
            self.assertFalse(any(message['type']=='confirmed' for message in emitted))
            with av.open(str(root/job['output'])) as container:
                timestamps=[float(frame.time) for frame in container.decode(video=0)]
            self.assertEqual(len(timestamps),8)
            for index,timestamp in enumerate(timestamps):
                self.assertAlmostEqual(timestamp,index*2/30,delta=.0011)

    def test_eight_fps_sampling_preserves_nonuniform_source_timestamps(self):
        cfg=json.loads((ROOT/'config.json').read_text());cfg['video_analysis_fps']=8
        with TemporaryDirectory(prefix='video_pts_',dir=ROOT/'results') as temporary:
            root=Path(temporary);(root/'videos').mkdir()
            source=root/'videos'/'fixture.mp4'
            with av.open(str(source),'w') as output:
                stream=output.add_stream('libx264',rate=30)
                stream.width=64;stream.height=64;stream.pix_fmt='yuv420p'
                for _ in range(16):
                    frame=av.VideoFrame.from_ndarray(np.zeros((64,64,3),np.uint8),format='bgr24')
                    for packet in stream.encode(frame):output.mux(packet)
                for packet in stream.encode():output.mux(packet)
            store=Store(root);pid=store.create_person('Anonymous timestamp fixture')
            store.edit_person(pid,'Anonymous timestamp fixture','',True)
            vector=np.zeros(512,np.float32);vector[0]=1
            store.add_photo(pid,'unused.jpg',vector,'synthetic')
            engine=SimpleNamespace(face=SimpleNamespace(checkpoint_hash='synthetic'),
                status={'state':'ready','selected':'cpu'},tracker=Mock(return_value=object()),
                infer=Mock(return_value=([],[])),last_diagnostics={})
            worker=Worker(store,cfg,engine,lambda message:None)
            with patch('backend.worker.slots',side_effect=AssertionError('No camera access')):
                worker.start('video',str(source),'fixture.mp4',cfg);worker.wait(10)
            self.assertEqual(worker.current['status'],'completed',worker.current)
            with av.open(str(root/worker.current['output'])) as output:
                timestamps=[float(frame.time) for frame in output.decode(video=0)]
            self.assertEqual(len(timestamps),4)
            for actual,expected in zip(timestamps,[0,4/30,8/30,12/30]):
                self.assertAlmostEqual(actual,expected,delta=.0011)


if __name__=='__main__':unittest.main()
