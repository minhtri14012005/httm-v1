"""Exercise the real BoT-SORT low-confidence recovery with the engine adapter."""
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from backend.engine import Engine
from ultralytics.engine.results import Boxes

CFG=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())


class TrackingContractTests(unittest.TestCase):
    def setUp(self):
        self.engine=Engine.__new__(Engine)
        self.engine.lock=threading.RLock()
        self.engine.face=SimpleNamespace(device=torch.device('cpu'), det_threshold=.5, det_size=640,
                                         detect=Mock(return_value=[]), embed=Mock())
        self.frame=np.zeros((320,320,3),np.uint8)
        self.confidence=.8
        def predict(frame, **kwargs):
            data=np.array([[40,30,140,280,self.confidence,0]],np.float32)
            data=data[data[:,4]>=kwargs['conf']]
            return [SimpleNamespace(boxes=Boxes(data,frame.shape[:2]))]
        self.engine.person=SimpleNamespace(predict=Mock(side_effect=predict))

    def test_low_confidence_can_recover_but_never_start_a_track(self):
        tracker=self.engine.tracker(CFG,CFG['analysis_fps'])
        first,_=self.engine.infer(self.frame,tracker,CFG)
        self.assertEqual(len(first),1)
        self.confidence=.15
        second,_=self.engine.infer(self.frame,tracker,CFG)
        self.assertEqual(second[0]['track_id'],first[0]['track_id'])
        fresh=self.engine.tracker(CFG,CFG['analysis_fps'])
        unconfirmed,_=self.engine.infer(self.frame,fresh,CFG)
        self.assertEqual(unconfirmed,[])
        self.assertFalse(tracker.args.with_reid)
        self.assertEqual(tracker.max_frames_lost,16)

    def test_small_detection_is_reported_and_never_embedded(self):
        self.engine.face.detect.return_value=[{'bbox':[10,10,25,30], 'detection_score':.9, 'landmarks':[]}]
        _,faces=self.engine.infer(self.frame,self.engine.tracker(CFG,8),CFG)
        self.assertEqual(faces[0]['quality_status'],'too_small')
        self.engine.face.embed.assert_not_called()
        self.assertEqual(self.engine.last_diagnostics['faces_detected'],1)
        self.assertEqual(self.engine.last_diagnostics['faces_usable'],0)

    def test_roi_is_opt_in_and_preserves_the_quality_gate(self):
        frame=np.zeros((1280,720,3),np.uint8)
        self.engine.infer(frame,self.engine.tracker(CFG,8),CFG)
        self.engine.face.detect.assert_called_once()
        self.engine.face.detect.reset_mock()
        self.engine.face.detect.side_effect=[[],[{
            'bbox':[45,25,65,50],'landmarks':[[50,30]]*5,'detection_score':.8}]]
        _,faces=self.engine.infer(frame,self.engine.tracker(CFG,8),dict(CFG,face_roi_max_tracks=3))
        self.assertEqual(self.engine.face.detect.call_count,2)
        self.assertEqual(faces[0]['detection_source'],'head_roi')
        self.assertEqual(faces[0]['quality_status'],'too_small')
        self.engine.face.embed.assert_not_called()
        self.assertEqual(self.engine.last_diagnostics['face_roi_recovered'],1)


if __name__=='__main__':unittest.main()
