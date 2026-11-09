"""Regression for artificial black warp edges passing the sharpness gate."""
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
import torch

from backend.engine import Engine
from backend.face_quality import sharpness_without_padding
from baseline.pipeline import REFERENCE
from ultralytics.engine.results import Boxes


class FaceQualityTests(unittest.TestCase):
    def test_black_padding_does_not_make_a_uniform_crop_sharp(self):
        source = np.full((128, 128, 3), 128, np.uint8)
        matrix = np.array([[1., 0., 0.], [0., 1., 56.]])
        aligned = cv2.warpAffine(source, matrix, (112, 112))
        legacy = cv2.Laplacian(cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        self.assertGreater(legacy, 15)
        self.assertEqual(sharpness_without_padding(aligned, matrix, source.shape), 0)

    def test_crop_without_padding_keeps_the_exact_score(self):
        image = np.random.default_rng(42).integers(0, 256, (112, 112, 3), dtype=np.uint8)
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
        expected = cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        self.assertEqual(sharpness_without_padding(image, matrix, image.shape), expected)

    def test_no_valid_source_pixels_scores_zero(self):
        self.assertEqual(sharpness_without_padding(np.zeros((112,112,3),np.uint8),
                         np.array([[1.,0.,1000.],[0.,1.,1000.]]), (128,128,3)), 0)

    def test_video_gate_blocks_padding_before_embedding(self):
        frame = np.full((128, 128, 3), 128, np.uint8)
        engine = Engine.__new__(Engine)
        engine.lock = threading.RLock()
        engine.face = SimpleNamespace(device=torch.device('cpu'), det_threshold=.5, det_size=640,
            detect=Mock(side_effect=lambda _: [{'bbox':[10,0,70,60], 'detection_score':.9,
                                                'landmarks':(REFERENCE-[0,56]).tolist()}]),
            embed=Mock(return_value=(np.zeros(512,np.float32),1.)))
        engine.person = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(
            boxes=Boxes(np.empty((0,6),np.float32),frame.shape[:2]))]))
        tracker = SimpleNamespace(update=lambda *a: [], args=SimpleNamespace(track_low_thresh=.1),
                                  max_frames_lost=16)
        cfg = dict(person_confidence=.25,person_size=640,min_face_pixels=32,min_face_sharpness=15)
        _, legacy = engine.infer(frame,tracker,cfg)
        self.assertEqual(legacy[0]['quality_status'],'usable')
        engine.face.embed.reset_mock()
        _, corrected = engine.infer(frame,tracker,dict(cfg,exclude_alignment_padding=True))
        self.assertEqual(corrected[0]['quality_status'],'blurred')
        engine.face.embed.assert_not_called()


if __name__=='__main__':unittest.main()
