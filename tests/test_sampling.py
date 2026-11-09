"""Sampling controls must be independent for uploads and live cameras."""
import unittest

from backend.app import Settings
from backend.worker import Worker


class SamplingTests(unittest.TestCase):
    def test_legacy_rate_and_independent_overrides(self):
        cfg=Settings(analysis_fps=8).model_dump()
        self.assertEqual(cfg['video_face_roi_max_tracks'],0)
        self.assertEqual(Worker.sampling_fps(cfg,'video'),8)
        self.assertEqual(Worker.sampling_fps(cfg,'live'),8)
        cfg=Settings(analysis_fps=8,video_analysis_fps=15,live_analysis_fps=6).model_dump()
        self.assertEqual(Worker.sampling_fps(cfg,'video'),15)
        self.assertEqual(Worker.sampling_fps(cfg,'live'),6)

    def test_frame_selection_uses_source_timestamps(self):
        timestamps=[n/30 for n in range(30)]
        chosen=[];last=-1e10
        for tick in timestamps:
            if Worker.sample_due(tick,last,15,'video'):
                chosen.append(tick);last=tick
        self.assertEqual(len(chosen),15)
        self.assertAlmostEqual(chosen[-1],28/30)
        last=-1e10
        all_frames=[]
        for tick in timestamps:
            if Worker.sample_due(tick,last,30,'video'):
                all_frames.append(tick);last=tick
        self.assertEqual(len(all_frames),30)


if __name__=='__main__':unittest.main()
