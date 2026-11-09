"""ROI retries must preserve coordinates and reject ambiguous or clipped faces."""
import unittest

import numpy as np

from backend.face_roi import recover_small_faces


class FaceRoiTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((1280,720,3),dtype=np.uint8)
        self.track={'track_id':7,'bbox':[180,640,280,920],'confidence':.8}

    @staticmethod
    def face(box):
        return {'bbox':box,'landmarks':[[35,35]]*5,'detection_score':.7}

    def test_recovers_face_and_maps_box_and_landmarks(self):
        calls=[]
        def detect(crop):
            calls.append(crop.shape)
            return [self.face([30,30,50,58])]
        faces,diag=recover_small_faces(self.frame,[self.track],[],detect)
        self.assertEqual(calls,[(177,150,3)])
        self.assertEqual(diag['face_roi_recovered'],1)
        self.assertEqual(faces[0]['bbox'],[185,647,205,675])
        self.assertEqual(faces[0]['landmarks'][0],[190,652])
        self.assertEqual(faces[0]['detection_source'],'head_roi')

    def test_no_retry_when_global_face_already_covers_track(self):
        global_face=self.face([185,647,205,675])
        faces,diag=recover_small_faces(self.frame,[self.track],[global_face],lambda _:self.fail('unexpected retry'))
        self.assertEqual(faces,[global_face])
        self.assertEqual(diag['face_roi_attempts'],0)

    def test_rejects_ambiguous_and_clipped_faces(self):
        overlapping=dict(self.track,track_id=8)
        faces,diag=recover_small_faces(self.frame,[self.track,overlapping],[],
                                       lambda _:[self.face([30,30,50,58])])
        self.assertEqual(faces,[])
        self.assertEqual(diag['face_roi_recovered'],0)
        faces,diag=recover_small_faces(self.frame,[self.track],[],
                                       lambda _:[self.face([-1,30,50,58])])
        self.assertEqual(faces,[])
        self.assertEqual(diag['face_roi_recovered'],0)

    def test_budget_can_disable_extra_inference(self):
        faces,diag=recover_small_faces(self.frame,[self.track],[],lambda _:self.fail('unexpected retry'),max_attempts=0)
        self.assertEqual(faces,[])
        self.assertEqual(diag['face_roi_skipped_budget'],1)

    def test_multiple_faces_in_one_roi_are_not_assumed_to_share_a_body(self):
        faces,diag=recover_small_faces(self.frame,[self.track],[],
            lambda _:[self.face([30,30,50,58]),self.face([70,30,90,58])])
        self.assertEqual(faces,[])
        self.assertEqual(diag['face_roi_recovered'],0)

    def test_invalid_landmarks_are_not_sent_downstream(self):
        invalid=self.face([30,30,50,58]);invalid['landmarks'][0]=[float('nan'),35]
        faces,diag=recover_small_faces(self.frame,[self.track],[],lambda _:[invalid])
        self.assertEqual(faces,[])


if __name__=='__main__':unittest.main()
