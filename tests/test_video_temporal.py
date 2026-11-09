"""Do not join pending evidence across an ambiguous person association."""
import json
from pathlib import Path
import unittest
import numpy as np
from backend.recognition import Evidence,Gallery,associate

CFG=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())


class VideoTemporalTests(unittest.TestCase):
    def setUp(self):
        self.vector=np.zeros(512,np.float32);self.vector[0]=1
        self.gallery=Gallery([{'id':'anonymous','name':'Anonymous fixture',
                               'embedding':json.dumps(self.vector.tolist())}])
        self.evidence=Evidence(CFG,self.gallery,'synthetic-video',reset_ambiguous_pending=True)

    def observe(self,now,ambiguous=False):
        tracks=[{'track_id':1,'bbox':[0,0,100,200]}]
        if ambiguous:tracks.append({'track_id':2,'bbox':[10,0,110,200]})
        faces=[{'bbox':[30,20,70,60],'quality_status':'usable','embedding':self.vector}]
        updates,alerts=self.evidence.update(tracks,associate(faces,tracks),now)
        return tracks[0],updates,alerts

    def test_one_new_face_after_ambiguous_link_cannot_confirm(self):
        self.observe(0);self.observe(.3)
        track,_,alerts=self.observe(.4,ambiguous=True)
        self.assertTrue(track['association_ambiguous'])
        self.assertEqual(len(self.evidence.states[1]['votes']),0)
        self.assertEqual(alerts,[])
        track,_,alerts=self.observe(.6)
        self.assertIsNone(track['person_id']);self.assertEqual(alerts,[])

    def test_fresh_evidence_can_confirm_after_reset(self):
        self.observe(0);self.observe(.3);self.observe(.4,ambiguous=True)
        self.observe(.6);self.observe(.9)
        track,_,alerts=self.observe(1.2)
        self.assertEqual(track['person_id'],'anonymous');self.assertEqual(len(alerts),1)

    def test_existing_event_remains_stable_after_ambiguity(self):
        self.observe(0);self.observe(.3)
        track,_,_=self.observe(.6);original=track['event_id']
        track,updates,alerts=self.observe(.7,ambiguous=True)
        self.assertIsNone(track['person_id']);self.assertEqual(alerts,[])
        self.assertEqual(updates[0]['id'],original)
        self.observe(.8);self.observe(1.1)
        track,_,alerts=self.observe(1.4)
        self.assertEqual(track['event_id'],original);self.assertEqual(alerts,[])


if __name__=='__main__':unittest.main()
