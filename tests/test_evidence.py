import json
from pathlib import Path
import unittest
import numpy as np
from backend.recognition import Gallery, Evidence, associate

CFG=json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.v=np.zeros(512,np.float32);self.v[0]=1
        gallery=Gallery([{'id':'p1','name':'Sample','embedding':json.dumps(self.v.tolist())}])
        self.e=Evidence(CFG,gallery,'job')

    def observe(self,now,tid=1,face=True):
        track={'track_id':tid,'bbox':[0,0,100,200]}
        updates,alerts=self.e.update([track],{tid:{'embedding':self.v}} if face else {},now)
        return track,updates,alerts

    def test_single_high_score_not_confirmed(self):
        t,_,alerts=self.observe(0)
        self.assertIsNone(t['person_id']);self.assertEqual(alerts,[])

    def test_multiframe_confirmation_hidden_face_and_no_repeated_alert(self):
        self.observe(0);self.observe(.3)
        t,_,alerts=self.observe(.6)
        self.assertEqual(t['person_id'],'p1');self.assertEqual(len(alerts),1)
        t,_,alerts=self.observe(.9,face=False)
        self.assertEqual(t['identity_status'],'tracking_unverified');self.assertEqual(alerts,[])

    def test_new_track_never_inherits_identity(self):
        self.observe(0);self.observe(.3);self.observe(.6)
        t,_,_=self.observe(.9,tid=9,face=False)
        self.assertIsNone(t['person_id'])

    def test_expired_same_id_requires_reconfirmation(self):
        self.observe(0);self.observe(.3);self.observe(.6)
        t,updates,_=self.observe(4,face=False)
        self.assertIsNone(t['person_id'])
        closed=[e for e in updates if e['end_reason']=='lost']
        self.assertEqual(closed[0]['last_seen'],.6)
        self.assertEqual(closed[0]['lost_at'],4)

    def test_ambiguous_body_association_rejected(self):
        face={'bbox':[30,20,60,60]}
        self.assertEqual(associate([face],[{'track_id':1,'bbox':[0,0,100,200]},
                                                {'track_id':2,'bbox':[10,0,110,200]}]),{})

    def test_hidden_face_binding_expires_even_with_continuous_body_track(self):
        self.observe(0);self.observe(.3);self.observe(.6)
        original_id=self.e.states[1]['event']['id']
        for now in (.9, 1.5, 2.5, 3.5, 4.5, 5.5):
            track,_,_=self.observe(now,face=False)
            self.assertEqual(track['person_id'],'p1')
        track,updates,_=self.observe(5.7,face=False)
        self.assertIsNone(track['person_id'])
        self.assertEqual(updates[0]['id'],original_id)
        self.assertEqual(updates[0]['verification_status'],'unconfirmed')
        self.assertIsNone(updates[0]['ended_at'])
        self.observe(5.9);self.observe(6.2)
        track,updates,alerts=self.observe(6.5)
        self.assertEqual(track['person_id'],'p1')
        self.assertEqual(track['event_id'],original_id)
        self.assertEqual(alerts,[])

    def test_reappearing_same_id_cannot_reuse_pre_occlusion_binding(self):
        self.observe(0);self.observe(.3);self.observe(.6)
        original_id=self.e.states[1]['event']['id']
        self.e.update([],{},.7)
        track,updates,_=self.observe(.8,face=False)
        self.assertIsNone(track['person_id'])
        self.assertEqual(updates[0]['id'],original_id)
        self.observe(1.0);self.observe(1.3)
        track,_,alerts=self.observe(1.6)
        self.assertEqual(track['person_id'],'p1')
        self.assertEqual(track['event_id'],original_id)
        self.assertEqual(alerts,[])

    def test_visible_unknown_face_revokes_old_name(self):
        self.observe(0);self.observe(.3);self.observe(.6)
        original_id=self.e.states[1]['event']['id']
        unknown=np.zeros(512,np.float32);unknown[1]=1
        track={'track_id':1,'bbox':[0,0,100,200]}
        updates,alerts=self.e.update([track],{1:{'embedding':unknown}},.9)
        self.assertIsNone(track['person_id'])
        self.assertEqual(alerts,[])
        self.assertEqual(updates[0]['id'],original_id)
        self.assertEqual(updates[0]['verification_status'],'unconfirmed')
        self.observe(1.0);self.observe(1.3)
        track,_,alerts=self.observe(1.6)
        self.assertEqual(track['event_id'],original_id)
        self.assertEqual(alerts,[])

    def test_ambiguous_face_revokes_existing_binding(self):
        self.observe(0);self.observe(.3);self.observe(.6)
        tracks=[{'track_id':1,'bbox':[0,0,100,200]}, {'track_id':2,'bbox':[10,0,110,200]}]
        linked=associate([{'bbox':[30,20,60,60],'embedding':self.v}],tracks)
        updates,_=self.e.update(tracks,linked,.9)
        self.assertTrue(all(t['person_id'] is None for t in tracks))
        self.assertEqual(updates[0]['verification_status'],'unconfirmed')
        self.assertIsNone(updates[0]['ended_at'])

    def test_conflicting_identity_ends_event_without_inheriting_name(self):
        other=np.zeros(512,np.float32);other[1]=1
        self.e.gallery=Gallery([
            {'id':'p1','name':'Sample','embedding':json.dumps(self.v.tolist())},
            {'id':'p2','name':'Other','embedding':json.dumps(other.tolist())}])
        self.observe(0);self.observe(.3);self.observe(.6)
        old_id=self.e.states[1]['event']['id']
        track={'track_id':1,'bbox':[0,0,100,200]}
        updates,alerts=self.e.update([track],{1:{'embedding':other}},.9)
        self.assertIsNone(track['person_id'])
        self.assertEqual(alerts,[])
        self.assertTrue(any(e['id']==old_id and e['end_reason']=='conflicting_face' for e in updates))
        for now in (1.2,1.5):
            track={'track_id':1,'bbox':[0,0,100,200]}
            updates,alerts=self.e.update([track],{1:{'embedding':other}},now)
        self.assertEqual(track['person_id'],'p2')
        self.assertNotEqual(track['event_id'],old_id)
        self.assertEqual(len(alerts),1)

    def test_duplicate_identity_proposals_do_not_label_two_bodies(self):
        self.observe(0);self.observe(.3);self.observe(.6)
        tracks=[{'track_id':1}, {'track_id':2}]
        _,alerts=self.e.update(tracks,{tid:{'embedding':self.v} for tid in (1,2)},.9)
        self.assertTrue(all(t['person_id'] is None for t in tracks))
        self.assertEqual(alerts,[])


if __name__=='__main__':unittest.main()
