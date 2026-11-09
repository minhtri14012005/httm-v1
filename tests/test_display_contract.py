"""Rendering rules shared by live preview and annotated uploads."""
import unittest

import numpy as np

from backend.worker import Worker


class DisplayContractTests(unittest.TestCase):
    def test_two_colors_follow_verified_association(self):
        image=np.zeros((200,220,3),dtype=np.uint8)
        tracks=[
            {'bbox':[10,10,90,180],'track_id':1,'person_id':'p1','name':'A','identity_status':'face_verified'},
            {'bbox':[110,10,190,180],'track_id':2,'person_id':None,'name':None,'identity_status':'unconfirmed'}]
        faces=[
            {'bbox':[20,30,50,60],'quality_status':'usable','matching_status':'verified','association_status':'linked'},
            {'bbox':[120,30,150,60],'quality_status':'usable','matching_status':'candidate','association_status':'linked'}]
        marked=Worker.draw(image,tracks,faces)
        green=[70,200,70];red=[50,50,220]
        self.assertEqual(marked[10,10].tolist(),green)
        self.assertEqual(marked[30,20].tolist(),green)
        self.assertEqual(marked[10,110].tolist(),red)
        self.assertEqual(marked[30,120].tolist(),red)

    def test_tracking_grace_and_missing_face(self):
        image=np.zeros((200,100,3),dtype=np.uint8)
        track={'bbox':[10,10,90,180],'track_id':1,'person_id':'p1','name':'A',
               'identity_status':'tracking_unverified'}
        marked=Worker.draw(image,[track],[])
        self.assertEqual(marked[10,10].tolist(),[70,200,70])
        self.assertEqual(marked[60,50].tolist(),[0,0,0])
        track.update(person_id=None,name=None,identity_status='unconfirmed')
        marked=Worker.draw(image,[track],[])
        self.assertEqual(marked[10,10].tolist(),[50,50,220])


if __name__=='__main__':unittest.main()
