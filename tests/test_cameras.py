import unittest
from unittest.mock import patch
from backend.cameras import slots
from backend.worker import Worker


class CameraLifecycleTests(unittest.TestCase):
    def test_named_slots_resolve_current_order_without_fixed_indexes(self):
        cfg={'camera_names':{'laptop':None,'camo':None}}
        with patch('backend.cameras.enumerate_devices',return_value=['OBS Virtual Camera','Camo','ACER HD User Facing']):
            devices=slots(cfg)
        self.assertEqual(devices[0]['index'],2)
        self.assertEqual(devices[1]['index'],1)
        self.assertEqual(len(devices),2)

    def test_disconnect_is_reported(self):
        class Disconnected:
            def read(self):return False,None
        with patch('backend.worker.time.sleep'):
            with self.assertRaisesRegex(ValueError,'Camera'):
                next(Worker.camera_frames(Disconnected()))


if __name__=='__main__':unittest.main()
