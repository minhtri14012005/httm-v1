"""Numerical contracts that catch color/alignment mistakes before using models."""
import unittest
import numpy as np
from baseline.pipeline import REFERENCE, align_face, normalized_input


class PreprocessingContract(unittest.TestCase):
    def test_similarity_recovers_known_rotation_scale_translation(self):
        theta = 0.4
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        source = (REFERENCE @ rotation.T) * 1.7 + [40, -12]
        _, matrix = align_face(np.zeros((256, 256, 3), np.uint8), source)
        recovered = source @ matrix[:, :2].T + matrix[:, 2]
        np.testing.assert_allclose(recovered, REFERENCE, atol=1e-7)
        self.assertGreater(np.linalg.det(matrix[:, :2]), 0)

    def test_rgb_channels_and_normalization(self):
        bgr = np.full((112, 112, 3), [0, 127, 255], np.uint8)
        x = normalized_input(bgr)
        self.assertEqual(tuple(x.shape), (1, 3, 112, 112))
        np.testing.assert_allclose(x.numpy()[0, :, 0, 0], [1., 127 / 127.5 - 1, -1.], atol=1e-7)

    def test_invalid_landmarks_rejected(self):
        with self.assertRaises(ValueError):
            align_face(np.zeros((112, 112, 3), np.uint8), np.zeros((5, 2)))


if __name__ == '__main__':
    unittest.main()
