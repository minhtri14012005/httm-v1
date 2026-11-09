"""Sharpness of aligned video crops, excluding artificial warp padding."""
import cv2
import numpy as np


def sharpness_without_padding(aligned, matrix, source_shape):
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    height, width = aligned.shape[:2]
    source_height, source_width = source_shape[:2]
    inverse = cv2.invertAffineTransform(matrix)
    corners = np.array([[0, 0], [width-1, 0], [0, height-1],
                        [width-1, height-1]], dtype=np.float64)
    footprint = corners @ inverse[:, :2].T + inverse[:, 2]
    if (np.all(footprint[:, 0] >= 0) and np.all(footprint[:, 0] <= source_width-1)
            and np.all(footprint[:, 1] >= 0) and np.all(footprint[:, 1] <= source_height-1)):
        return float(laplacian.var())

    # Match the crop's bilinear warp. Partly padded pixels are invalid too.
    coverage = cv2.warpAffine(np.ones((source_height, source_width), np.float32),
                             matrix, (width, height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    valid = (coverage >= .999).astype(np.uint8)
    # The Laplacian also samples neighboring pixels: omit the padding boundary.
    valid = cv2.erode(valid, np.ones((3, 3), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=1).astype(bool)
    return float(laplacian[valid].var()) if valid.any() else 0.0
