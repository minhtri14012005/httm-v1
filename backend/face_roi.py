"""A bounded SCRFD retry on the head region of small person tracks."""
from math import isfinite


def _candidate_tracks(box, tracks):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    area = max((x2 - x1) * (y2 - y1), 1)
    matches = []
    for track in tracks:
        a, b, c, d = track['bbox']
        overlap = max(0, min(x2, c) - max(x1, a)) * max(0, min(y2, d) - max(y1, b)) / area
        if a <= cx <= c and b <= cy <= b + .65 * (d - b) and overlap >= .85:
            matches.append(track['track_id'])
    return matches


def _iou(left, right):
    a, b, c, d = left
    e, f, g, h = right
    intersection = max(0, min(c, g) - max(a, e)) * max(0, min(d, h) - max(b, f))
    area_left = max(0, c - a) * max(0, d - b)
    area_right = max(0, g - e) * max(0, h - f)
    return intersection / max(area_left + area_right - intersection, 1)


def recover_small_faces(frame, tracks, faces, detect_crop, max_attempts=3):
    """Retry only where a small body lacks a global face; keep image coordinates.

    The retry contributes detections, never identity evidence by itself. The
    caller applies the unchanged quality and matching gates afterward.
    """
    height, width = frame.shape[:2]
    found = list(faces)
    attempts = skipped = recovered = 0
    for track in sorted(tracks, key=lambda item: item.get('confidence', 0), reverse=True):
        x1, y1, x2, y2 = track['bbox']
        body_width, body_height = x2 - x1, y2 - y1
        if body_width < 32 or not 64 <= body_height <= .45 * height:
            continue
        if any(track['track_id'] in _candidate_tracks(face['bbox'], tracks) for face in found):
            continue
        if attempts >= max_attempts:
            skipped += 1
            continue
        left = max(0, int(x1 - .25 * body_width))
        top = max(0, int(y1 - .08 * body_height))
        right = min(width, int(x2 + .25 * body_width))
        bottom = min(height, int(y1 + .55 * body_height))
        if right - left < 32 or bottom - top < 32:
            continue
        crop = frame[top:bottom, left:right]
        attempts += 1
        candidates = []
        for face in detect_crop(crop):
            a, b, c, d = face['bbox']
            points = face['landmarks']
            if (not all(isfinite(v) for v in (a, b, c, d)) or c <= a or d <= b or
                    len(points) != 5 or any(len(p) != 2 or not all(isfinite(v) for v in p) for p in points)):
                continue
            # A face cut by the ROI edge has unreliable landmarks.
            if a < 0 or b < 0 or c > crop.shape[1] or d > crop.shape[0]:
                continue
            mapped = [a + left, b + top, c + left, d + top]
            if _candidate_tracks(mapped, tracks) != [track['track_id']]:
                continue
            if any(_iou(mapped, other['bbox']) > .3 for other in found):
                continue
            candidates.append(dict(face, bbox=mapped,
                                   landmarks=[[px + left, py + top] for px, py in points],
                                   detection_source='head_roi'))
        # Several faces in one head region do not establish a unique association.
        if len(candidates) == 1:
            found.append(candidates[0])
            recovered += 1
    return found, {'face_roi_attempts': attempts, 'face_roi_skipped_budget': skipped,
                   'face_roi_recovered': recovered}
