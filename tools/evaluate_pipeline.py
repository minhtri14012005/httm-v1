"""Offline inference diagnostics on local video; never modifies application data.

Counts are model outputs, not ground-truth recall/precision or ID-switch metrics.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

import av
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.engine import Engine
from backend.recognition import Evidence, Gallery, associate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--det-size', type=int)
    parser.add_argument('--label', default='current')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((ROOT / 'config.json').read_text())
    if args.det_size:
        cfg['det_size'] = args.det_size
    engine = Engine(cfg)
    connection = sqlite3.connect((ROOT / 'data/app.sqlite3').as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute('''SELECT p.id,p.name,f.embedding FROM people p
        JOIN photos f ON p.id=f.person_id WHERE p.watched=1 AND f.checkpoint=?''', (engine.face.checkpoint_hash,))]
    connection.close()
    gallery = Gallery(rows)
    evidence = Evidence(cfg, gallery, 'offline-evaluation')
    tracker = engine.tracker(cfg, cfg['analysis_fps'])
    counts, sizes = Counter(), Counter()
    original_detect = engine.face.detector.detect
    raw_faces = []
    def detect(image, *a, **kw):
        boxes, points = original_detect(image, *a, **kw)
        raw_faces[:] = boxes.tolist()
        return boxes, points
    engine.face.detector.detect = detect
    original_forward = engine.face.detector.forward
    def forward(image, *a, **kw):
        sizes[f'SCRFD tensor: {image.shape[1]}x{image.shape[0]}'] += 1
        return original_forward(image, *a, **kw)
    engine.face.detector.forward = forward
    handle = engine.person.predictor.model.model.register_forward_pre_hook(
        lambda _, values: sizes.update([f'YOLO tensor: {values[0].shape[-1]}x{values[0].shape[-2]}']))
    durations, records, events, seen = [], [], {}, set()
    with args.video.open('rb') as source:
        video_hash = hashlib.file_digest(source, 'sha256').hexdigest()
    with av.open(str(args.video)) as container:
        stream = container.streams.video[0]
        metadata = {'width': stream.width, 'height': stream.height, 'fps': float(stream.average_rate),
                    'declared_frames': stream.frames, 'duration': float(stream.duration * stream.time_base)}
        origin = None
        last = -1e10
        decoded = 0
        for number, frame in enumerate(container.decode(stream)):
            decoded += 1
            timestamp = float(frame.time) if frame.time is not None else number / float(stream.average_rate)
            if origin is None:
                origin = timestamp
            now = max(0, timestamp - origin)
            if now - last < 1 / cfg['analysis_fps'] - .001:
                continue
            last = now
            image = frame.to_ndarray(format='bgr24')
            width = min(cfg['processing_width'], image.shape[1])
            height = round(image.shape[0] * width / image.shape[1])
            width -= width % 2
            height -= height % 2
            image = cv2.resize(image, (width, height))
            sizes[f'worker frame: {width}x{height}'] += 1
            if engine.face.device.type == 'cuda':
                torch.cuda.synchronize(engine.face.device)
            start = time.perf_counter()
            tracks, faces = engine.infer(image, tracker, cfg)
            if engine.face.device.type == 'cuda':
                torch.cuda.synchronize(engine.face.device)
            durations.append(time.perf_counter() - start)
            linked = associate(faces, tracks)
            updates, alerts = evidence.update(tracks, linked, now)
            for event in updates:
                events[event['id']] = event.copy()
            counts['processed_frames'] += 1
            counts['frames_with_detected_face'] += bool(raw_faces)
            counts['raw_face_detections'] += len(raw_faces)
            usable = [f for f in faces if f.get('quality_status', 'usable') == 'usable']
            counts['usable_faces'] += len(usable)
            counts['frames_with_usable_face'] += bool(usable)
            counts['linked_faces'] += len(linked)
            counts['track_observations'] += len(tracks)
            counts['frames_with_track'] += bool(tracks)
            counts['confirmed_observations'] += sum(t.get('identity_status') == 'face_verified' for t in tracks)
            counts['track_only_observations'] += sum(t.get('identity_status') == 'tracking_unverified' for t in tracks)
            for f in faces:
                counts['face_quality_' + f.get('quality_status', 'usable')] += 1
            seen.update(t['track_id'] for t in tracks)
            ranked = [{'track_id': tid, 'cosine': gallery.rank(f['embedding'])[0][0] if gallery.people else None}
                      for tid, f in linked.items()]
            records.append({'time': now, 'infer_ms': durations[-1] * 1000, 'raw_faces': raw_faces.copy(),
                            'usable_faces': [{'bbox': f['bbox'], 'detection_score': f['detection_score']} for f in usable],
                            'detected_faces': [{k: v for k,v in f.items() if k != 'embedding'} for f in faces],
                            'diagnostics': getattr(engine, 'last_diagnostics', {}),
                            'tracks': tracks, 'matching': ranked})
            if counts['processed_frames'] % 100 == 0:
                print(args.label, counts['processed_frames'], 'frames', flush=True)
        for event in evidence.end(last, 'completed'):
            events[event['id']] = event.copy()
    handle.remove()
    counts['decoded_frames'] = decoded
    counts['skipped_frames'] = decoded - counts['processed_frames']
    counts['unique_track_ids'] = len(seen)
    counts['identity_events'] = len(events)
    counts['confirmed_track_ids'] = len({e['track_id'] for e in events.values()})
    summary = {'label': args.label, 'video_sha256': video_hash, 'video': metadata, 'config': cfg,
               'runtime': engine.status, 'counts': dict(counts), 'input_shapes': dict(sizes),
               'tracker': vars(tracker.args), 'track_buffer_frames': tracker.max_frames_lost,
               'inference_seconds': sum(durations), 'median_infer_ms': float(np.median(durations) * 1000),
               'p95_infer_ms': float(np.percentile(durations, 95) * 1000),
               'ground_truth': None, 'note': 'Output counts only; no accuracy or ID-switch claims.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    (args.output / 'frames.json').write_text(json.dumps(records), encoding='utf8')
    (args.output / 'events.json').write_text(json.dumps(list(events.values()), indent=2), encoding='utf8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
