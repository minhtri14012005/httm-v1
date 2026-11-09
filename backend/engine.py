import gc
import os
from pathlib import Path
import threading
import hashlib

from baseline.pipeline import Pipeline, ROOT, align_face
from download_models import EXPECTED
import cv2
import numpy as np
import torch

# Ultralytics state and cache stay in this local workspace. Disable external sync.
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / 'data' / 'ultralytics')
os.environ['YOLO_AUTOINSTALL'] = 'false'
os.environ['YOLO_OFFLINE'] = 'true'
Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
from ultralytics import YOLO, settings
from ultralytics.trackers.bot_sort import BOTSORT
from ultralytics.utils import YAML, IterableSimpleNamespace
from .face_roi import recover_small_faces
from .face_quality import sharpness_without_padding


class Engine:
    def __init__(self, cfg):
        self.lock = threading.RLock()
        self.cfg = cfg
        self.face = self.person = None
        self.status = {'state': 'loading'}
        settings.update({'sync': False})
        with (ROOT / 'models/yolo11n.pt').open('rb') as f:
            if hashlib.file_digest(f, 'sha256').hexdigest()!=EXPECTED['yolo11n.pt']:
                raise ValueError('YOLO checkpoint hash mismatch')
        self.load(cfg['mode'])

    def load(self, mode):
        self.face = self.person = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
        face = Pipeline(mode, device=self.cfg.get('device') if mode != 'cpu' else None,
                        det_size=self.cfg['det_size'], det_threshold=self.cfg.get('face_detection_threshold', .5))
        try:
            person = YOLO(str(ROOT / 'models/yolo11n.pt'), task='detect')
            result = person.predict(np.zeros((320,320,3),np.uint8), imgsz=320, classes=[0],
                                    device=str(face.device), verbose=False, save=False)[0]
            actual = str(person.predictor.model.device)
            if torch.device(actual).type != face.device.type:
                raise RuntimeError('YOLO device mismatch')
        except Exception as exc:
            if mode == 'auto' and face.device.type == 'cuda':
                del face
                self.load('cpu')
                self.status['fallback_reason'] = 'Person detector CUDA failed: ' + str(exc)
                return
            raise
        self.face, self.person = face, person
        self.status = dict(face.status, state='ready', person=actual, person_model='YOLO11n', tracker='BoT-SORT (ReID off)')

    def fallback_after_oom(self, exc):
        if self.cfg['mode'] != 'auto' or self.face.device.type != 'cuda':
            raise exc
        self.load('cpu')
        self.status['fallback_reason'] = 'GPU inference error; whole pipeline restarted on CPU: ' + str(exc)

    def enroll(self, bgr):
        with self.lock:
            faces = self.face.analyze(bgr)
            if len(faces) != 1:
                raise ValueError(f'Ảnh đăng ký cần đúng một khuôn mặt; tìm thấy {len(faces)}')
            f = faces[0]
            x1,y1,x2,y2 = f['bbox']
            aligned,_ = align_face(bgr, f['landmarks'])
            sharpness = cv2.Laplacian(cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
            if min(x2-x1,y2-y1)<80 or sharpness<20:
                raise ValueError('Khuôn mặt đăng ký quá nhỏ hoặc mờ; chọn ảnh rõ hơn')
            return f['embedding'], self.face.checkpoint_hash

    def tracker(self, cfg, fps):
        import ultralytics
        path = Path(ultralytics.__file__).parent / 'cfg/trackers/botsort.yaml'
        args = YAML.load(path)
        args.update(with_reid=False, gmc_method='sparseOptFlow', track_buffer=max(1,round(cfg['lost_seconds']*fps)))
        # User confidence controls starting/high-score matches, not low-score recovery.
        args['track_high_thresh'] = max(args['track_high_thresh'], cfg['person_confidence'])
        args['new_track_thresh'] = max(args['new_track_thresh'], cfg['person_confidence'])
        return BOTSORT(IterableSimpleNamespace(**args))

    def infer(self, frame, tracker, cfg):
        with self.lock:
            result = self.person.predict(frame, classes=[0], conf=min(cfg['person_confidence'], tracker.args.track_low_thresh),
                                         imgsz=cfg['person_size'], device=str(self.face.device),
                                         verbose=False, save=False)[0]
            raw = tracker.update(result.boxes.cpu().numpy(), frame)
            tracks = [{'bbox': [float(v) for v in row[:4]], 'track_id': int(row[4]), 'confidence': float(row[5])}
                      for row in raw]
            faces = self.face.detect(frame)
            roi_diagnostics = {'face_roi_attempts': 0, 'face_roi_skipped_budget': 0,
                               'face_roi_recovered': 0}
            if cfg.get('face_roi_max_tracks', 0) > 0:
                faces, roi_diagnostics = recover_small_faces(
                    frame, tracks, faces, self.face.detect,
                    max_attempts=cfg['face_roi_max_tracks'])
            for f in faces:
                x1,y1,x2,y2 = f['bbox']
                if min(x2-x1,y2-y1) < cfg['min_face_pixels']:
                    f['quality_status'] = 'too_small'
                    continue
                aligned,matrix = align_face(frame,f['landmarks'])
                if cfg.get('exclude_alignment_padding', False):
                    sharp = sharpness_without_padding(aligned, matrix, frame.shape)
                else:
                    sharp = cv2.Laplacian(cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
                f['sharpness'] = float(sharp)
                if sharp < cfg['min_face_sharpness']:
                    f['quality_status'] = 'blurred'
                    continue
                embedding,norm = self.face.embed(aligned)
                f.update(quality_status='usable', embedding=embedding,
                         embedding_norm_before_l2=norm, alignment_matrix=matrix.tolist())
            self.last_diagnostics = {
                'frame_width': frame.shape[1], 'frame_height': frame.shape[0],
                'face_detection_threshold': self.face.det_threshold, 'face_detector_size': self.face.det_size,
                'faces_detected': len(faces), 'faces_usable': sum(f['quality_status']=='usable' for f in faces),
                'faces_too_small': sum(f['quality_status']=='too_small' for f in faces),
                'faces_blurred': sum(f['quality_status']=='blurred' for f in faces),
                **roi_diagnostics,
                'person_detections': len(result.boxes), 'tracks': len(tracks),
                'person_detector_floor': min(cfg['person_confidence'], tracker.args.track_low_thresh),
                'track_buffer_frames': tracker.max_frames_lost, 'person_reid': False}
            return tracks, faces
