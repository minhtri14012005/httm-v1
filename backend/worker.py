from fractions import Fraction
import json
from pathlib import Path
import threading
import time
import traceback
import uuid

import av
import cv2
import numpy as np
import torch

from .cameras import slots
from .recognition import Gallery, Evidence, associate


class Worker:
    @staticmethod
    def sampling_fps(cfg, kind):
        return cfg.get(f'{kind}_analysis_fps') or cfg['analysis_fps']

    @staticmethod
    def sample_due(now, last_processed, fps, kind):
        tolerance = .001 if kind == 'video' else 0
        return now - last_processed >= 1 / fps - tolerance

    def __init__(self, store, config, engine, emit):
        self.store, self.config, self.engine, self.emit = store, config, engine, emit
        self.lock = threading.RLock()
        self.current = None
        self.cancel = threading.Event()
        self.thread = None
        self.latest = None
        for old in store.jobs():
            if old['status'] in ('running','stopping'):
                old.update(status='interrupted', error='Backend đã dừng trong phiên trước')
                store.put_job(old)

    def active(self):
        return self.current and self.current['status'] in ('running','stopping')

    def start(self, kind, source, original_name, config):
        with self.lock:
            if self.active() or (self.thread and self.thread.is_alive()):
                raise ValueError('Đang có một phiên xử lý. Dừng phiên đó trước khi bắt đầu/chuyển nguồn.')
            if not self.store.templates(self.engine.face.checkpoint_hash):
                raise ValueError('Watchlist chưa có ảnh đăng ký tương thích')
            jid = uuid.uuid4().hex
            job = {'id':jid,'kind':kind,'name':original_name,'status':'running','progress':0,
                   'created':time.time(),'processed_frames':0,'video_time':0,'duration':None,
                   'config':config.copy(),'error':None,'output':None,'runtime':self.engine.status.copy()}
            if kind == 'video':
                job['input'] = Path(source).resolve().relative_to(self.store.root.resolve()).as_posix()
            self.cancel.clear(); self.latest = None; self.current = job
            self.store.put_job(job)
            self.thread = threading.Thread(target=self.run,args=(job,source),daemon=True)
            self.thread.start()
            return job.copy()

    def stop(self):
        with self.lock:
            if self.active():
                self.current['status']='stopping'
                self.cancel.set()
                self.store.put_job(self.current)

    def wait(self, seconds=10):
        if self.thread:
            self.thread.join(seconds)
            if self.thread.is_alive():
                raise ValueError('Nguồn cũ chưa dừng; chưa mở camera mới để tránh trùng thiết bị')

    def run(self, job, source):
        cfg = job['config']; now = 0.; cap = container = output = None; stream_out = None
        inference_cfg = dict(cfg, face_roi_max_tracks=(
            cfg.get('video_face_roi_max_tracks', 0) if job['kind']=='video' else 0))
        if job['kind']=='video':
            inference_cfg['exclude_alignment_padding'] = True
        sampling_fps = self.sampling_fps(cfg, job['kind'])
        job['sampling_fps'] = sampling_fps
        evidence = None
        try:
            templates = Gallery(self.store.templates(self.engine.face.checkpoint_hash))
            evidence = Evidence(cfg,templates,job['id'],reset_ambiguous_pending=job['kind']=='video')
            tracker = self.engine.tracker(cfg,sampling_fps)
            if job['kind']=='video':
                container=av.open(source); stream=container.streams.video[0]
                job['duration']=float(stream.duration*stream.time_base) if stream.duration else (container.duration/av.time_base if container.duration else None)
                frames = container.decode(stream)
                job['source_video'] = {'width': stream.width, 'height': stream.height,
                                       'fps': float(stream.average_rate or 0), 'frames': stream.frames}
            else:
                entry = next(x for x in slots(cfg) if x['key']==source)
                if not entry['available']:
                    raise ValueError(entry['error'])
                cap=cv2.VideoCapture(entry['index'],cv2.CAP_DSHOW)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,cfg['camera_width']);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,cfg['camera_height'])
                if not cap.isOpened():
                    raise ValueError('Không mở được camera; kiểm tra thiết bị bận/quyền Windows/Camo Studio')
                frames=self.camera_frames(cap)
            started=time.monotonic(); job['started_wall']=time.time()
            last_processed=-1e10; last_emit=0; frame_number=0; time_origin=None
            while not self.cancel.is_set():
                try:
                    frame=next(frames)
                except StopIteration:
                    break
                frame_number+=1
                if job['kind']=='video':
                    timestamp=float(frame.time) if frame.time is not None else (frame_number-1)/float(stream.average_rate or 25)
                    if time_origin is None:time_origin=timestamp
                    now=max(0,timestamp-time_origin)
                    if not self.sample_due(now,last_processed,sampling_fps,'video'):
                        continue
                    bgr=frame.to_ndarray(format='bgr24')
                else:
                    now=time.monotonic()-started; bgr=frame
                    if not self.sample_due(now,last_processed,sampling_fps,'live'):
                        continue
                last_processed=now
                width=min(cfg['processing_width'],bgr.shape[1]);height=round(bgr.shape[0]*width/bgr.shape[1])
                width-=width%2;height-=height%2
                bgr=cv2.resize(bgr,(width,height))
                try:
                    tracks,faces=self.engine.infer(bgr,tracker,inference_cfg)
                except (torch.cuda.OutOfMemoryError,RuntimeError) as exc:
                    message=str(exc).lower()
                    if not any(x in message for x in ('cuda','cudnn','out of memory')):
                        raise
                    self.engine.fallback_after_oom(exc)
                    for event in evidence.end(now,'runtime_reset'):
                        self.store.put_event(event)
                    tracker=self.engine.tracker(cfg,sampling_fps)
                    evidence=Evidence(cfg,templates,job['id'],reset_ambiguous_pending=job['kind']=='video')
                    tracks,faces=self.engine.infer(bgr,tracker,inference_cfg)
                    job['runtime']=self.engine.status.copy()
                updates,alerts=evidence.update(tracks,associate(faces,tracks),now)
                job['diagnostics'] = dict(self.engine.last_diagnostics,
                    decoded_frames=frame_number, skipped_frames=frame_number-job['processed_frames']-1,
                    faces_linked=sum(f.get('association_status')=='linked' for f in faces),
                    faces_verified=sum(f.get('matching_status')=='verified' for f in faces))
                marked=self.draw(bgr,tracks,faces,face_thickness=3 if job['kind']=='video' else 1)
                alert_ids={a['id'] for a in alerts}
                for event in updates:
                    if event['id'] in alert_ids:
                        rel=f"evidence/{event['id']}.jpg"
                        target=self.store.root/rel;target.parent.mkdir(exist_ok=True)
                        cv2.imencode('.jpg',marked)[1].tofile(str(target))
                        event['evidence']=rel
                        state=evidence.states.get(event['track_id'])
                        if state and state['event']:state['event']['evidence']=rel
                    self.store.put_event(event)
                    if event['id'] in alert_ids:
                        self.emit({'type':'confirmed','event':event})
                self.latest=cv2.imencode('.jpg',marked,[cv2.IMWRITE_JPEG_QUALITY,80])[1].tobytes()
                if job['kind']=='video':
                    if output is None:
                        rel=f"videos/{job['id']}_annotated.mp4";path=self.store.root/rel
                        output=av.open(str(path),'w');stream_out=output.add_stream('libx264',rate=Fraction(str(sampling_fps)))
                        stream_out.width=width;stream_out.height=height;stream_out.pix_fmt='yuv420p'
                        stream_out.time_base=Fraction(1,1000);stream_out.options={'crf':'24','preset':'veryfast'}
                        # The encoder otherwise rounds source PTS to 1/rate,
                        # which is incorrect when sampling is not evenly spaced.
                        stream_out.codec_context.time_base=Fraction(1,1000)
                        job['output']=rel
                    encoded=av.VideoFrame.from_ndarray(marked,format='bgr24');encoded.pts=round(now*1000);encoded.time_base=Fraction(1,1000)
                    for packet in stream_out.encode(encoded):output.mux(packet)
                job.update(video_time=now,processed_frames=job['processed_frames']+1,
                           progress=min(99.9,now/job['duration']*100) if job['duration'] else None)
                if time.monotonic()-last_emit>0.4:
                    self.store.put_job(job);self.emit({'type':'progress','job':job.copy(),'tracks':tracks});last_emit=time.monotonic()
            if job['processed_frames']==0 and not self.cancel.is_set():
                raise ValueError('Nguồn không trả về frame đọc được')
            job['decoded_frames']=frame_number
            job['skipped_frames']=frame_number-job['processed_frames']
            job['status']='cancelled' if self.cancel.is_set() else 'completed'
            if job['status']=='completed':job['progress']=100
        except Exception as exc:
            job.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            traceback.print_exc()
        finally:
            if evidence:
                for event in evidence.end(now,job['status']):self.store.put_event(event)
            if output:
                try:
                    for packet in stream_out.encode():output.mux(packet)
                    output.close()
                except Exception as exc:
                    job.update(status='failed',error='Video encode: '+str(exc))
            if container:container.close()
            if cap:cap.release()
            job['finished']=time.time();self.store.put_job(job);self.emit({'type':'finished','job':job.copy()})

    @staticmethod
    def camera_frames(cap):
        failures=0
        while True:
            ok,frame=cap.read()
            if not ok:
                failures+=1
                if failures>=15:raise ValueError('Camera mất kết nối hoặc không trả hình')
                time.sleep(.05);continue
            failures=0
            yield frame

    @staticmethod
    def draw(image,tracks,faces=(),*,face_thickness=1):
        result=image.copy()
        green=(70,200,70)
        red=(50,50,220)
        for face in faces:
            a,b,c,d=[int(v) for v in face['bbox']]
            usable=face.get('quality_status','usable')=='usable'
            color=green if face.get('matching_status')=='verified' and face.get('association_status')=='linked' else red
            cv2.rectangle(result,(a,b),(c,d),color,face_thickness)
            label=face.get('quality_status','usable')
            if usable:
                label=face.get('matching_status') or face.get('association_status') or 'detected'
            cv2.putText(result,'face: '+label,(max(0,a),max(12,d+12)),cv2.FONT_HERSHEY_SIMPLEX,.35,color,1)
        for t in tracks:
            x1,y1,x2,y2=[int(v) for v in t['bbox']]
            color=green if t.get('person_id') and t.get('identity_status') in ('face_verified','tracking_unverified') else red
            cv2.rectangle(result,(x1,y1),(x2,y2),color,2)
            label=f"#{t['track_id']} " + (t.get('name') or 'unconfirmed')
            if t.get('identity_status')=='tracking_unverified':label+=' [track only]'
            cv2.putText(result,label,(x1,max(18,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
        return result
