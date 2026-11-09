import asyncio
from contextlib import asynccontextmanager
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

import cv2
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from baseline.pipeline import ROOT, read_image
from .store import Store
from .engine import Engine
from .worker import Worker
from .cameras import slots

DATA = Path(os.environ.get('APP_DATA_DIR', ROOT / 'data')).resolve()
CONFIG_PATH = ROOT / 'config.json'
ALLOWED_ORIGINS = ['http://127.0.0.1:8000','http://localhost:8000','http://127.0.0.1:5173','http://localhost:5173']


class Settings(BaseModel):
    mode: str = 'auto'
    device: str | None = None
    det_size: int = Field(640, ge=320, le=1280, multiple_of=32)
    face_detection_threshold: float = Field(.5, gt=0, lt=1)
    person_size: int = Field(640, ge=320, le=1280, multiple_of=32)
    processing_width: int = Field(1280, ge=320, le=1920)
    analysis_fps: float = Field(8, ge=1, le=30)
    video_analysis_fps: float | None = Field(None, ge=1, le=30)
    live_analysis_fps: float | None = Field(None, ge=1, le=30)
    person_confidence: float = Field(.25, gt=0, lt=1)
    identity_threshold: float = Field(.55, gt=0, lt=1)
    identity_margin: float = Field(.08, ge=0, le=1)
    confirmation_hits: int = Field(3, ge=2, le=30)
    confirmation_seconds: float = Field(.5, ge=.1, le=20)
    evidence_window_seconds: float = Field(3, ge=.5, le=30)
    lost_seconds: float = Field(2, ge=.25, le=30)
    min_face_pixels: int = Field(32, ge=12, le=256)
    min_face_sharpness: float = Field(15, ge=0, le=1000)
    video_face_roi_max_tracks: int = Field(0, ge=0, le=20)
    revalidate_seconds: float = Field(5, ge=.5, le=60)
    camera_names: dict[str,str | None] = {'laptop':None,'camo':None}
    camera_width: int = Field(1280, ge=320, le=1920)
    camera_height: int = Field(720, ge=240, le=1080)
    max_upload_mb: int = Field(2048, ge=10, le=10240)


class PersonInput(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    notes: str = Field('', max_length=2000)
    watched: bool = False


class CameraInput(BaseModel):
    source: str


class Hub:
    def __init__(self,loop):
        self.loop,self.clients=loop,set()

    def emit(self, payload):
        self.loop.call_soon_threadsafe(self.deliver, payload)

    def deliver(self,payload):
        for queue in tuple(self.clients):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(payload)


@asynccontextmanager
async def lifespan(app):
    DATA.mkdir(parents=True,exist_ok=True)
    for folder in ('photos','videos','evidence'):(DATA/folder).mkdir(exist_ok=True)
    app.state.store=Store(DATA)
    app.state.config=Settings(**json.loads(CONFIG_PATH.read_text())).model_dump()
    app.state.hub=Hub(asyncio.get_running_loop())
    app.state.runtime={'state':'loading'}
    app.state.engine=None;app.state.worker=None
    async def initialize():
        try:
            engine=await run_in_threadpool(Engine,app.state.config)
            app.state.engine=engine
            app.state.worker=Worker(app.state.store,app.state.config,engine,app.state.hub.emit)
            app.state.runtime=engine.status
        except Exception as exc:
            app.state.runtime={'state':'failed','error':f'{type(exc).__name__}: {exc}'}
        app.state.hub.emit({'type':'runtime','runtime':app.state.runtime})
    task=asyncio.create_task(initialize())
    yield
    await task
    if app.state.worker:
        app.state.worker.stop()
        await run_in_threadpool(app.state.worker.wait,10)


app=FastAPI(title='Tìm người · Local AI',lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
app.add_middleware(CORSMiddleware,allow_origins=ALLOWED_ORIGINS,allow_methods=['*'],allow_headers=['*'])


@app.middleware('http')
async def local_origin(request,call_next):
    origin=request.headers.get('origin')
    if request.method not in ('GET','HEAD','OPTIONS') and origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse({'detail':'Ứng dụng chỉ chấp nhận nguồn cục bộ'},403)
    return await call_next(request)


@app.exception_handler(ValueError)
async def value_error(request,exc):
    return JSONResponse({'detail':str(exc)},422)


@app.exception_handler(KeyError)
async def key_error(request,exc):
    return JSONResponse({'detail':str(exc)},404)


def engine():
    if app.state.engine is None:
        raise HTTPException(503,app.state.runtime)
    return app.state.engine


def worker():
    engine()
    return app.state.worker


def idle():
    if app.state.worker and app.state.worker.active():
        raise HTTPException(409,'Dừng phiên đang xử lý trước khi đổi thư viện/watchlist/cấu hình')


@app.get('/api/status')
def status():
    e=app.state.engine;w=app.state.worker
    return {'runtime':e.status if e else app.state.runtime,'active_job':w.current if w else None,
            'configuration_calibrated':False}


@app.get('/api/config')
def get_config():
    return app.state.config


@app.put('/api/config')
async def put_config(body:Settings):
    idle();cfg=body.model_dump()
    if cfg['mode'] not in ('auto','cpu','cuda'):raise ValueError('mode không hợp lệ')
    if cfg['confirmation_seconds']>cfg['evidence_window_seconds']:raise ValueError('Cửa sổ bằng chứng phải lớn hơn thời gian xác nhận')
    e=engine()
    reload=any(cfg[k]!=app.state.config[k] for k in ('mode','device','det_size','face_detection_threshold'))
    if reload:
        old=e.cfg
        try:
            with e.lock:
                e.cfg=cfg
                await run_in_threadpool(e.load,cfg['mode'])
        except Exception:
            e.cfg=old
            await run_in_threadpool(e.load,old['mode'])
            raise
    e.cfg=cfg;app.state.config=cfg
    app.state.worker.config=cfg
    CONFIG_PATH.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    return cfg


@app.get('/api/people')
def people():return app.state.store.people()


@app.post('/api/people')
def create_person(body:PersonInput):
    idle();pid=app.state.store.create_person(body.name,body.notes)
    if body.watched:app.state.store.edit_person(pid,body.name,body.notes,body.watched)
    return {'id':pid}


@app.put('/api/people/{pid}')
def edit_person(pid:str,body:PersonInput):
    idle();app.state.store.edit_person(pid,body.name,body.notes,body.watched)
    return {'ok':True}


@app.delete('/api/people/{pid}')
def delete_person(pid:str):
    idle();app.state.store.delete_person(pid);return {'ok':True}


async def save_upload(upload,folder,limit,suffix):
    rel=f'{folder}/{uuid.uuid4().hex}{suffix}'
    path=DATA/rel;total=0
    try:
        with path.open('wb') as f:
            while chunk:=await upload.read(1024*1024):
                total+=len(chunk)
                if total>limit:raise HTTPException(413,'Tệp vượt dung lượng cho phép')
                f.write(chunk)
        if not total:raise ValueError('Tệp rỗng')
    except Exception:
        path.unlink(missing_ok=True);raise
    finally:
        await upload.close()
    return rel,path


@app.post('/api/people/{pid}/photos')
async def add_photo(pid:str,file:UploadFile=File(...)):
    idle();e=engine()
    if not any(p['id']==pid for p in app.state.store.people()):raise KeyError('Không tìm thấy người')
    rel,path=await save_upload(file,'photos',20*1024*1024,'.image')
    try:
        bgr=await run_in_threadpool(read_image,path)
        embedding,checkpoint=await run_in_threadpool(e.enroll,bgr)
        # Save normalized JPEG so browser can preview regardless of original image codec.
        path.unlink();path=path.with_suffix('.jpg');rel=str(Path(rel).with_suffix('.jpg')).replace('\\','/')
        cv2.imencode('.jpg',bgr)[1].tofile(str(path))
        photo=app.state.store.add_photo(pid,rel,embedding,checkpoint)
    except Exception as exc:
        path.unlink(missing_ok=True)
        if isinstance(exc,(ValueError,HTTPException)):raise
        raise HTTPException(422,f'Không xử lý được ảnh: {exc}') from exc
    return {'id':photo}


@app.get('/api/photos/{photo}')
def photo_file(photo:str):
    row=app.state.store.photo(photo)
    return FileResponse(DATA/row['path'],media_type='image/jpeg')


@app.delete('/api/photos/{photo}')
def delete_photo(photo:str):
    idle();app.state.store.delete_photo(photo);return {'ok':True}


@app.get('/api/jobs')
def jobs():return app.state.store.jobs()


def delete_saved_videos(job_id=None):
    w = app.state.worker
    with w.lock if w else nullcontext():
        blocked = w.current['id'] if w and w.current and w.thread and w.thread.is_alive() else None
        try:
            return app.state.store.delete_video_jobs(job_id, blocked)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc
        except OSError as exc:
            raise HTTPException(500, f'Không xóa được toàn bộ tệp; tác vụ vẫn được giữ để thử lại: {exc}') from exc


@app.delete('/api/videos')
def delete_all_saved_videos():return delete_saved_videos()


@app.delete('/api/videos/{jid}')
def delete_saved_video(jid:str):return delete_saved_videos(jid)


@app.post('/api/jobs/video')
async def upload_video(file:UploadFile=File(...)):
    w=worker();idle();name=file.filename or 'video'
    rel,path=await save_upload(file,'videos',app.state.config['max_upload_mb']*1024*1024,'.input')
    try:
        return w.start('video',str(path),name,app.state.config.copy())
    except Exception:
        path.unlink(missing_ok=True);raise


@app.post('/api/jobs/stop')
def stop():worker().stop();return {'ok':True}


@app.get('/api/events')
def events(job_id:str | None=None):return app.state.store.events(job_id)


@app.get('/api/events/{eid}/image')
def event_image(eid:str):
    event=app.state.store.event(eid)
    if not event.get('evidence'):raise HTTPException(404)
    path=DATA/event['evidence']
    if not path.is_file():raise HTTPException(404)
    return FileResponse(path,media_type='image/jpeg')


@app.get('/api/jobs/{jid}/video')
def job_video(jid:str,download:bool=False):
    job=next((j for j in app.state.store.jobs() if j['id']==jid),None)
    if (not re.fullmatch(r'[0-9a-f]{32}',jid) or not job or job.get('kind')!='video' or
            job.get('status') in ('running','stopping') or
            (download and job.get('status')!='completed') or
            job.get('output')!=f'videos/{jid}_annotated.mp4'):
        raise HTTPException(404)
    folder=(DATA/'videos').resolve()
    path=(DATA/job['output']).resolve()
    if folder.parent!=DATA.resolve() or path.parent!=folder or not path.is_file():
        raise HTTPException(404)
    original=Path(job.get('name') or 'video').stem
    stem=re.sub(r'[\\/:*?"<>|\x00-\x1f]','_',original).strip(' .')[:100] or 'video'
    return FileResponse(path,media_type='video/mp4',
                        filename=f'{stem}_annotated.mp4' if download else None)


@app.get('/api/cameras')
async def cameras():
    entries=await run_in_threadpool(slots,app.state.config)
    return [{k:v for k,v in entry.items() if k!='index'} for entry in entries]


@app.post('/api/live/start')
async def live_start(body:CameraInput):
    if body.source not in ('laptop','camo'):raise ValueError('Nguồn camera không hợp lệ')
    w=worker()
    if w.active():
        if w.current['kind']!='live':raise HTTPException(409,'Dừng video trước khi mở camera')
        w.stop();await run_in_threadpool(w.wait,10)
    return w.start('live',body.source,'Webcam Laptop' if body.source=='laptop' else 'Camo Camera',app.state.config.copy())


@app.get('/api/preview')
async def preview():
    w=worker()
    async def images():
        while w.active():
            if w.latest:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+w.latest+b'\r\n'
            await asyncio.sleep(.12)
    return StreamingResponse(images(),media_type='multipart/x-mixed-replace; boundary=frame')


@app.websocket('/ws')
async def websocket(ws:WebSocket):
    if ws.headers.get('origin') and ws.headers['origin'] not in ALLOWED_ORIGINS:
        await ws.close(code=1008);return
    await ws.accept();q=asyncio.Queue(maxsize=100);app.state.hub.clients.add(q)
    try:
        while True:
            try:
                payload=await asyncio.wait_for(q.get(),timeout=20)
            except asyncio.TimeoutError:
                payload={'type':'heartbeat'}
            await ws.send_json(payload)
    except (WebSocketDisconnect,RuntimeError):pass
    finally:app.state.hub.clients.discard(q)


dist=ROOT/'frontend/dist'
if dist.exists():
    app.mount('/',StaticFiles(directory=dist,html=True),name='ui')
