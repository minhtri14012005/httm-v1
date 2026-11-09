"""Image-only SCRFD -> five-point alignment -> CVLFace AdaFace baseline."""
import gc
import hashlib
import json
from pathlib import Path
import uuid
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps
import torch  # Load bundled CUDA/cuDNN DLLs before importing ONNX Runtime.
import onnxruntime as ort
from safetensors.torch import load_file

from .iresnet import IR_101
from .scrfd import SCRFD
from download_models import EXPECTED

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = np.array([[38.2946, 51.6963], [73.5318, 51.5014],
                      [56.0252, 71.7366], [41.5493, 92.3655],
                      [70.7299, 92.2041]], dtype=np.float64)


def read_image(path):
    """Honor EXIF rotation and Unicode paths; detector receives BGR uint8."""
    with Image.open(path) as image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB'))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def align_face(bgr, landmarks):
    """Least-squares similarity (no reflection), all five landmarks, 112 square."""
    src = np.asarray(landmarks, dtype=np.float64)
    if src.shape != (5, 2) or not np.isfinite(src).all():
        raise ValueError('Expected five finite 2D landmarks')
    # x' = a*x - b*y + tx; y' = b*x + a*y + ty.
    a = np.zeros((10, 4)); target = REFERENCE.reshape(-1)
    a[0::2, 0] = src[:, 0]; a[0::2, 1] = -src[:, 1]; a[0::2, 2] = 1
    a[1::2, 0] = src[:, 1]; a[1::2, 1] = src[:, 0]; a[1::2, 3] = 1
    params, _, rank, _ = np.linalg.lstsq(a, target, rcond=None)
    if rank < 4 or np.hypot(*params[:2]) < 1e-8:
        raise ValueError('Degenerate landmarks')
    aa, bb, tx, ty = params
    matrix = np.array([[aa, -bb, tx], [bb, aa, ty]], dtype=np.float64)
    aligned = cv2.warpAffine(bgr, matrix, (112, 112), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned, matrix


def normalized_input(aligned_bgr):
    # The pinned CVLFace checkpoint is RGB, unlike original AdaFace BGR ckpts.
    rgb = aligned_bgr[:, :, ::-1].astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray((rgb / 127.5 - 1).transpose(2, 0, 1))).unsqueeze(0)


class Pipeline:
    def __init__(self, mode='auto', models=ROOT / 'models', device=None, det_size=640,
                 det_threshold=0.5, threads=4):
        if mode not in ('auto', 'cpu', 'cuda'):
            raise ValueError('mode must be auto, cpu or cuda')
        if det_size < 32 or det_size % 32:
            raise ValueError('det_size must be a positive multiple of 32')
        if not 0 < det_threshold < 1 or threads < 1:
            raise ValueError('Invalid detector threshold / threads')
        if device is not None and mode == 'cpu':
            raise ValueError('CPU mode does not accept a CUDA device selector')
        self.models = Path(models)
        self.det_size, self.det_threshold = det_size, det_threshold
        self.threads = threads
        torch.set_num_threads(threads)
        self.status = {'requested': mode, 'fallback_reason': None}
        self.detector = self.recognizer = None
        self._verify_assets()
        candidates = ['cpu'] if mode == 'cpu' else ['cuda', 'cpu'] if mode == 'auto' else ['cuda']
        for backend in candidates:
            try:
                self._load(backend, device)
                return
            except Exception as exc:
                self.detector = self.recognizer = None
                gc.collect()
                if backend == 'cuda' and mode == 'auto':
                    self.status['fallback_reason'] = f'{type(exc).__name__}: {exc}'
                    warnings.warn('Entire pipeline falling back to CPU: ' + self.status['fallback_reason'])
                    if torch.cuda.is_initialized():
                        torch.cuda.empty_cache()
                else:
                    raise

    def _verify_assets(self):
        manifest = json.loads((self.models / 'manifest.json').read_text(encoding='utf8'))
        for filename in ('scrfd_500m_kps.onnx', 'adaface_ir101_webface12m.safetensors', 'adaface_config.json'):
            matches = [x for x in manifest['assets'] if Path(x['file']).name == filename]
            if len(matches) != 1:
                raise ValueError(f'Missing manifest entry for {filename}')
            with (self.models / filename).open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            if actual != matches[0]['sha256'] or actual != EXPECTED[filename]:
                raise ValueError(f'Asset hash mismatch: {filename}')
        self.checkpoint_hash = next(x['sha256'] for x in manifest['assets']
                                    if Path(x['file']).name == 'adaface_ir101_webface12m.safetensors')
        cfg = json.loads((self.models / 'adaface_config.json').read_text())['conf']
        if (cfg['name'], cfg['color_space'], cfg['input_size'], cfg['output_dim']) != ('ir101', 'RGB', [3, 112, 112], 512):
            raise ValueError('Unsupported checkpoint configuration')

    def _load(self, backend, selector):
        self.device = torch.device('cpu')
        if backend == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('PyTorch CUDA unavailable')
            self.device = torch.device(selector or 'cuda')
            if self.device.type != 'cuda':
                raise ValueError('CUDA selector must be a CUDA device')
            # Resolve current device dynamically; optional selector is user supplied.
            index = self.device.index if self.device.index is not None else torch.cuda.current_device()
            if index not in range(torch.cuda.device_count()):
                raise ValueError('CUDA device selector out of range')
            self.device = torch.device('cuda', index)
            torch.ones(1, device=self.device).sum().item()
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('ONNX Runtime CUDA provider unavailable')
            providers = [('CUDAExecutionProvider', {'device_id': index}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        # Real model smoke inference and profiling determine execution, not provider presence.
        options.enable_profiling = True
        # Keep evidence in project results; avoid masking ORT errors with Windows
        # temporary-directory cleanup while a session still has an open profile.
        tmp = ROOT / 'results' / 'profiles' / uuid.uuid4().hex
        tmp.mkdir(parents=True, exist_ok=True)
        options.profile_file_prefix = str(Path(tmp) / 'scrfd')
        session = ort.InferenceSession(str(self.models / 'scrfd_500m_kps.onnx'),
                                      sess_options=options, providers=providers)
        session.disable_fallback()
        required = 'CUDAExecutionProvider' if backend == 'cuda' else 'CPUExecutionProvider'
        if required not in session.get_providers():
            raise RuntimeError(f'Session did not retain {required}')
        detector = SCRFD(session=session)
        if not detector.use_kps:
            raise RuntimeError('SCRFD model must provide five landmarks')
        detector.detect(np.zeros((self.det_size, self.det_size, 3), np.uint8),
                        thresh=self.det_threshold, input_size=(self.det_size, self.det_size))
        events = json.loads(Path(session.end_profiling()).read_text())
        placement = {}; cpu_ops = {}; cpu_metadata_only = True
        for event in events:
            provider = event.get('args', {}).get('provider')
            if provider:
                placement[provider] = placement.get(provider, 0) + 1
                if provider == 'CPUExecutionProvider':
                    op = event.get('args', {}).get('op_name', '?')
                    cpu_ops[op] = cpu_ops.get(op, 0) + 1
                    shapes = event.get('args', {}).get('input_type_shape', [])
                    cpu_metadata_only &= bool(shapes) and all(set(s) <= {'int64'} for s in shapes)
        if placement.get(required, 0) == 0:
            raise RuntimeError(f'No executed kernels found for {required}: {placement}')
        if backend == 'cuda' and (not cpu_metadata_only or set(cpu_ops) - {'Shape', 'Slice', 'Gather', 'Unsqueeze', 'Concat'}):
            raise RuntimeError(f'Unexpected CPU compute in CUDA session: {cpu_ops}')
        recognizer = IR_101((112, 112), output_dim=512)
        state = load_file(str(self.models / 'adaface_ir101_webface12m.safetensors'), device='cpu')
        if not all(k.startswith('model.net.') for k in state):
            raise ValueError('Unexpected safetensors state keys')
        recognizer.load_state_dict({k.removeprefix('model.net.'): v for k, v in state.items()}, strict=True)
        del state
        recognizer.eval().requires_grad_(False).to(self.device)
        with torch.inference_mode():
            output = recognizer(torch.zeros((1, 3, 112, 112), device=self.device))
        if output.shape != (1, 512) or not torch.isfinite(output).all() or output.norm() <= 0:
            raise RuntimeError('Invalid AdaFace smoke output')
        self.detector, self.recognizer = detector, recognizer
        self.status.update({'selected': backend, 'scrfd': required, 'scrfd_executed_kernels': placement,
                            'scrfd_cpu_ops': cpu_ops,
                            'scrfd_cpu_metadata_only': cpu_metadata_only if backend == 'cuda' else None,
                            'adaface': str(next(recognizer.parameters()).device),
                            'torch_version': torch.__version__, 'torch_cuda_runtime': torch.version.cuda,
                            'ort_version': ort.__version__, 'embedding_dim': 512,
                            'checkpoint_sha256': self.checkpoint_hash})
        if backend == 'cuda':
            self.status['gpu_name'] = torch.cuda.get_device_name(self.device)

    def detect(self, bgr):
        """Detection is separate from recognition so quality gates run before AdaFace."""
        boxes, points = self.detector.detect(bgr, thresh=self.det_threshold,
                                            input_size=(self.det_size, self.det_size))
        return [{'bbox': box[:4].tolist(), 'detection_score': float(box[4]),
                 'landmarks': landmarks.tolist()} for box, landmarks in zip(boxes, points)]

    def embed(self, aligned):
        with torch.inference_mode():
            feature = self.recognizer(normalized_input(aligned).to(self.device))
            norm = feature.norm(dim=1, keepdim=True)
            if not torch.isfinite(feature).all() or norm.item() <= 1e-12:
                raise RuntimeError('Nonfinite or zero embedding')
            return (feature / norm)[0].cpu().numpy(), float(norm.item())

    def analyze(self, bgr):
        results = self.detect(bgr)
        for face in results:
            aligned, matrix = align_face(bgr, face['landmarks'])
            embedding, norm = self.embed(aligned)
            face.update(alignment_matrix=matrix.tolist(), embedding_norm_before_l2=norm,
                        embedding=embedding)
        return results
