"""CLI: smoke test, enroll one reference image, or compare every detected face."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from baseline.pipeline import Pipeline, read_image


def save_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--device', help='Optional CUDA selector, e.g. cuda:1; otherwise runtime current device')
    parser.add_argument('--models', type=Path, default=Path(__file__).resolve().parent / 'models')
    parser.add_argument('--det-size', type=int, default=640)
    parser.add_argument('--det-threshold', type=float, default=0.5, help='Detection confidence, NOT identity threshold')
    parser.add_argument('--threads', type=int, default=4)
    sub = parser.add_subparsers(dest='command', required=True)
    smoke = sub.add_parser('smoke'); smoke.add_argument('--image'); smoke.add_argument('--output', default='results/smoke.json')
    enroll = sub.add_parser('enroll'); enroll.add_argument('--image', required=True); enroll.add_argument('--identity', required=True)
    enroll.add_argument('--face-index', type=int, help='Explicit choice when reference contains multiple faces')
    enroll.add_argument('--output', required=True)
    match = sub.add_parser('match'); match.add_argument('--image', required=True); match.add_argument('--reference', required=True)
    match.add_argument('--output', default='results/match.json')
    args = parser.parse_args()
    pipeline = Pipeline(args.mode, args.models, args.device, args.det_size, args.det_threshold, args.threads)
    payload = {'runtime': pipeline.status}
    if args.command == 'smoke':
        if args.image:
            faces = pipeline.analyze(read_image(args.image))
            payload['faces'] = [{k: v for k, v in f.items() if k != 'embedding'} |
                                {'embedding_dim': len(f['embedding']), 'l2_norm': float(np.linalg.norm(f['embedding']))}
                                for f in faces]
        payload['smoke_ok'] = True
    elif args.command == 'enroll':
        faces = pipeline.analyze(read_image(args.image))
        if not faces:
            raise ValueError('No reference face detected')
        if len(faces) != 1 and args.face_index is None:
            raise ValueError(f'{len(faces)} reference faces: specify --face-index explicitly')
        index = args.face_index if args.face_index is not None else 0
        if index < 0 or index >= len(faces):
            raise ValueError('Reference face index out of range')
        payload.update({'identity': args.identity, 'checkpoint_sha256': pipeline.checkpoint_hash,
                        'preprocessing': 'cvlface-rgb112-fivepoint-v1',
                        'embedding': faces[index]['embedding'].tolist(), 'reference_face': index,
                        'reference_face_count': len(faces)})
    else:
        reference = json.loads(Path(args.reference).read_text(encoding='utf8'))
        if reference['checkpoint_sha256'] != pipeline.checkpoint_hash or reference['preprocessing'] != 'cvlface-rgb112-fivepoint-v1':
            raise ValueError('Reference checkpoint/preprocessing incompatible')
        ref = np.asarray(reference['embedding'], dtype=np.float32)
        if ref.shape != (512,) or not np.isfinite(ref).all() or np.linalg.norm(ref) <= 1e-12:
            raise ValueError('Invalid reference embedding')
        ref /= np.linalg.norm(ref)
        faces = pipeline.analyze(read_image(args.image))
        payload.update({'identity': reference['identity'], 'identity_threshold': None,
                        'faces': [{k: v for k, v in f.items() if k != 'embedding'} |
                                  {'face_index': i, 'cosine_similarity': float(np.clip(f['embedding'] @ ref, -1, 1))}
                                  for i, f in enumerate(faces)]})
    save_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
