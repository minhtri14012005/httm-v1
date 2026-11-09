"""Short integration checks on local pretrained models and upstream samples.

This measures software behavior, not recognition accuracy or speed.
"""
import gc
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from baseline.pipeline import Pipeline, ROOT, read_image


def main():
    results = {}
    reference_embeddings = {}
    sample = read_image(ROOT / 'samples/img1.jpeg')
    for mode in ('cpu', 'cuda'):
        try:
            pipe = Pipeline(mode)
            item = {'runtime': pipe.status, 'samples': []}
            for name in ('img1.jpeg', 'img2.jpeg', 'img3.jpeg'):
                faces = pipe.analyze(read_image(ROOT / 'samples' / name))
                entry = {'file': name, 'face_count': len(faces)}
                if len(faces) == 1:
                    feature = faces[0]['embedding']
                    if name == 'img1.jpeg':
                        reference_embeddings[mode] = feature
                    entry['cosine_to_img1'] = float(feature @ reference_embeddings[mode])
                    entry['l2_norm'] = float(np.linalg.norm(feature))
                item['samples'].append(entry)
            doubled = np.concatenate([sample, sample], axis=1)
            item['duplicated_image_face_count'] = len(pipe.analyze(doubled))
            if item['duplicated_image_face_count'] != 2:
                raise AssertionError('Expected two faces in duplicate sample collage')
            item['blank_image_face_count'] = len(pipe.analyze(np.zeros((256, 256, 3), np.uint8)))
            if item['blank_image_face_count'] != 0:
                raise AssertionError('Unexpected detection on blank input')
            results[mode] = item
            del pipe; gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception as exc:
            results[mode] = {'error': f'{type(exc).__name__}: {exc}'}
    if len(reference_embeddings) == 2:
        results['cpu_cuda_img1_embedding_cosine'] = float(reference_embeddings['cpu'] @ reference_embeddings['cuda'])
    # Controlled failure injection is labeled, not presented as a real missing GPU.
    with patch('torch.cuda.is_available', return_value=False):
        pipe = Pipeline('auto')
        results['simulated_no_cuda_auto'] = pipe.status
        results['simulated_no_cuda_auto']['sample_face_count'] = len(pipe.analyze(sample))
        assert pipe.status['selected'] == 'cpu'
    out = ROOT / 'results/integration.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2, allow_nan=False), encoding='utf8')
    print(out.read_text())
    if any('error' in results.get(mode, {}) for mode in ('cpu', 'cuda')):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
