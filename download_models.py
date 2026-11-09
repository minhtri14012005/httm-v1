"""Download only the pinned baseline assets. Never called during inference."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
REV = '54f602a0737bd1ee4a4e7e9fd089a485f397fefd'
HF = f'https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m/resolve/{REV}/'
EXPECTED = {
    'adaface_ir101_webface12m.safetensors': '2ea535a43877bd3de8091903935c783ce335be66a9f8917fae9a7a18ae4bbf56',
    'adaface_config.json': '6311d95b23368f06932e6c04b52431c29857e1fcbf5e1d41a2e05f5e4b946d10',
    'buffalo_sc.zip': '57d31b56b6ffa911c8a73cfc1707c73cab76efe7f13b675a05223bf42de47c72',
    'scrfd_500m_kps.onnx': '5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a',
    'img1.jpeg': '519346b222f4b8b9be453d5866c5e235eec4f92540afd7fb0e1fc975e20be7e1',
    'img2.jpeg': 'a077b3c0f30cb717178b42bb71bd997d8d514958b09b288587f727163b4265f7',
    'img3.jpeg': 'dd70ecd097fa82f133d9791721749002748ed6b42b03db04a2050e57d7bfb6e1',
    'yolo11n.pt': '0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1',
}


def fetch(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.partial')
        print('Downloading', target.name, flush=True)
        request = urllib.request.Request(url, headers={'Accept': 'application/octet-stream', 'User-Agent': 'local-ai-development'})
        with urllib.request.urlopen(request, timeout=120) as src, partial.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        partial.replace(target)
    digest = sha256(target)
    if digest != EXPECTED[target.name]:
        raise ValueError(f'Unexpected asset hash: {target}; no model will be loaded')
    return {'url': url, 'file': target.relative_to(ROOT).as_posix(),
            'sha256': digest,
            'bytes': target.stat().st_size}


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    models = ROOT / 'models'
    records = [fetch(HF + 'model.safetensors', models / 'adaface_ir101_webface12m.safetensors'),
               fetch(HF + 'config.json', models / 'adaface_config.json')]
    records.append(fetch('https://api.github.com/repos/ultralytics/assets/releases/assets/196201276', models / 'yolo11n.pt'))
    archive = models / 'buffalo_sc.zip'
    records.append(fetch('https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip', archive))
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if Path(n).name == 'det_500m.onnx']
        if len(names) != 1:
            raise RuntimeError(f'Expected det_500m.onnx; archive has {z.namelist()}')
        target = models / 'scrfd_500m_kps.onnx'
        with z.open(names[0]) as src, target.open('wb') as dst:
            shutil.copyfileobj(src, dst)
        if sha256(target) != EXPECTED[target.name]:
            raise ValueError('Extracted SCRFD hash mismatch')
        records.append({'archive_member': names[0], 'file': target.relative_to(ROOT).as_posix(),
                        'sha256': sha256(target), 'bytes': target.stat().st_size})
    for n in ('img1.jpeg', 'img2.jpeg', 'img3.jpeg'):
        records.append(fetch('https://raw.githubusercontent.com/mk-minchul/AdaFace/c60eaa786a42c03444f3df7096dbaf9d57ae010d/face_alignment/test_images/' + n,
                             ROOT / 'samples' / n))
    (models / 'manifest.json').write_text(json.dumps({'adaface_revision': REV, 'assets': records}, indent=2), encoding='utf8')
    print('Saved models/manifest.json', flush=True)


if __name__ == '__main__':
    main()
