"""Optional workspace-local Node.js for Windows. No system PATH changes."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parent


def main():
    base='https://nodejs.org/dist/latest-v22.x/'
    lines=urllib.request.urlopen(base+'SHASUMS256.txt',timeout=40).read().decode().splitlines()
    digest,name=next(s for s in lines if s.endswith('-win-x64.zip')).split()
    tools=ROOT/'.tools';tools.mkdir(exist_ok=True);archive=tools/name
    if not archive.exists():
        with urllib.request.urlopen(base+name,timeout=120) as src,archive.open('wb') as dst:
            import shutil
            shutil.copyfileobj(src,dst)
    with archive.open('rb') as f:
        if hashlib.file_digest(f,'sha256').hexdigest()!=digest:
            raise RuntimeError('Node archive SHA256 mismatch')
    with zipfile.ZipFile(archive) as z:z.extractall(tools)
    print(tools/name.removesuffix('.zip'))


if __name__=='__main__':main()
