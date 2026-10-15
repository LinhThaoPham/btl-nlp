"""Download the authors' public educational-use PhoW2V mirror; no model redistribution.
Source/terms: https://github.com/datquocnguyen/PhoW2V
"""
from html.parser import HTMLParser
from pathlib import Path
import shutil
import time
import zipfile
import requests

ROOT = Path(__file__).resolve().parent.parent / 'models'
ROOT.mkdir(exist_ok=True)
ARCHIVE = ROOT / 'word2vec_vi_words_100dims.zip'
URL = 'https://drive.usercontent.google.com/download'
FILE_ID = '11u9vJTaVBgB7z-eucBqvWJi3bYq4OgIw'

class Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.params = {}

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'input' and a.get('type') == 'hidden' and 'name' in a:
            self.params[a['name']] = a.get('value', '')

if __name__ == '__main__':
    print('PhoW2V: educational/research use only. Do not redistribute model files.', flush=True)
    if shutil.disk_usage(ROOT).free < 4 * 1024**3:
        raise SystemExit('Cần ít nhất 4 GiB trống để tải, giải nén và lập chỉ mục PhoW2V. Chưa bắt đầu tải.')
    if not ARCHIVE.exists():
        session = requests.Session()
        response = session.get(URL, params={'id': FILE_ID, 'export': 'download'}, stream=True, timeout=60)
        response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            form = Form()
            form.feed(response.text)
            if form.params.get('id') != FILE_ID or 'uuid' not in form.params:
                raise RuntimeError('Mirror did not return the expected download form.')
            response.close()
            response = session.get(URL, params=form.params, stream=True, timeout=120)
            response.raise_for_status()
        if 'text/html' in response.headers.get('Content-Type', ''):
            raise RuntimeError('Mirror returned HTML instead of a model archive.')
        size = 0
        last = time.monotonic()
        with ARCHIVE.with_suffix('.zip.part').open('wb') as output:
            for chunk in response.iter_content(1024 * 1024):
                output.write(chunk)
                size += len(chunk)
                if time.monotonic() - last >= 10:
                    print(f'Downloaded {size / 1024**2:.0f} MiB', flush=True)
                    last = time.monotonic()
        ARCHIVE.with_suffix('.zip.part').replace(ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for item in archive.infolist():
            # Extract model data only, under a verified fixed directory.
            name = Path(item.filename).name
            if item.is_dir() or Path(name).suffix not in {'.bin', '.vec', '.txt'}:
                continue
            destination = (ROOT / name).resolve()
            if destination.parent != ROOT.resolve():
                raise RuntimeError('Invalid archive member.')
            if not destination.exists() or destination.stat().st_size != item.file_size:
                print(f'Extracting {name} ({item.file_size / 1024**2:.0f} MiB)', flush=True)
                with archive.open(item) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output, 1024 * 1024)
            print(f'Ready: {destination.name}', flush=True)
