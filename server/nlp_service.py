"""Local JSON-lines NLP service. Text and model vectors never leave this machine."""
import contextlib
import importlib.metadata
import json
import math
import re
import sqlite3
import struct
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
MODELS.mkdir(exist_ok=True)
MODEL = None
TOKEN = re.compile(r'[^\W_]+(?:_[^\W_]+)*', re.UNICODE)
# PhoW2V's documented tone placement convention, applied only for vector lookup.
TONE = dict(zip('òa óa ỏa õa ọa òe óe ỏe õe ọe ùy úy ủy ũy ụy'.split(),
                'oà oá oả oã oạ oè oé oẻ oẽ oẹ uỳ uý uỷ uỹ uỵ'.split()))

def canonical(text):
    return unicodedata.normalize('NFC', text.lower()).replace(' ', '_')

def vector_key(text):
    for a, b in TONE.items():
        text = text.replace(a, b)
    return text

def available_models():
    return sorted(p.name for p in MODELS.iterdir() if p.suffix.lower() in {'.bin', '.vec', '.txt'})

def load_model(name):
    global MODEL
    path = (MODELS / name).resolve()
    if path.parent != MODELS.resolve() or not path.is_file() or path.name not in available_models():
        raise ValueError('Chọn tệp .bin/.vec/.txt nằm trong thư mục models.')
    identity = f'{path.name}:{path.stat().st_size}:{path.stat().st_mtime_ns}'
    if MODEL and MODEL['id'] == identity:
        return model_info()
    db_path = MODELS / (path.name + '.sqlite')
    conn = sqlite3.connect(db_path)
    try:
        conn.execute('CREATE TABLE IF NOT EXISTS meta (id TEXT, dim INTEGER, count INTEGER)')
        meta = conn.execute('SELECT id,dim,count FROM meta').fetchone()
        if not meta or meta[0] != identity:
            with conn:
                conn.execute('DROP TABLE IF EXISTS vectors')
                conn.execute('CREATE TABLE vectors (word TEXT PRIMARY KEY, vector BLOB NOT NULL)')
                conn.execute('DELETE FROM meta')
                with path.open('rb') as stream:
                    header = stream.readline().decode('utf-8').strip().split()
                    if len(header) != 2 or not all(x.isdigit() for x in header):
                        raise ValueError('Word2Vec cần dòng đầu: số_từ số_chiều. Hỗ trợ text và binary.')
                    count, dim = map(int, header)
                    if not 1 <= count <= 3_000_000 or not 1 <= dim <= 1024:
                        raise ValueError('Kích thước mô hình không hợp lệ.')
                    batch = []
                    for i in range(count):
                        if path.suffix.lower() == '.bin':
                            word = bytearray()
                            while True:
                                char = stream.read(1)
                                if not char:
                                    raise ValueError('Tệp binary thiếu dữ liệu.')
                                if char == b' ':
                                    break
                                if char not in b'\r\n':
                                    word.extend(char)
                                if len(word) > 4096:
                                    raise ValueError('Token trong mô hình quá dài.')
                            values = struct.unpack('<' + 'f' * dim, stream.read(4 * dim))
                            word = word.decode('utf-8')
                        else:
                            # PhoW2V includes a non-breaking-space token. Split only
                            # ASCII separators; Unicode split() would erase that token.
                            parts = [part for part in stream.readline().decode('utf-8').rstrip('\r\n').split(' ') if part]
                            if len(parts) < dim + 1:
                                raise ValueError(f'Sai số chiều tại dòng {i + 2}.')
                            word, values = ' '.join(parts[:-dim]), tuple(map(float, parts[-dim:]))
                        if not all(math.isfinite(v) for v in values):
                            raise ValueError('Vector chứa NaN hoặc Infinity.')
                        batch.append((canonical(word), struct.pack('<' + 'f' * dim, *values)))
                        if len(batch) == 4000:
                            conn.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?)', batch)
                            batch = []
                    conn.executemany('INSERT OR REPLACE INTO vectors VALUES (?,?)', batch)
                actual = conn.execute('SELECT COUNT(*) FROM vectors').fetchone()[0]
                conn.execute('INSERT INTO meta VALUES (?,?,?)', (identity, dim, actual))
                meta = (identity, dim, actual)
    except Exception:
        conn.close()
        raise
    if MODEL:
        MODEL['db'].close()
    MODEL = {'id': identity, 'name': path.name, 'dim': meta[1], 'count': meta[2], 'db': conn}
    return model_info()

def model_info():
    return {k: v for k, v in MODEL.items() if k != 'db'} if MODEL else None

def tokenize_rows(lines, mode):
    if mode == 'basic':
        return [[{'word': w, 'pos': ''} for w in TOKEN.findall(unicodedata.normalize('NFC', s.lower()))] for s in lines]
    # Preserve case for POS, then normalize model output. Existing underscores stay tokens.
    from underthesea import pos_tag
    rows = []
    for line in lines:
        tagged = pos_tag(unicodedata.normalize('NFC', line))
        rows.append([{'word': canonical(word), 'pos': tag} for word, tag in tagged
                     if TOKEN.fullmatch(canonical(word))])
    return rows

def handle(data):
    action = data.get('action', 'status')
    if action == 'status':
        try:
            version = importlib.metadata.version('underthesea')
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {'underthesea': version, 'models': available_models(), 'model': model_info()}
    if action == 'load':
        return load_model(data['name'])
    if action != 'prepare':
        raise ValueError('Yêu cầu NLP không hợp lệ.')
    lines = data.get('lines', [])
    if not isinstance(lines, list) or not all(isinstance(s, str) for s in lines) or len(lines) > 500 or sum(map(len, lines)) > 500_000:
        raise ValueError('Mỗi lô tối đa 500 đoạn và 500.000 ký tự.')
    started = time.perf_counter()
    rows = tokenize_rows(lines, data.get('tokenizer', 'basic'))
    vectors = None
    if data.get('embedding'):
        if not MODEL:
            names = available_models()
            if not names:
                raise ValueError('Chưa có mô hình Word2Vec. Nạp tệp ở mục Mô hình ngôn ngữ.')
            load_model(names[0])
        wanted = {t['word'] for row in rows for t in row}
        wanted |= {part for word in list(wanted) for part in word.split('_')}
        vectors = {}
        for word in wanted:
            hit = MODEL['db'].execute('SELECT vector FROM vectors WHERE word=?', (vector_key(word),)).fetchone()
            if hit is None:
                hit = MODEL['db'].execute('SELECT vector FROM vectors WHERE word=?', (word,)).fetchone()
            if hit:
                vectors[word] = list(struct.unpack('<' + 'f' * MODEL['dim'], hit[0]))
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except ImportError:
        rss = None
    return {'rows': rows, 'vectors': vectors, 'model': model_info() if data.get('embedding') else None,
            'elapsedMs': (time.perf_counter() - started) * 1000, 'processRssBytes': rss,
            'tokenizer': data.get('tokenizer', 'basic')}

if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    for raw in sys.stdin:
        request = {}
        try:
            request = json.loads(raw)
            # Keep third-party progress output out of the JSON protocol.
            with contextlib.redirect_stdout(sys.stderr):
                result = handle(request)
            response = {'id': request['id'], 'result': result}
        except Exception as exc:
            response = {'id': request.get('id'), 'error': str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)
