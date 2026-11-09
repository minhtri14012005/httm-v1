import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / 'app.sqlite3'
        self.lock = threading.RLock()
        with self.connection() as c:
            c.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS people(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, notes TEXT NOT NULL DEFAULT '',
                    watched INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS photos(
                    id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
                    path TEXT NOT NULL, embedding TEXT NOT NULL, checkpoint TEXT NOT NULL, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(
                    id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), person_id TEXT NOT NULL,
                    payload TEXT NOT NULL);
            ''')

    @contextmanager
    def connection(self):
        c = sqlite3.connect(self.db, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON')
        try:
            with c:
                yield c
        finally:
            c.close()

    def people(self):
        with self.lock, self.connection() as c:
            people = [dict(x) for x in c.execute('SELECT * FROM people ORDER BY created DESC')]
            for p in people:
                p['watched'] = bool(p['watched'])
                p['photos'] = [dict(x) for x in c.execute('SELECT id,person_id,created FROM photos WHERE person_id=?', (p['id'],))]
        return people

    def create_person(self, name, notes=''):
        if not name.strip():
            raise ValueError('Tên không được để trống')
        pid = uuid.uuid4().hex
        with self.lock, self.connection() as c:
            c.execute('INSERT INTO people VALUES(?,?,?,?,?)', (pid, name.strip()[:150], notes[:2000], 0, time.time()))
        return pid

    def edit_person(self, pid, name, notes, watched):
        if not name.strip():
            raise ValueError('Tên không được để trống')
        with self.lock, self.connection() as c:
            if not c.execute('UPDATE people SET name=?,notes=?,watched=? WHERE id=?',
                             (name.strip()[:150], notes[:2000], int(watched), pid)).rowcount:
                raise KeyError('Không tìm thấy người')

    def delete_person(self, pid):
        with self.lock, self.connection() as c:
            paths = [x['path'] for x in c.execute('SELECT path FROM photos WHERE person_id=?', (pid,))]
            c.execute('DELETE FROM people WHERE id=?', (pid,))
        for path in paths:
            (self.root / path).unlink(missing_ok=True)

    def add_photo(self, pid, path, embedding, checkpoint):
        photo = uuid.uuid4().hex
        with self.lock, self.connection() as c:
            c.execute('INSERT INTO photos VALUES(?,?,?,?,?,?)',
                      (photo, pid, path, json.dumps(embedding.tolist()), checkpoint, time.time()))
        return photo

    def photo(self, photo):
        with self.connection() as c:
            row = c.execute('SELECT * FROM photos WHERE id=?', (photo,)).fetchone()
        if row is None:
            raise KeyError('Không tìm thấy ảnh')
        return dict(row)

    def delete_photo(self, photo):
        row = self.photo(photo)
        with self.lock, self.connection() as c:
            c.execute('DELETE FROM photos WHERE id=?', (photo,))
        (self.root / row['path']).unlink(missing_ok=True)

    def templates(self, checkpoint):
        with self.connection() as c:
            return [dict(x) for x in c.execute('''SELECT p.id,p.name,f.embedding FROM people p
                JOIN photos f ON p.id=f.person_id WHERE p.watched=1 AND f.checkpoint=?''', (checkpoint,))]

    def put_job(self, job):
        with self.lock, self.connection() as c:
            c.execute('INSERT OR REPLACE INTO jobs VALUES(?,?,?)', (job['id'], job['kind'], json.dumps(job)))

    def jobs(self):
        with self.connection() as c:
            return [json.loads(x['payload']) for x in c.execute('SELECT payload FROM jobs ORDER BY rowid DESC')]

    def delete_video_jobs(self, job_id=None, blocked_job_id=None):
        """Delete only completed video jobs and their own identified files."""
        with self.lock, self.connection() as c:
            if job_id is None:
                rows = c.execute("SELECT id,payload FROM jobs WHERE kind='video'").fetchall()
            else:
                rows = c.execute("SELECT id,payload FROM jobs WHERE kind='video' AND id=?", (job_id,)).fetchall()
                if not rows:
                    raise KeyError('Không tìm thấy video đã lưu')
            jobs = [json.loads(row['payload']) for row in rows]
            if any(j['status'] in ('running', 'stopping') or j['id'] == blocked_job_id for j in jobs):
                raise RuntimeError('Không thể xóa video đang xử lý. Hãy đợi hoặc dừng tác vụ trước.')

            paths = set()
            untracked = 0
            root = self.root.resolve()
            def add_path(relative, folder, pattern):
                if not relative:
                    return
                rel = Path(relative)
                if rel.is_absolute() or len(rel.parts) != 2 or rel.parts[0] != folder or not re.fullmatch(pattern, rel.name):
                    raise ValueError('Đường dẫn tệp của tác vụ không hợp lệ; chưa xóa dữ liệu.')
                path = (root / rel).resolve()
                if path.parent != (root / folder).resolve():
                    raise ValueError('Đường dẫn tệp của tác vụ không hợp lệ; chưa xóa dữ liệu.')
                paths.add(path)

            for job in jobs:
                jid = job['id']
                if not re.fullmatch(r'[0-9a-f]{32}', jid):
                    raise ValueError('Mã tác vụ không hợp lệ; chưa xóa dữ liệu.')
                add_path(job.get('output'), 'videos', re.escape(jid + '_annotated.mp4'))
                if job.get('input'):
                    add_path(job['input'], 'videos', r'[0-9a-f]{32}\.input')
                else:
                    untracked += 1
                for row in c.execute('SELECT id,payload FROM events WHERE job_id=?', (jid,)):
                    event = json.loads(row['payload'])
                    eid = row['id']
                    if not re.fullmatch(r'[0-9a-f]{32}', eid):
                        raise ValueError('Mã sự kiện không hợp lệ; chưa xóa dữ liệu.')
                    add_path(event.get('evidence'), 'evidence', re.escape(eid + '.jpg'))
            for path in paths:
                path.unlink(missing_ok=True)
            for job in jobs:
                c.execute('DELETE FROM events WHERE job_id=?', (job['id'],))
                c.execute('DELETE FROM jobs WHERE id=?', (job['id'],))
        return {'deleted': len(jobs), 'untracked_inputs': untracked}

    def put_event(self, event):
        with self.lock, self.connection() as c:
            c.execute('INSERT OR REPLACE INTO events VALUES(?,?,?,?)',
                      (event['id'], event['job_id'], event['person_id'], json.dumps(event)))

    def events(self, job=None):
        with self.connection() as c:
            if job:
                rows = c.execute('SELECT payload FROM events WHERE job_id=? ORDER BY rowid', (job,))
            else:
                rows = c.execute('SELECT payload FROM events ORDER BY rowid DESC LIMIT 500')
            return [json.loads(x['payload']) for x in rows]

    def event(self, event_id):
        with self.connection() as c:
            row = c.execute('SELECT payload FROM events WHERE id=?', (event_id,)).fetchone()
        if row is None:
            raise KeyError('Không tìm thấy sự kiện')
        return json.loads(row['payload'])
