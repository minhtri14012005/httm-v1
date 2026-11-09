"""Conservative, per-track face evidence; never inherit identity across IDs."""
from collections import defaultdict, deque
import uuid

import numpy as np


class Gallery:
    def __init__(self, rows):
        self.people = {}
        for row in rows:
            p = self.people.setdefault(row['id'], {'id': row['id'], 'name': row['name'], 'vectors': []})
            import json
            v = np.asarray(json.loads(row['embedding']), dtype=np.float32)
            if v.shape == (512,) and np.isfinite(v).all() and np.linalg.norm(v) > 0:
                p['vectors'].append(v / np.linalg.norm(v))

    def rank(self, embedding):
        ranked = []
        for p in self.people.values():
            if p['vectors']:
                score = max(float(v @ embedding) for v in p['vectors'])
                ranked.append((score, p['id'], p['name']))
        return sorted(ranked, reverse=True)


def associate(faces, tracks):
    """Require a unique person box enclosing face in its upper region."""
    linked = {}
    for face in faces:
        if face.get('quality_status', 'usable') != 'usable':
            continue
        face['association_status'] = 'no_person'
        x1, y1, x2, y2 = face['bbox']
        cx, cy = (x1+x2)/2, (y1+y2)/2
        area = max((x2-x1)*(y2-y1), 1)
        possible = []
        for track in tracks:
            a, b, c, d = track['bbox']
            containment = max(0, min(x2,c)-max(x1,a))*max(0,min(y2,d)-max(y1,b))/area
            if a <= cx <= c and b <= cy <= b+(d-b)*0.65 and containment >= 0.85:
                possible.append(track['track_id'])
        if len(possible) == 1:
            tid = possible[0]
            face['association_status'] = 'linked'
            # Two faces assigned to one body are ambiguous: suppress both.
            if tid in linked:
                if linked[tid] is not None:
                    linked[tid]['association_status'] = 'ambiguous'
                linked[tid] = None
                next(t for t in tracks if t['track_id']==tid)['association_ambiguous'] = True
                face['association_status'] = 'ambiguous'
            else:
                linked[tid] = face
        elif len(possible) > 1:
            face['association_status'] = 'ambiguous'
            for track in tracks:
                if track['track_id'] in possible:
                    track['association_ambiguous'] = True
    return {k: v for k, v in linked.items() if v is not None}


class Evidence:
    def __init__(self, config, gallery, job_id, *, reset_ambiguous_pending=False):
        self.cfg, self.gallery, self.job_id = config, gallery, job_id
        self.reset_ambiguous_pending = reset_ambiguous_pending
        self.states = {}
        self.closed = []

    @staticmethod
    def new_state(now):
        return {'first_seen': now, 'last_seen': now, 'votes': deque(), 'event': None,
                'last_verified': None, 'was_missing': False, 'suspended': False}

    def expire(self, now):
        for tid, state in list(self.states.items()):
            if now - state['last_seen'] > self.cfg['lost_seconds']:
                self.finish(tid, now, 'lost')

    def finish(self, tid, now, reason):
        state = self.states.pop(tid, None)
        if state and state.get('event'):
            event = state['event']
            event.update(last_seen=state['last_seen'], lost_at=now if reason == 'lost' else None,
                         ended_at=now, end_reason=reason)
            self.closed.append(event.copy())

    def update(self, tracks, linked, now):
        self.expire(now)
        updates, alerts = [], []
        present = {t['track_id'] for t in tracks}
        for tid, state in self.states.items():
            if tid not in present:
                state['was_missing'] = True
                state['votes'].clear()
                if state['event']:
                    state['suspended'] = True
        # A face of one identity assigned to several tracks is not reliable.
        proposals = {}
        for tid, face in linked.items():
            ranked = self.gallery.rank(face['embedding'])
            face['matching_status'] = 'below_threshold'
            if ranked:
                face['similarity'] = ranked[0][0]
            if ranked and ranked[0][0] >= self.cfg['identity_threshold']:
                second = ranked[1][0] if len(ranked)>1 else -1
                face['matching_status'] = 'ambiguous_margin'
                if ranked[0][0]-second >= self.cfg['identity_margin']:
                    proposals[tid] = ranked[0]
                    face['matching_status'] = 'candidate'
        duplicates = defaultdict(list)
        for tid, result in proposals.items():
            duplicates[result[1]].append(tid)
        proposals = {tid: r for tid,r in proposals.items() if len(duplicates[r[1]])==1}
        for tid, face in linked.items():
            if face['matching_status']=='candidate' and tid not in proposals:
                face['matching_status'] = 'ambiguous_identity'
        for track in tracks:
            tid = track['track_id']
            state = self.states.setdefault(tid, self.new_state(now))
            proposal = proposals.get(tid)
            if track.get('association_ambiguous'):
                proposal = None
                # Pending votes predate an uncertain face-to-person link. They
                # cannot be combined with evidence from after that interruption.
                if self.reset_ambiguous_pending and not state['event']:
                    state['votes'].clear()
                if tid in linked:
                    linked[tid]['matching_status'] = 'ambiguous_association'
            # A different face is conflicting evidence. End this encounter before
            # collecting votes for another identity; never relabel it in place.
            if state['event'] and proposal and proposal[1] != state['event']['person_id']:
                self.finish(tid, now, 'conflicting_face')
                state = self.new_state(now)
                self.states[tid] = state
            state['last_seen'] = now
            state['was_missing'] = False
            while state['votes'] and now-state['votes'][0][0] > self.cfg['evidence_window_seconds']:
                state['votes'].popleft()
            # Loss of face evidence suspends the visible identity, not the event.
            # A continuous body track alone is trusted only for the bounded grace period.
            ambiguous_or_mismatch = track.get('association_ambiguous') or (tid in linked and not proposal)
            if state['event'] and (ambiguous_or_mismatch or
                    now - state['last_verified'] > self.cfg['revalidate_seconds']):
                if not state['suspended'] or ambiguous_or_mismatch:
                    state['votes'].clear()
                state['suspended'] = True
            if tid in linked and not proposal:
                state['votes'].clear()
            if proposal:
                score, pid, name = proposal
                if state['votes'] and state['votes'][-1][1] != pid:
                    state['votes'].clear()
                state['votes'].append((now, pid, score))
                if state['event'] and state['event']['person_id'] == pid:
                    if state['suspended']:
                        if (len(state['votes']) >= self.cfg['confirmation_hits'] and
                                now-state['votes'][0][0] >= self.cfg['confirmation_seconds']):
                            state['suspended'] = False
                            state['last_verified'] = now
                    else:
                        state['last_verified'] = now
                if not state['event'] and len(state['votes']) >= self.cfg['confirmation_hits'] and now-state['votes'][0][0] >= self.cfg['confirmation_seconds']:
                    # Do not confirm another simultaneous track of the same identity.
                    occupied = any(s['event'] and s['event']['person_id']==pid for k,s in self.states.items() if k!=tid)
                    if not occupied:
                        state['event'] = {'id': uuid.uuid4().hex, 'job_id': self.job_id, 'person_id': pid,
                                          'name': name, 'track_id': tid, 'first_seen': state['first_seen'],
                                          'confirmed_at': now, 'last_seen': now, 'lost_at': None,
                                          'ended_at': None, 'end_reason': None, 'score': score, 'evidence': None}
                        state['last_verified'] = now
                        alerts.append(state['event'].copy())
            event = state['event']
            if event:
                event['last_seen'] = now
                status = ('unconfirmed' if state['suspended'] else
                          'face_verified' if proposal else 'tracking_unverified')
                event['verification_status'] = status
                updates.append(event.copy())
                if status == 'unconfirmed':
                    track.update(person_id=None, name=None, identity_status=status)
                else:
                    track.update(person_id=event['person_id'], name=event['name'], identity_status=status)
                    track['event_id'] = event['id']
                    track['last_face_verified'] = state['last_verified']
                if status == 'face_verified' and tid in linked:
                    linked[tid]['matching_status'] = 'verified'
            else:
                track.update(person_id=None, name=None, identity_status='unconfirmed')
            track['face_state'] = linked[tid].get('matching_status') if tid in linked else 'not_usable_or_not_associated'
            if proposal:
                track['candidate_score'] = proposal[0]
                if not event:
                    track['candidate_id'] = proposal[1]
                    track['candidate_name'] = proposal[2]
            track['confirmation_count'] = len(state['votes'])
        closed, self.closed = self.closed, []
        return updates + closed, alerts

    def end(self, now, reason):
        for tid in list(self.states):
            self.finish(tid, now, reason)
        closed, self.closed = self.closed, []
        return closed
