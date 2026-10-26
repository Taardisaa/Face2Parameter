"""Bridge HTTP steps of the chin-fit paired sessions (Maker, card, hair, expression, state).

Used between the probe session script steps (``paired_capture_session.ps1``). Every call and
its response is written to the session directory; nothing is retried silently.

    python tools/chin_fit/session_bridge.py --session <dir> status
    python tools/chin_fit/session_bridge.py --session <dir> maker-start
    python tools/chin_fit/session_bridge.py --session <dir> load-card --card <png> [--expect-sha <sha>]
    python tools/chin_fit/session_bridge.py --session <dir> record --name <label>
    python tools/chin_fit/session_bridge.py --session <dir> hair-catalog --part front
    python tools/chin_fit/session_bridge.py --session <dir> set-hair --part front --id <id>
    python tools/chin_fit/session_bridge.py --session <dir> express --blink false
    python tools/chin_fit/session_bridge.py --session <dir> apply-abmx --patch <json>
    python tools/chin_fit/session_bridge.py --session <dir> save-card --card <new png> --thumbnail <png>
    python tools/chin_fit/session_bridge.py --session <dir> renderstyle --body <json>

Measurement-domain changes (front hair removed, blink disabled) are scene state of the capture
windows only; they are never saved into a card. ``save-card`` refuses an existing path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE = 'http://127.0.0.1:43127/'
ROUTES = {'snapshot': 'maker/snapshot?region=all', 'abmx': 'maker/abmx',
          'expression': 'maker/face/express', 'renderstyle': 'maker/renderstyle'}
CARD_DIR = Path('E:/HoneySelect2_ArcticFox/UserData/chara/female/Codex')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def call(route, body=None, timeout=300):
    req = urllib.request.Request(BASE + route, data=None if body is None else json.dumps(body).encode('utf-8'),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, {'error_body': e.read().decode('utf-8', 'replace')}


def state():
    out = {}
    for k, route in ROUTES.items():
        code, body = call(route)
        if code != 200:
            raise RuntimeError('%s -> %s %s' % (route, code, body))
        out[k] = body
    return out


class Session:
    def __init__(self, path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)

    def log(self, name, value):
        p = self.dir / (name + '.json')
        if p.exists():
            p = self.dir / ('%s_%d.json' % (name, int(time.time() * 1000)))
        p.write_text(json.dumps(value, ensure_ascii=False, indent=1), encoding='utf-8')
        return p

    def request(self, name, method, route, body=None, timeout=300):
        code, resp = call(route, body if method == 'POST' else None, timeout=timeout)
        self.log('bridge_' + name, {'request': {'method': method, 'route': route, 'body': body}, 'status': code,
                                    'response': resp, 'utc': time.time()})
        if code not in (200, 202):
            raise RuntimeError('%s %s -> %s: %s' % (method, route, code, json.dumps(resp)[:2000]))
        return resp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--session', required=True, type=Path)
    ap.add_argument('step', choices=['status', 'maker-start', 'load-card', 'record', 'hair-catalog', 'set-hair',
                                     'express', 'apply-abmx', 'save-card', 'renderstyle'])
    ap.add_argument('--card', type=Path)
    ap.add_argument('--expect-sha')
    ap.add_argument('--name')
    ap.add_argument('--part')
    ap.add_argument('--id', type=int)
    ap.add_argument('--blink', choices=['true', 'false'])
    ap.add_argument('--patch', type=Path)
    ap.add_argument('--thumbnail', type=Path)
    ap.add_argument('--body', type=Path)
    a = ap.parse_args(argv)
    s = Session(a.session)
    if a.step == 'status':
        code, body = call('status')
        print(code, json.dumps(body, ensure_ascii=False)[:2000])
    elif a.step == 'maker-start':
        print(json.dumps(s.request('maker_start', 'POST', 'game/maker/start', {'sex': 'female'}), ensure_ascii=False))
    elif a.step == 'load-card':
        if a.expect_sha and sha(a.card) != a.expect_sha:
            raise RuntimeError('card sha differs from expected: %s' % sha(a.card))
        r = s.request('card_load', 'POST', 'maker/card/load', {'path': str(a.card)})
        print(json.dumps(r, ensure_ascii=False))
    elif a.step == 'record':
        st = dict(state(), utc=time.time())
        if a.card:
            st['card_sha256'] = sha(a.card)
        p = s.log('state_' + a.name, st)
        print('recorded', p)
    elif a.step == 'hair-catalog':
        pages, offset = [], 0
        while True:
            code, body = call('maker/hair/catalog?offset=%d&limit=500' % offset)
            if code != 200:
                raise RuntimeError('catalog %s %s' % (code, body))
            pages.append(body)
            offset += body['count']
            if body['count'] == 0 or offset >= body['total']:
                break
        items = [i for b in pages for i in b['items'] if i['part'] == (a.part or 'front')]
        s.log('hair_catalog_' + (a.part or 'front'), {'pages': pages})
        for i in items:
            if i['id'] in (0, -1) or any(t in (i['name'] or '') for t in ('无', 'なし', 'None', 'none', 'NONE')):
                print(json.dumps({k: i[k] for k in ('id', 'name', 'mod_guid', 'main_ab', 'main_data')},
                                 ensure_ascii=False))
    elif a.step == 'set-hair':
        r = s.request('set_hair_%s_%d' % (a.part, a.id), 'POST', 'maker/hair', {'part': a.part, 'id': a.id})
        print(json.dumps(r, ensure_ascii=False)[:2000])
    elif a.step == 'express':
        cur = state()['expression']
        body = dict(cur, eyes_blink=(a.blink == 'true'))
        r = s.request('express', 'POST', 'maker/face/express', body)
        print(json.dumps(r, ensure_ascii=False)[:2000])
    elif a.step == 'apply-abmx':
        patch = json.loads(a.patch.read_text(encoding='utf-8'))
        bones = [{k: b[k] for k in ('name', 'position', 'rotation', 'scale', 'length')} for b in patch['bones']]
        r = s.request('abmx_batch', 'POST', 'maker/abmx/batch', {'bones': bones})
        print(json.dumps(r, ensure_ascii=False)[:3000])
    elif a.step == 'save-card':
        card = a.card.resolve()
        if card.exists():
            raise FileExistsError('never overwrite an existing card: %s' % card)
        if card.parent != CARD_DIR.resolve():
            raise ValueError('new cards go to %s' % CARD_DIR)
        r = s.request('card_save', 'POST', 'maker/card/save', {'path': str(card), 'thumbnail': str(a.thumbnail)})
        s.log('card_saved', {'path': str(card), 'sha256': sha(card), 'response': r})
        print(json.dumps(r, ensure_ascii=False), sha(card))
    elif a.step == 'renderstyle':
        body = json.loads(a.body.read_text(encoding='utf-8'))
        r = s.request('renderstyle_capture', 'POST', 'maker/renderstyle/capture', body, timeout=600)
        print(json.dumps(r, ensure_ascii=False)[:2000])


if __name__ == '__main__':
    main()
