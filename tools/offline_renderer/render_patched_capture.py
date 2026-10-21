"""Patch -> capture -> official full-driver replay -> images, with full receipts.

Single entry point for rendering a recorded HS2 D3D11 frame with explicitly
witnessed byte patches at their real upload points (CreateBuffer InitialData,
Unmap MapWrittenData, UpdateSubresource contents, ...). All real shaders, SSS,
GrabPass, shared depth and postprocess run unchanged inside the official
RenderDoc 1.46 D3D11 replay. WHICH bytes to patch is decided elsewhere (frame
graph / skinning payload work); this module only transfers and executes them.

Routes
  fast      (default) persistent `renderdoc_replay_msvc.exe session`: original
            compressed RDC held in memory; each job verifies the exact original
            ranges, re-encodes only the touched LZ4 pages (+ the following page
            whose history changed) with the pinned v1.46 LZ4 source, writes a
            complete standard RDC and runs official OpenFile/OpenCapture/replay.
  official  reference route: capture_mutation.mutate_capture on the official
            zip.xml export -> official zip.xml->rdc Convert -> official replay.
  both      runs both and records byte equivalence (frame section + targets).

Patch plan: the existing capture_mutation plan format (one plan file serves
both routes). Fields per patch: chunk_index, chunk_name, buffer_ordinal,
expected_buffer_index, expected_blob_sha256, byte_offset, expected_range_sha256,
replacement_file (relative to the plan), replacement_sha256, source_witness,
semantic_role, optional resource_id. Plan-level: source_rdc_sha256 (fast) and
source_xml_sha256/source_zip_sha256 (official).

Output orientation: the official read-back is D3D subresource memory order
(row 0 first; RenderDoc D3D11Replay::GetTextureData -> MapIntercept::CopyFromD3D,
no flip). Unity renders into RenderTextures with a flipped projection on D3D so
memory row 0 is the visual bottom; ReadPixels+EncodeToPNG writes the visual top
first. `orientation: "unity_texture_rows_bottom_up"` therefore reverses rows to
produce top-down RGBA matching the native PNG (accepted exactly for target
59322 of the 2026-10-12 capture). `"d3d_rows_top_down"` keeps memory order
(e.g. a swapchain back buffer). The raw read-back is always kept.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from logged_process import run_logged  # noqa: E402

JOB_SCHEMA = 'hs2_patched_capture_job/1'
ORIENTATIONS = ('unity_texture_rows_bottom_up', 'd3d_rows_top_down')
PLAN_PATCH_KEYS = ('chunk_index', 'chunk_name', 'buffer_ordinal', 'expected_buffer_index', 'expected_blob_sha256',
                   'byte_offset', 'expected_range_sha256', 'replacement_file', 'replacement_sha256',
                   'source_witness', 'semantic_role')
# ResourceFormatType values (pinned replay_enums.h): Regular=0, Undefined=1, BC1..BC7=2..8, ETC2=9, EAC=10, ASTC=11,
# packed 12..21, YUV8..YUV16=22..25, PVRTC=26, A8=27. Row reversal needs whole texel rows.
NON_ROW_FORMATS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 22, 23, 24, 25, 26}
PNG_COMP_TYPES = {0: 'Typeless', 2: 'UNorm', 9: 'UNormSRGB'}
NAME_RE = re.compile(r'^[A-Za-z0-9_.-]{1,64}$')


class PipelineRejected(ValueError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 24), b''):
            h.update(block)
    return h.hexdigest()


class Tools:
    """Explicit pinned binaries. GNU-built clients are never accepted (MSVC build receipt required)."""

    def __init__(self, exe, renderdoc_dir):
        self.exe = Path(exe).resolve()
        self.renderdoc_dir = Path(renderdoc_dir).resolve()
        dll = self.renderdoc_dir / 'renderdoc.dll'
        receipt = self.exe.parent / 'build_receipt.json'
        if not self.exe.is_file() or not dll.is_file() or not receipt.is_file():
            raise PipelineRejected('MSVC replay tool with build_receipt.json and pinned renderdoc.dll required')
        self.exe_sha256 = sha256_file(self.exe)
        built = {o['path']: o['sha256'] for o in json.loads(receipt.read_text())['outputs']}
        if built.get(str(self.exe)) != self.exe_sha256:
            raise PipelineRejected('Replay tool differs from its MSVC build receipt')
        self.dll = dll
        self.dll_sha256 = sha256_file(dll)
        self.build_receipt = str(receipt)
        self.env = dict(os.environ)
        self.env['PATH'] = str(self.renderdoc_dir) + os.pathsep + self.env.get('PATH', '')

    def describe(self):
        return {'exe': str(self.exe), 'exe_sha256': self.exe_sha256, 'build_receipt': self.build_receipt,
                'renderdoc_dll': str(self.dll), 'renderdoc_dll_sha256': self.dll_sha256,
                'pinned_commit': 'e4bd23b671d3d5a747ff5221dbe08a63eb6ca200 (checked by the tool before any replay call)'}

    def run(self, args, log_dir, timeout):
        started = time.perf_counter()
        result = run_logged([self.exe, *map(str, args)], log_dir, timeout=timeout, env=self.env)
        result['wall_seconds'] = time.perf_counter() - started
        if result['status'] != 'pass':
            raise PipelineRejected(f'{args[0]} failed ({result["status"]}); logs retained in {log_dir}')
        return result

    def json_lines(self, result):
        return [json.loads(l) for l in Path(result['stdout']).read_text(encoding='utf-8').splitlines() if l.startswith('{')]


def build_index(tools: Tools, rdc, index_path, log_dir, timeout=900):
    """Official structured-data walk -> exact section offsets of every recorded byte buffer."""
    index_path = Path(index_path)
    if index_path.exists():
        raise PipelineRejected('Fresh index path required')
    result = tools.run(['index', Path(rdc).resolve(), index_path.resolve()], log_dir, timeout)
    return {'index': str(index_path), 'index_sha256': sha256_file(index_path), 'process': result}


def load_index(index_path, rdc_sha256):
    index = json.loads(Path(index_path).read_text(encoding='utf-8'))
    if index.get('schema_version') != 1 or index.get('rdc_sha256') != rdc_sha256:
        raise PipelineRejected('Index does not belong to this exact RDC')
    if len(index['chunks']) != index['chunk_count']:
        raise PipelineRejected('Truncated index')
    return index


def resolve_plan(plan_path, index, rdc_sha256):
    """Bind capture_mutation-format patches to exact recorded section offsets. Never guesses a target."""
    if plan_path is None:
        return [], None
    plan_path = Path(plan_path).resolve()
    raw = plan_path.read_bytes()
    plan = json.loads(raw.decode('utf-8-sig'))
    if plan.get('source_rdc_sha256') != rdc_sha256:
        raise PipelineRejected('Plan source_rdc_sha256 does not bind this capture')
    patches = plan.get('patches')
    if not isinstance(patches, list) or not patches:
        raise PipelineRejected('Explicit source-derived byte patches required (omit the plan for identity)')
    resolved, spans = [], []
    for n, patch in enumerate(patches):
        missing = [k for k in PLAN_PATCH_KEYS if k not in patch]
        if missing:
            raise PipelineRejected(f'patch {n}: missing {missing}')
        if not patch['source_witness'] or not patch['semantic_role']:
            raise PipelineRejected(f'patch {n}: source identity and semantic role required; never guessed')
        ci, ordinal, offset = patch['chunk_index'], patch['buffer_ordinal'], patch['byte_offset']
        if not all(isinstance(v, int) and not isinstance(v, bool) for v in (ci, ordinal, offset, patch['expected_buffer_index'])):
            raise PipelineRejected(f'patch {n}: integer selectors required')
        if not 0 <= ci < len(index['chunks']):
            raise PipelineRejected(f'patch {n}: chunk index out of range')
        chunk = index['chunks'][ci]
        if chunk['chunk_index'] != ci or chunk['name'] != patch['chunk_name']:
            raise PipelineRejected(f'patch {n}: original command selector mismatch')
        if not 0 <= ordinal < len(chunk['buffers']):
            raise PipelineRejected(f'patch {n}: buffer occurrence selector out of bounds')
        buf = chunk['buffers'][ordinal]
        if buf['sd_buffer_index'] != patch['expected_buffer_index'] or buf['sha256'] != patch['expected_blob_sha256']:
            raise PipelineRejected(f'patch {n}: original selected blob mismatch')
        if 'resource_id' in patch:
            ids = [r['id'] for r in chunk['resources'] if r['name'] != 'Context']
            if patch['resource_id'] not in ids:
                raise PipelineRejected(f'patch {n}: resource_id {patch["resource_id"]} is not a resource of chunk {ci}')
        replacement = (plan_path.parent / patch['replacement_file']).resolve()
        data = replacement.read_bytes()
        if sha256_bytes(data) != patch['replacement_sha256']:
            raise PipelineRejected(f'patch {n}: candidate source bytes changed')
        if offset < 0 or not data or offset + len(data) > buf['bytes']:
            raise PipelineRejected(f'patch {n}: exact source byte patch range invalid')
        if not re.fullmatch(r'[0-9a-f]{64}', patch['expected_range_sha256']):
            raise PipelineRejected(f'patch {n}: expected_range_sha256 must be a SHA-256 hex digest')
        start = buf['data_offset'] + offset
        end = start + len(data)
        if any(start < e and end > s for s, e in spans):
            raise PipelineRejected(f'patch {n}: overlapping patches')
        spans.append((start, end))
        resolved.append({'patch_index': n, 'chunk_index': ci, 'chunk_name': chunk['name'], 'buffer_ordinal': ordinal,
                         'sd_buffer_index': buf['sd_buffer_index'], 'resources': chunk['resources'],
                         'size_offset': buf['size_offset'], 'data_offset': buf['data_offset'], 'blob_bytes': buf['bytes'],
                         'expected_blob_sha256': buf['sha256'], 'byte_offset': offset, 'replacement_bytes': len(data),
                         'expected_range_sha256': patch['expected_range_sha256'], 'replacement_file': str(replacement),
                         'replacement_sha256': patch['replacement_sha256'], 'source_witness': patch['source_witness'],
                         'semantic_role': patch['semantic_role'], 'section_range': [start, end],
                         'opaque_chunk_payload': buf['opaque']})
    return resolved, {'path': str(plan_path), 'sha256': sha256_bytes(raw)}


def validate_targets(targets):
    if not isinstance(targets, list) or not targets:
        raise PipelineRejected('At least one explicit read-back target required')
    names, out = set(), []
    for t in targets:
        name = t.get('name', '')
        if not NAME_RE.match(name) or name in names:
            raise PipelineRejected(f'Unique target name [A-Za-z0-9_.-] required: {name!r}')
        names.add(name)
        kind = t.get('kind')
        for key in ('event_id', 'resource_id'):
            if not isinstance(t.get(key), int) or isinstance(t.get(key), bool) or t[key] <= 0:
                raise PipelineRejected(f'{name}: explicit positive {key} required (no default target)')
        row = {'name': name, 'kind': kind, 'event_id': t['event_id'], 'resource_id': t['resource_id']}
        if kind == 'texture':
            if t.get('orientation') not in ORIENTATIONS:
                raise PipelineRejected(f'{name}: orientation must be one of {ORIENTATIONS}')
            row.update(orientation=t['orientation'], mip=int(t.get('mip', 0)), slice=int(t.get('slice', 0)), png=bool(t.get('png', True)))
        elif kind == 'buffer':
            row.update(byte_offset=int(t.get('byte_offset', 0)), byte_length=int(t.get('byte_length', 0)))
            if row['byte_offset'] < 0 or row['byte_length'] < 0:
                raise PipelineRejected(f'{name}: negative buffer range')
        else:
            raise PipelineRejected(f'{name}: kind must be texture or buffer')
        out.append(row)
    return out


def orient_texture(raw: bytes, info: dict, orientation: str):
    """Return (top-down bytes or None, note). Exact row permutation only; no colour/format conversion."""
    if info['dimension'] != 2:
        return None, 'not a 2D texture; memory order retained only'
    if info['format_type'] in NON_ROW_FORMATS:
        return None, 'block/planar format; row reversal undefined'
    h, w = info['height'], info['width']
    if h <= 0 or len(raw) % h:
        raise PipelineRejected('Read-back size is not whole rows')
    row = len(raw) // h
    if row < w:
        raise PipelineRejected('Read-back row smaller than width')
    if orientation == 'd3d_rows_top_down':
        return raw, 'memory order is visual top-down for this target'
    import numpy as np  # exact byte-row permutation (no dtype interpretation)
    flipped = np.frombuffer(raw, dtype=np.uint8).reshape(h, row)[::-1].tobytes()
    return flipped, 'rows reversed: Unity RenderTexture memory row 0 is the visual bottom'


def png_eligible(info: dict) -> bool:
    return (info['format_type'] == 0 and info['component_bytes'] == 1 and info['component_count'] == 4
            and info['component_type'] in PNG_COMP_TYPES)


def write_png(topdown: bytes, info: dict, path: Path):
    from PIL import Image  # PNG container only; pixels are the exact read-back bytes
    data = topdown
    if info['bgra_order']:
        data = bytearray(topdown)
        data[0::4], data[2::4] = topdown[2::4], topdown[0::4]
        data = bytes(data)
    Image.frombytes('RGBA', (info['width'], info['height']), data).save(path, format='PNG')


def _postprocess(targets, result_targets, out_dir: Path):
    rows = []
    by_name = {t['name']: t for t in result_targets}
    for t in targets:
        r = dict(by_name[t['name']])
        raw_path = Path(r['raw'])
        raw = raw_path.read_bytes()
        if sha256_bytes(raw) != r['sha256']:
            raise PipelineRejected(f'{t["name"]}: raw read-back changed after the tool wrote it')
        row = {'name': t['name'], 'kind': t['kind'], 'event_id': t['event_id'], 'resource_id': t['resource_id'],
               'raw': {'path': str(raw_path), 'bytes': len(raw), 'sha256': r['sha256'],
                       'convention': 'official read-back, D3D subresource memory order'}}
        if t['kind'] == 'texture':
            info = {k: r[k] for k in ('width', 'height', 'depth', 'dimension', 'array_size', 'mips', 'format_type', 'component_type',
                                      'component_count', 'component_bytes', 'bgra_order', 'srgb_corrected', 'mip', 'slice')}
            row['texture'] = info
            row['orientation'] = t['orientation']
            topdown, note = orient_texture(raw, info, t['orientation'])
            row['orientation_note'] = note
            if topdown is not None:
                p = out_dir / f'{t["name"]}.topdown.bin'
                p.write_bytes(topdown)
                row['topdown'] = {'path': str(p), 'bytes': len(topdown), 'sha256': sha256_bytes(topdown)}
                if t['png'] and png_eligible(info):
                    pp = out_dir / f'{t["name"]}.png'
                    write_png(topdown, info, pp)
                    row['png'] = {'path': str(pp), 'sha256': sha256_file(pp), 'pixels': 'top-down RGBA8, exact read-back bytes'}
                elif t['png']:
                    row['png'] = None
                    row['png_note'] = 'not an 8-bit RGBA UNorm/Typeless/sRGB target; raw/top-down bytes only'
        else:
            row['buffer'] = {k: r[k] for k in ('buffer_length', 'offset', 'requested_length')}
        rows.append(row)
    return rows


class ReplaySession:
    """One owned `session` process per capture; renders many patch sets. Only its own child is ever stopped."""

    def __init__(self, tools: Tools, rdc, index_path, session_dir, startup_timeout=600):
        self.tools = tools
        self.rdc = Path(rdc).resolve()
        self.session_dir = Path(session_dir).resolve()
        if self.session_dir.exists():
            raise PipelineRejected('Fresh session directory required')
        self.session_dir.mkdir(parents=True)
        t = time.perf_counter()
        self.rdc_sha256 = sha256_file(self.rdc)
        self.index_path = Path(index_path).resolve()
        self.index_sha256 = sha256_file(self.index_path)
        self.index = load_index(self.index_path, self.rdc_sha256)
        self.python_startup = {'rdc_sha256_and_index_load_s': time.perf_counter() - t}
        self.stdout_log = (self.session_dir / 'stdout.jsonl').open('w', encoding='utf-8')
        self.stderr_file = (self.session_dir / 'stderr.bin').open('wb')
        self.stdin_log = (self.session_dir / 'stdin.tsv').open('w', encoding='utf-8')
        self.proc = subprocess.Popen([str(tools.exe), 'session', str(self.rdc), self.rdc_sha256, self.index['frame_section_sha256']],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr_file, env=tools.env,
                                     creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.jobs = 0
        self.closed = False
        try:
            self.tool_line = self._next({'tool'}, startup_timeout)
            self.ready = self._next({'ready'}, startup_timeout)
        except Exception:
            self.close(abort=True)
            raise

    def _read(self):
        for raw in self.proc.stdout:
            line = raw.decode('utf-8', 'replace').rstrip('\r\n')
            self.stdout_log.write(line + '\n')
            self.stdout_log.flush()
            self.lines.put(line)
        self.lines.put(None)

    def _next(self, events, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close(abort=True)
                raise PipelineRejected(f'Session deadline exceeded waiting for {sorted(events)}; owned process stopped')
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                raise PipelineRejected(f'Session process ended (returncode {self.proc.poll()}); see {self.session_dir}')
            if not line.startswith('{'):
                continue
            msg = json.loads(line)
            if msg.get('event') in events:
                return msg

    def _send(self, *fields):
        line = '\t'.join(str(f) for f in fields)
        if any(c in line for c in '\r\n') or any('\t' in str(f) for f in fields):
            raise PipelineRejected('Field contains a control separator')
        self.stdin_log.write(line + '\n')
        self.stdin_log.flush()
        self.proc.stdin.write((line + '\n').encode('utf-8'))

    def render(self, plan_path, targets, out_dir, *, keep_candidate=False, timeout=900):
        """Render one patch set (None = identity). Returns the full receipt (also written to out_dir/receipt.json)."""
        t0 = time.perf_counter()
        out_dir = Path(out_dir).resolve()
        if out_dir.exists():
            raise PipelineRejected('Fresh output directory required')
        targets = validate_targets(targets)
        patches, plan = resolve_plan(plan_path, self.index, self.rdc_sha256)
        out_dir.mkdir(parents=True)
        t_resolve = time.perf_counter() - t0
        self.jobs += 1
        job_id = f'job{self.jobs:05d}'
        candidate = out_dir / 'candidate.rdc'
        self._send('job', job_id, candidate, 1 if keep_candidate else 0)
        for p in patches:
            self._send('patch', '-' if p['opaque_chunk_payload'] else p['size_offset'], p['data_offset'], p['blob_bytes'],
                       p['expected_blob_sha256'], p['byte_offset'], p['expected_range_sha256'], p['replacement_file'],
                       p['replacement_sha256'], 'end')
        for t in targets:
            raw = out_dir / f'{t["name"]}.raw.bin'
            if t['kind'] == 'texture':
                self._send('texture', t['name'], t['event_id'], t['resource_id'], t['mip'], t['slice'], raw)
            else:
                self._send('buffer', t['name'], t['event_id'], t['resource_id'], t['byte_offset'], t['byte_length'], raw)
        t_send = time.perf_counter()
        self._send('run')
        self.proc.stdin.flush()
        msg = self._next({'job_result', 'job_rejected'}, timeout)
        t_replay = time.perf_counter() - t_send
        if msg['event'] == 'job_rejected':
            receipt = {'schema': JOB_SCHEMA, 'status': 'rejected', 'error': msg['error'], 'job': job_id,
                       'plan': plan, 'patches': patches, 'outputs_written': sorted(p.name for p in out_dir.iterdir())}
            (out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
            raise PipelineRejected(msg['error'])
        t_post = time.perf_counter()
        rows = _postprocess(targets, msg['targets'], out_dir)
        receipt = {
            'schema': JOB_SCHEMA, 'status': 'pass', 'route': 'fast_lz4_page_splice', 'job': job_id,
            'scope': 'Official RenderDoc 1.46 full D3D11 frame replay of the recorded capture with explicit byte patches; '
                     'patch selection/dependency coverage is the caller\'s responsibility and not accepted here.',
            'tools': self.tools.describe(),
            'capture': {'rdc': str(self.rdc), 'rdc_sha256': self.rdc_sha256, 'frame_section_sha256': self.index['frame_section_sha256']},
            'index': {'path': str(self.index_path), 'sha256': self.index_sha256},
            'plan': plan, 'patches': patches,
            'candidate': {'rdc': msg['replayed_rdc'], 'written': msg['candidate_written'], 'sha256': msg['candidate_rdc_sha256'],
                          'bytes': msg['candidate_rdc_bytes'], 'retained': msg['candidate_retained'],
                          'changed_lz4_pages': msg['changed_pages'], 'reencoded_lz4_pages': msg['reencoded_pages']},
            'targets': rows,
            'timing': {'tool_ms': msg['timing_ms'], 'python_resolve_s': t_resolve, 'python_roundtrip_s': t_replay,
                       'python_postprocess_s': time.perf_counter() - t_post, 'python_total_s': time.perf_counter() - t0},
            'session': {'dir': str(self.session_dir), 'pid': self.proc.pid, 'ready': self.ready},
            'candidate_geometry_complete': False, 'full_renderer_accepted': False,
        }
        (out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        return receipt

    def close(self, abort=False, timeout=60):
        if self.closed:
            return
        self.closed = True
        try:
            if not abort and self.proc.poll() is None:
                self._send('quit')
                self.proc.stdin.flush()
                self.proc.stdin.close()
                self.proc.wait(timeout=timeout)
        except Exception:
            abort = True
        if self.proc.poll() is None:
            self.proc.kill()  # owned child only
            self.proc.wait()
        self.reader.join(timeout=10)
        for pipe in (self.proc.stdin, self.proc.stdout):
            try:
                pipe.close()
            except Exception:
                pass
        for f in (self.stdout_log, self.stderr_file, self.stdin_log):
            f.close()
        (self.session_dir / 'receipt.json').write_text(json.dumps(
            {'pid': self.proc.pid, 'returncode': self.proc.returncode, 'aborted': abort, 'jobs': self.jobs,
             'rdc': str(self.rdc), 'rdc_sha256': self.rdc_sha256, 'index': str(self.index_path),
             'python_startup': self.python_startup, 'tools': self.tools.describe()}, indent=2), encoding='utf-8')

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close(abort=exc[0] is not None)


def frame_section_sha(tools: Tools, rdc, log_dir, timeout=600):
    res = tools.run(['sections', Path(rdc).resolve()], log_dir, timeout)
    rows = [r for r in tools.json_lines(res) if r.get('event') == 'section' and r['type'] == 1]
    if len(rows) != 1:
        raise PipelineRejected('Exactly one frame capture section expected')
    return rows[0], res


def official_route(tools: Tools, rdc, zip_xml, plan_path, targets, out_dir, timeout=1800):
    """Reference route through official zip.xml import/convert. Slow; used for equivalence proofs."""
    from src.render.capture_mutation import mutate_capture
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise PipelineRejected('Fresh official-route directory required')
    out_dir.mkdir(parents=True)
    timing = {}
    rdc_sha = sha256_file(rdc)
    t = time.perf_counter()
    if plan_path is not None:
        candidate_xml = out_dir / 'official_candidate.zip.xml'
        mutation = mutate_capture(Path(zip_xml), Path(plan_path), candidate_xml)
    else:
        candidate_xml, mutation = Path(zip_xml).resolve(), None
    timing['mutate_capture_s'] = time.perf_counter() - t
    candidate_rdc = out_dir / 'official_candidate.rdc'
    conv = tools.run(['convert', candidate_xml, 'zip.xml', candidate_rdc, 'rdc'], out_dir / 'log_convert', timeout)
    timing['official_convert_s'] = conv['wall_seconds']
    section, sec_proc = frame_section_sha(tools, candidate_rdc, out_dir / 'log_sections', timeout)
    timing['official_sections_s'] = sec_proc['wall_seconds']
    t = time.perf_counter()
    cand_sha = sha256_file(candidate_rdc)
    # replay-only session over the official zstd output, indexed by its own official section hash
    fake_index = out_dir / 'official_candidate.index.json'
    fake_index.write_text(json.dumps({'schema_version': 1, 'rdc_sha256': cand_sha, 'frame_section_sha256': section['contents_sha256'],
                                      'chunk_count': 0, 'chunks': [], 'scope': 'replay-only binding of the official converted candidate'}))
    with ReplaySession(tools, candidate_rdc, fake_index, out_dir / 'session') as s:
        receipt = s.render(None, targets, out_dir / 'replay', timeout=timeout)
    timing['official_replay_s'] = time.perf_counter() - t
    result = {'route': 'official_zip_xml_convert', 'source_rdc_sha256': rdc_sha, 'zip_xml': str(zip_xml), 'mutation': mutation,
              'candidate_rdc': str(candidate_rdc), 'candidate_rdc_sha256': cand_sha, 'frame_section': section,
              'replay_receipt': str(out_dir / 'replay' / 'receipt.json'), 'targets': receipt['targets'], 'timing': timing,
              'tools': tools.describe()}
    (out_dir / 'official_route.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def load_job(path):
    job = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if job.get('schema') != JOB_SCHEMA:
        raise PipelineRejected(f'Job schema {JOB_SCHEMA} required')
    for key in ('capture_rdc', 'capture_rdc_sha256', 'index', 'targets'):
        if key not in job:
            raise PipelineRejected(f'Job missing {key}')
    return job


def run_job(tools: Tools, job, out_dir, route='fast'):
    out_dir = Path(out_dir).resolve()
    if out_dir.exists():
        raise PipelineRejected('Fresh output directory required')
    out_dir.mkdir(parents=True)
    rdc = Path(job['capture_rdc']).resolve()
    plan = job.get('plan')
    summary = {'schema': JOB_SCHEMA, 'route': route, 'job': job}
    if route in ('fast', 'both'):
        with ReplaySession(tools, rdc, job['index'], out_dir / 'fast_session') as s:
            if s.rdc_sha256 != job['capture_rdc_sha256']:
                raise PipelineRejected('capture_rdc_sha256 mismatch')
            fast = s.render(plan, job['targets'], out_dir / 'fast', keep_candidate=(route == 'both') or bool(job.get('keep_candidate_rdc')))
        summary['fast'] = str(out_dir / 'fast' / 'receipt.json')
    if route in ('official', 'both'):
        if 'official_zip_xml' not in job:
            raise PipelineRejected('official_zip_xml required for the official route')
        official = official_route(tools, rdc, job['official_zip_xml'], plan, job['targets'], out_dir / 'official')
        summary['official'] = str(out_dir / 'official' / 'official_route.json')
    if route == 'both':
        fast_section, _ = (frame_section_sha(tools, fast['candidate']['rdc'], out_dir / 'log_fast_sections')
                           if fast['candidate']['written'] else (
                               {'contents_sha256': fast['capture']['frame_section_sha256']}, None))
        fast_t = {r['name']: r['raw']['sha256'] for r in fast['targets']}
        off_t = {r['name']: r['raw']['sha256'] for r in official['targets']}
        eq = {'frame_section_fast': fast_section['contents_sha256'], 'frame_section_official': official['frame_section']['contents_sha256'],
              'frame_section_equal': fast_section['contents_sha256'] == official['frame_section']['contents_sha256'],
              'targets': {n: {'fast': fast_t[n], 'official': off_t[n], 'equal': fast_t[n] == off_t[n]} for n in fast_t},
              }
        eq['all_equal'] = eq['frame_section_equal'] and all(v['equal'] for v in eq['targets'].values())
        summary['equivalence'] = eq
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tool', required=True, type=Path, help='MSVC renderdoc_replay_msvc.exe with build_receipt.json')
    ap.add_argument('--renderdoc-dir', required=True, type=Path, help='pinned portable RenderDoc 1.46 directory')
    sub = ap.add_subparsers(dest='command', required=True)
    ix = sub.add_parser('index', help='build the official offset index of an RDC (once per capture)')
    ix.add_argument('--rdc', required=True, type=Path)
    ix.add_argument('--out', required=True, type=Path, help='fresh index JSON path')
    rn = sub.add_parser('render', help='render one job spec')
    rn.add_argument('--job', required=True, type=Path)
    rn.add_argument('--out', required=True, type=Path)
    rn.add_argument('--route', choices=('fast', 'official', 'both'), default='fast')
    a = ap.parse_args()
    tools = Tools(a.tool, a.renderdoc_dir)
    if a.command == 'index':
        print(json.dumps(build_index(tools, a.rdc, a.out, Path(str(a.out) + '.log')), indent=2))
    else:
        print(json.dumps(run_job(tools, load_job(a.job), a.out, a.route), indent=2))


if __name__ == '__main__':
    main()
