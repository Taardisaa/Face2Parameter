"""Acquire one declared neutral Multiface tuple, using original camera/mesh readers.

The upstream image TAR is large. Only two same-frame photos are extracted.
Inputs/assets remain local; no reconstruction or game mutation is performed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.request import urlopen

import numpy as np
from PIL import Image

from .artifact import sha

ROOT = 'https://fb-baas-f32eacb9-8abb-11eb-b2b8-4857dd089e15.s3.amazonaws.com/MugsyDataRelease/v0.0/identities/'


def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + '.partial')
    if temporary.exists():
        raise FileExistsError('Incomplete prior download retained: ' + str(temporary))
    with urlopen(url, timeout=30) as response, temporary.open('xb') as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
    temporary.rename(path)


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def extract_selected(archive, names, out):
    found = set()
    with tarfile.open(archive, 'r:') as stream:
        for member in stream:
            name = member.name.removeprefix('./')
            if name not in names:
                continue
            if not member.isfile() or not 0 < member.size <= 16 * 1024 * 1024:
                raise ValueError('Selected dataset member has unsupported type/size')
            target = (out / name).resolve()
            if not target.is_relative_to(out.resolve()):
                raise ValueError('Unsafe dataset member path')
            data = stream.extractfile(member).read()
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Prior extracted input differs: ' + name)
            target.write_bytes(data)
            if name in found:
                raise ValueError('Duplicate selected dataset member')
            found.add(name)
    if found != names:
        raise ValueError('Missing associated dataset members: ' + str(names - found))


def upstream_readers(source):
    # Reuse the pinned source's literal readers, without importing unrelated CUDA
    # training dependencies or copying a guessed OBJ/KRT interpretation.
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'load_obj', 'load_krt'}]
    if {n.name for n in functions} != {'load_obj', 'load_krt'}:
        raise ValueError('Original Multiface readers missing')
    scope = {'np': np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), scope)
    return scope['load_obj'], scope['load_krt']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('multiface_neutral_protocol.json'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--download', action='store_true', help='Fetch official archives if absent, including ~3.1 GB image TAR')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['format'] != 'multiface_neutral_case_v1' or protocol['segment'] != 'E001_Neutral_Eyes_Open':
        raise ValueError('Require explicitly declared neutral case')
    if args.out.exists():
        raise FileExistsError('Preserve existing case; use a fresh output path')
    cache = args.cache.resolve(); cache.mkdir(parents=True, exist_ok=True)
    base = ROOT + protocol['subject'] + '/'
    names = ['metadata.tar', 'tracked_mesh--' + protocol['segment'] + '.tar', 'images--' + protocol['segment'] + '.tar']
    if args.download:
        for name in ['CHECKSUM', *names]:
            fetch(base + name, cache / name)
    sums = (cache / 'CHECKSUM').read_text().splitlines()
    archives = []
    for name in names:
        lines = [s for s in sums if s.strip().endswith(name)]
        if len(lines) != 1 or md5(cache / name) != lines[0].split()[0]:
            raise ValueError('Official dataset checksum differs: ' + name)
        archives.append({'name': name, 'url': base + name, 'bytes': (cache/name).stat().st_size,
                         'official_md5': lines[0].split()[0], 'sha256': sha(cache/name)})
    revision = protocol['upstream_revision']
    source_url = f'https://raw.githubusercontent.com/facebookresearch/multiface/{revision}/dataset.py'
    source = cache / ('dataset_' + revision + '.py')
    if args.download:
        fetch(source_url, source)
    expected_source = 'b465d02b618111c3cbc3826b514c138aaa14e17e698a3ff4802747e06edfadde'
    if sha(source) != expected_source:
        raise ValueError('Original Multiface dataset source changed')
    capture, segment, frame = (protocol[k] for k in ('capture', 'segment', 'frame'))
    mesh_prefix = f'{capture}/tracked_mesh/{segment}/{frame}'
    selections = [{f'{capture}/KRT', f'{capture}/frame_list.txt'},
                  {mesh_prefix + '.bin', mesh_prefix + '.obj', mesh_prefix + '_transform.txt'},
                  {f'{capture}/images/{segment}/{cam}/{frame}.png' for cam in protocol['cameras']}]
    for name, selected in zip(names, selections):
        extract_selected(cache/name, selected, cache)
    root = cache/capture; prefix = root/'tracked_mesh'/segment/frame
    load_obj, load_krt = upstream_readers(source)
    mesh = load_obj(str(prefix.with_suffix('.obj')))
    vertices = np.fromfile(prefix.with_suffix('.bin'), dtype=np.float32).reshape(-1, 3)
    faces = mesh['vert_ids']; pose = np.loadtxt(str(prefix)+'_transform.txt')
    if vertices.shape != (7306, 3) or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError('Unexpected tracked mesh topology')
    if pose.shape != (3, 4) or not np.isfinite(pose).all() or not np.allclose(pose[:, :3].T@pose[:, :3], np.eye(3), atol=1e-6) or np.linalg.det(pose[:, :3]) <= 0:
        raise ValueError('Unsupported head pose')
    world = vertices.astype(float) @ pose[:, :3].T + pose[:, 3]
    # OBJ is exported in capture coordinates; BIN is the original head-local input.
    # Check export precision, never use it to warp either surface.
    error = float(np.max(np.abs(world - mesh['verts'])))
    if error > 0.001:
        raise ValueError('Original OBJ/BIN/headpose correspondence differs')
    if f'{segment} {frame}' not in (root/'frame_list.txt').read_text().splitlines():
        raise ValueError('Selected frame missing from original frame list')
    all_cameras = load_krt(str(root/'KRT')); cameras = {}; input_rows = []
    args.out.mkdir(parents=True)
    inputs = args.out/'model_inputs'; inputs.mkdir()
    for cam in protocol['cameras']:
        camera = all_cameras[cam]; k, rt, dist = (camera[key] for key in ('intrin', 'extrin', 'dist'))
        if k.shape != (3,3) or rt.shape != (3,4) or dist.shape != (5,) or np.any(dist) or not np.isfinite(k).all() or not np.isfinite(rt).all() or not np.allclose(rt[:,:3].T@rt[:,:3], np.eye(3), atol=1e-6) or np.linalg.det(rt[:,:3]) <= 0:
            raise ValueError('Unsupported original camera branch')
        photo = root/'images'/segment/cam/(frame+'.png')
        with Image.open(photo) as image:
            width, height = image.size
        copied = inputs/(cam+'.png'); copied.write_bytes(photo.read_bytes())
        cameras[cam] = {'K':k.tolist(),'Rt':rt.tolist(),'distortion':dist.tolist(),'width':width,'height':height}
        input_rows.append({'camera':cam,'path':str(photo),'sha256':sha(photo),'model_input':str(copied.resolve()),'width':width,'height':height})
    geometry = args.out/'tracked_geometry.npz'
    np.savez_compressed(geometry, head_vertices=vertices, vertices=world, faces=faces, head_pose=pose)
    paths = [root/'KRT', root/'frame_list.txt', prefix.with_suffix('.bin'), prefix.with_suffix('.obj'), Path(str(prefix)+'_transform.txt')]
    case = {'format':'multiface_associated_neutral_v1','protocol':protocol,'protocol_sha256':sha(args.protocol),
            'upstream_source':{'url':source_url,'sha256':sha(source)},'archives':archives,
            'source_files':[{'path':str(p),'sha256':sha(p)} for p in paths], 'images':input_rows,'cameras':cameras,
            'geometry':{'path':str(geometry.resolve()),'sha256':sha(geometry),'vertices':len(vertices),'triangles':len(faces)},
            'headpose_obj_max_export_difference':error,'geometry_kind':'tracked_reconstruction_not_raw_scan',
            'source_accuracy_certified':False,'builder_sha256':sha(Path(__file__))}
    (args.out/'case.json').write_text(json.dumps(case,indent=2)+'\n')
    print(json.dumps({'case':str(args.out/'case.json'),'same_frame_images':len(input_rows),'raw_scan_truth':False}))


if __name__ == '__main__':
    main()
