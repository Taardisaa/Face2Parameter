"""Installed shader object of each scene material, from the renderer's own asset bundle.

A shader name can have several installed copies (one per bundle that carries it, e.g.
``AIT/Clothes True`` in every clothes bundle, ``AIT/Skin Translucency simple`` in each head
bundle), and the copies differ in their stripped programs and constant-buffer layouts.  The
copy a material uses is the object its serialized ``m_Shader`` PPtr references; HS2 materials
are instantiated from the prefab's materials ("(Instance)"), which keeps the PPtr target.

Chain (installed assets only, read with UnityPy):

1. the resource acceptance pairs the renderer's mesh streams byte-exactly with a mesh of an
   installed bundle and records the prefab component that references it
   (``provenance.prefab_binding[].component_path_id``, ``provenance.source``);
2. that component's ``m_Materials[slot]`` PPtr -> the material object (its ``m_Name`` must equal
   the scene material name without the " (Instance)" suffixes);
3. the material's ``m_Shader`` PPtr -> (file, path id): file 0 is the same serialized file;
   another file is named by ``m_Externals`` (``archive:/CAB-.../CAB-...``) and is located
   among the shader-library sources by its CAB name.

The output (``resolve`` -> JSON) maps scene material id -> {shader, source spec, path id,
evidence}; ``SceneAdapter.shader_copies`` and the orchestrator's pass-program selection
then use exactly that copy.

    python -m src.render.hs2pipe.material_sources --scene SCENE.json --acceptance-table T.json \
        --library-source-index LIB.json [...] --out material_sources.json
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

_ENV = {}


def instance_base(name):
    while name.endswith(' (Instance)'):
        name = name[:-len(' (Instance)')]
    return name


def load_env(spec):
    """UnityPy environment of ``PATH`` or ``ZIP::MEMBER`` (cached)."""
    import UnityPy
    if spec not in _ENV:
        if '::' in spec:
            archive, member = spec.split('::', 1)
            with zipfile.ZipFile(archive) as z:
                data = z.read(member)
            _ENV[spec] = UnityPy.load(io.BytesIO(data))
        else:
            _ENV[spec] = UnityPy.load(spec)
    return _ENV[spec]


def objects_by_path_id(env):
    out = {}
    for obj in env.objects:
        out.setdefault(obj.assets_file.name, {})[obj.path_id] = obj
    return out


def cab_names(spec):
    return {o.assets_file.name.lower() for o in load_env(spec).objects}


HAIR_SHADER_BUNDLE = 'abdata/chara/hair_shader_mat.unity3d'      # ChaABDefine.HairShaderMatAssetBundle
HAIR_SHADER_MATERIAL = {0: 'hair_dithering', 1: 'hair_cutout'}   # Manager.Character (Character.cs 258..261)


def hair_renderer_ids(env):
    """Component path ids listed by the CmpHair.rendHair arrays of a hair bundle."""
    out = set()
    for o in env.objects:
        if o.type.name != 'MonoBehaviour':
            continue
        try:
            t = o.read_typetree()
        except Exception:                      # noqa: BLE001 - MonoBehaviours without a type tree
            continue
        for pp in t.get('rendHair') or []:
            if pp.get('m_FileID') == 0:
                out.add(int(pp['m_PathID']))
    return out


def hair_shader_override(game_root, shader_type):
    """ChaControl.ChangeSettingHairShader: every CmpHair.rendHair material gets
    Character.shaderDithering (shaderType 0) or shaderCutout, the shaders of the materials
    hair_dithering / hair_cutout of chara/hair_shader_mat.unity3d."""
    spec = str(Path(game_root) / HAIR_SHADER_BUNDLE)
    name = HAIR_SHADER_MATERIAL[int(shader_type)]
    for o in load_env(spec).objects:
        if o.type.name == 'Material':
            t = o.read_typetree()
            if t['m_Name'] == name and t['m_Shader']['m_FileID'] == 0:
                return {'shader_source': spec, 'shader_path_id': int(t['m_Shader']['m_PathID']),
                        'shader_file': o.assets_file.name,
                        'shader_set_by': 'ChaControl.ChangeSettingHairShader (card hair.shaderType %d): '
                                         'Character.%s = %s.shader' % (int(shader_type),
                                                                       'shaderDithering' if int(shader_type) == 0
                                                                       else 'shaderCutout', name)}
    raise KeyError('%s has no material %s' % (spec, name))


def resolve(scene, acceptance_rows, library_sources=(), hair_shader_type=None, game_root='E:/HoneySelect2_ArcticFox'):
    """{scene material id: record} for every enabled renderer the acceptance placed in a bundle.

    ``hair_shader_type``: the card's ChaFileHair.shaderType; renderers listed by a CmpHair.rendHair
    of their bundle then use the hair shader of :func:`hair_shader_override`."""
    mats = {m['id']: m for m in scene['materials']}
    prov_of = {}
    for r in acceptance_rows:
        rq = r.get('request') if isinstance(r.get('request'), dict) else {}
        p = ((r.get('entry') or {}).get('provenance') or {})
        if 'scene_renderer_id' in rq and p.get('prefab_binding') and p.get('source') and r.get('status') == 'byte_exact':
            prov_of.setdefault(int(rq['scene_renderer_id']), p)
    cab_to_spec = None
    out = {}
    for ren in scene['renderers']:
        p = prov_of.get(int(ren['id']))
        if p is None:
            continue
        spec = p['source']
        env = load_env(spec)
        by_file = objects_by_path_id(env)
        comp = None
        for b in p['prefab_binding']:
            for fname, objs in by_file.items():
                if b['component_path_id'] in objs:
                    comp = (fname, objs[b['component_path_id']], b)
                    break
            if comp:
                break
        if comp is None:
            continue
        fname, cobj, binding = comp
        tree = cobj.read_typetree()
        hair = (hair_shader_type is not None and int(binding['component_path_id']) in hair_renderer_ids(env))
        refs = [m for m in (ren['properties'].get('sharedMaterials') or [])]
        base_spec, base_by_file, base_fname = spec, by_file, fname
        for slot, (pptr, ref) in enumerate(zip(tree.get('m_Materials') or [], refs)):
            spec, by_file, fname = base_spec, base_by_file, base_fname
            if ref is None or ref.get('ref_id') not in mats:
                continue
            m = mats[ref['ref_id']]
            rec = {'renderer': ren['name'], 'slot': slot, 'bundle': spec, 'component_path_id': binding['component_path_id'],
                   'component_path': binding.get('path')}
            if pptr['m_FileID'] != 0:
                rec['unresolved'] = 'material PPtr in another file (m_FileID %d)' % pptr['m_FileID']
                out[m['id']] = rec
                continue
            mobj = by_file[fname].get(pptr['m_PathID'])
            mt = mobj.read_typetree()
            rec['material_asset'] = mt['m_Name']
            if mt['m_Name'] != instance_base(m['name']):
                # the game replaced the prefab material at runtime: locate the scene material's asset
                # by name among the renderer's bundle and the library sources; accepted only when
                # exactly one material object carries the name (labelled 'by unique name')
                hits = []
                for s in [base_spec] + [x for x in library_sources if x != base_spec]:
                    for o in load_env(s).objects:
                        if o.type.name == 'Material' and o.read_typetree()['m_Name'] == instance_base(m['name']):
                            hits.append((s, o))
                own = [h for h in hits if h[0] == base_spec]
                if len(own) == 1:
                    # e.g. the face material: ChaControl loads ListInfo MatData from the head's MainAB
                    # (InitBaseCustomTextureFace(MainManifest, MainAB, MatData)), the bundle the head
                    # prefab itself came from
                    hits = own
                    located = "by name in the renderer's own bundle (list MatData from MainAB)"
                else:
                    located = 'by unique name (runtime replacement of the prefab material)'
                if len(hits) != 1:
                    rec['unresolved'] = ('prefab material %s is not the scene material %s; %d named assets'
                                         % (mt['m_Name'], m['name'], len(hits)))
                    out[m['id']] = rec
                    continue
                spec_m, mobj = hits[0]
                mt = mobj.read_typetree()
                fname = mobj.assets_file.name
                by_file = objects_by_path_id(load_env(spec_m))
                rec.update({'material_asset': mt['m_Name'], 'material_bundle': spec_m, 'material_located': located})
                spec = spec_m
            sp = mt['m_Shader']
            if sp['m_FileID'] == 0:
                rec.update({'shader_source': spec, 'shader_path_id': int(sp['m_PathID']), 'shader_file': fname})
            else:
                ext = mobj.assets_file.externals[sp['m_FileID'] - 1].path
                cab = ext.split('/')[-1].lower()
                if cab_to_spec is None:
                    cab_to_spec = {}
                    for s in library_sources:
                        for c in cab_names(s):
                            cab_to_spec.setdefault(c, s)
                if cab not in cab_to_spec:
                    rec['unresolved'] = 'shader in %s, not a shader-library source' % cab
                    out[m['id']] = rec
                    continue
                rec.update({'shader_source': cab_to_spec[cab], 'shader_path_id': int(sp['m_PathID']), 'shader_file': cab})
            if hair:
                rec['prefab_shader'] = {'shader_source': rec['shader_source'], 'shader_path_id': rec['shader_path_id']}
                rec.update(hair_shader_override(game_root, hair_shader_type))
            sh = (objects_by_path_id(load_env(rec['shader_source'])).get(rec['shader_file'], {})
                  .get(rec['shader_path_id']))
            if sh is not None:
                rec['shader_name'] = sh.read_typetree()['m_ParsedForm']['m_Name']
            out[m['id']] = rec
    return out


def copy_matches(copy, rec):
    """True when a shader-library copy is the object ``rec`` names (source spec + path id)."""
    if rec is None or 'shader_path_id' not in rec:
        return False
    return (int(copy.record.get('path_id')) == int(rec['shader_path_id'])
            and _norm(copy.source_label) == _norm(rec['shader_source']))


def _norm(spec):
    return str(spec).replace('/', '\\').lower()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scene', required=True)
    ap.add_argument('--acceptance-table', required=True)
    ap.add_argument('--library-source', action='append', default=[],
                    help='shader-library source spec (PATH or ZIP::MEMBER) for external shader files')
    ap.add_argument('--card', help='character card PNG (ChaFileHair.shaderType for the hair shader rule)')
    ap.add_argument('--game-root', default='E:/HoneySelect2_ArcticFox')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    scene = json.loads(Path(a.scene).read_text(encoding='utf-8'))
    rows = json.loads(Path(a.acceptance_table).read_text(encoding='utf-8'))
    hst = None
    if a.card:
        from src.utils.chara_loader.AiSyoujyoCharaData import AiSyoujyoCharaData
        hst = int(AiSyoujyoCharaData.load(a.card)['Custom']['hair']['shaderType'])
    out = resolve(scene, rows, a.library_source, hst, a.game_root)
    Path(a.out).write_text(json.dumps({'schema': 'hs2pipe.material_sources/1', 'scene': a.scene,
                                       'acceptance_table': a.acceptance_table, 'card': a.card,
                                       'hair_shader_type': hst, 'materials': out}, indent=1),
                           encoding='utf-8')
    print(json.dumps({'materials': len(out), 'unresolved': sum(1 for v in out.values() if 'unresolved' in v)}))


if __name__ == '__main__':
    main()
