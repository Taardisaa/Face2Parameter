"""Extract the HS2 head rig + appearance assets a specific CARD uses.

Every id on the card (headId / skinId / pupilId / ...) is resolved through the game's own asset
lists (src/hs2_assets.py::ChaList) instead of being guessed, so we extract exactly what the game
would load. See docs/hs2-renderer-and-mesh.md for the reverse-engineered pipeline.

Two traps this deliberately avoids (both were live bugs in the previous "pick the biggest o_head"
version):
  * a head bundle holds BOTH the render prefab `p_cf_head_NN` and a *collision* prefab
    `p_cf_head_NN_hit` — the hit meshes are bigger (up to 9811 verts) but have useless UVs;
  * headId does NOT index the bundle: headId=2 lives in chara/38/fo_head_38.unity3d and reuses
    `cf_anmShapeHead_01`. Only the list knows that.

Layout under data/hs2_head/ (gitignored — needs the game installed):
    head_<headId>/o_head_mesh.npz      verts/faces/normals/tangents/uv/uv1/colors/skin weights
    head_<headId>/skeleton.json        bone hierarchy + REST local transforms (FK source)
    head_<headId>/anmShapeHead.json    the head's shapeValueFace keyframe table
    head_<headId>/submeshes/*.npz      eyes / lashes / tear / eyeshadow / teeth / tongue
    head_<headId>/materials/*.json     real shading params (m_Colors + m_Floats)
    textures/<asset>.png               shared texture pool (in-bundle + card-selected)
    cards/<stem>.json                  this card's ids -> texture assets (renderer manifest)
    customhead.json                    slider(category) -> cf_s_ bone + DOF flags (head-agnostic)

Run with the project venv:
    .venv/Scripts/python.exe scripts/hs2_extract_head.py --card tests/HS2ChaF_20240901192905747.png
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys

import numpy as np
import UnityPy
from UnityPy.helpers.MeshHelper import MeshHandler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root on path

from src.hs2_assets import AB, ChaList, face_ids  # noqa: E402

CUSTOMSHAPE = os.path.join(AB, "list", "customshape.unity3d")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hs2_head")

# card id field -> (list category, column holding the texture asset name)
TEXTURE_IDS = [
    ("skinId", "ft_skin_f", "MainTex"),
    ("skinId", "ft_skin_f", "OcclusionMapTex"),
    ("skinId", "ft_skin_f", "NormalMapTex"),
    ("detailId", "ft_detail_f", "AddTex"),
    ("eyebrowId", "st_eyebrow", "AddTex"),
    ("eyelashesId", "st_eyelash", "AddTex"),
    ("pupilId", "st_eye", "AddTex"),
    ("blackId", "st_eyeblack", "AddTex"),
    ("hlId", "st_eye_hl", "AddTex"),
    ("eyeshadowId", "st_eyeshadow", "AddTex"),
    ("cheekId", "st_cheek", "AddTex"),
    ("lipId", "st_lip", "AddTex"),
    ("moleId", "st_mole", "AddTex"),
]


# ---------------------------------------------------------------- mesh / skeleton
def _read_transforms(objs):
    """(trans, go2t): path_id -> dict(name, parent, pos, rot(xyzw), scale), and GameObject -> Transform."""
    trans, go2t = {}, {}
    for o in objs:
        if o.type.name != "Transform":
            continue
        t = o.read()
        name, go_pid = None, None
        try:
            go_pid = t.m_GameObject.path_id
            name = t.m_GameObject.read().m_Name
        except Exception:  # noqa: BLE001
            pass
        try:
            father = t.m_Father.path_id if t.m_Father and t.m_Father.path_id else None
        except Exception:  # noqa: BLE001
            father = None
        lp, lr, ls = t.m_LocalPosition, t.m_LocalRotation, t.m_LocalScale
        trans[o.path_id] = {"name": name, "parent": father,
                            "pos": [lp.x, lp.y, lp.z], "rot": [lr.x, lr.y, lr.z, lr.w],
                            "scale": [ls.x, ls.y, ls.z]}
        if go_pid is not None:
            go2t[go_pid] = o.path_id
    return trans, go2t


def _chain(trans, pid):
    """Names from a transform up to the root — used to tell p_cf_head_02 from p_cf_head_02_hit."""
    out, cur, guard = [], pid, 0
    while cur in trans and guard < 32:
        out.append(trans[cur]["name"])
        cur = trans[cur]["parent"]
        guard += 1
    return out


def _read_smr_mesh(mesh):
    """All per-vertex arrays we need. Tangents/normals are AUTHORED — required for normal mapping."""
    h = MeshHandler(mesh)
    h.process()
    n = len(h.m_Vertices)

    def arr(attr, width, default=None):
        v = getattr(h, attr, None)
        if v is None or not len(v):
            return np.zeros((n, width), np.float32) if default is None else default
        return np.asarray(v, np.float32).reshape(n, width)

    # Vertex colours come back as 0..255 bytes; the shaders read them as 0..1 (they gate layers —
    # e.g. `1 - color.z` masks where the eyebrow may appear), so normalise here.
    colors = arr("m_Colors", 4, np.ones((n, 4), np.float32))
    if colors.size and colors.max() > 1.5:
        colors = colors / 255.0

    return {
        "verts": np.asarray(h.m_Vertices, np.float32).reshape(n, 3),
        "faces": np.asarray(h.m_IndexBuffer, np.int64).reshape(-1, 3),
        "normals": arr("m_Normals", 3),
        "tangents": arr("m_Tangents", 4),
        "uv": arr("m_UV0", 2),
        "uv1": arr("m_UV1", 2),
        "colors": colors,
        "bone_idx": np.asarray(h.m_BoneIndices, np.int64).reshape(n, 4),
        "bone_w": np.asarray(h.m_BoneWeights, np.float32).reshape(n, 4),
        "bindpose": np.asarray([[[m.e00, m.e01, m.e02, m.e03], [m.e10, m.e11, m.e12, m.e13],
                                 [m.e20, m.e21, m.e22, m.e23], [m.e30, m.e31, m.e32, m.e33]]
                                for m in mesh.m_BindPose], np.float32),
    }


def extract_head_rig(bundle: str, prefab: str, out_dir: str):
    """Extract the render prefab's o_head + every submesh under it, plus the FK skeleton."""
    env = UnityPy.load(bundle)
    objs = list(env.objects)
    trans, go2t = _read_transforms(objs)

    # every SMR that lives under `prefab` (exact name match excludes `<prefab>_hit`)
    found = []
    for o in objs:
        if o.type.name != "SkinnedMeshRenderer":
            continue
        smr = o.read()
        try:
            mesh = smr.m_Mesh.read()
            pid = go2t[smr.m_GameObject.path_id]
        except Exception:  # noqa: BLE001
            continue
        if prefab in _chain(trans, pid):
            found.append((mesh.m_Name, smr, mesh, o))
    if not found:
        raise SystemExit(f"no SkinnedMeshRenderer under prefab {prefab} in {bundle}")

    head = [f for f in found if f[0] == "o_head"]
    if len(head) != 1:
        raise SystemExit(f"expected exactly one o_head under {prefab}, got {len(head)}")

    mat_by_pid = {o.path_id: o for o in objs if o.type.name == "Material"}
    shader_name = {o.path_id: (o.read_typetree().get("m_ParsedForm", {}) or {}).get("m_Name")
                   for o in objs if o.type.name == "Shader"}
    tex_name = {o.path_id: o.read().m_Name for o in objs if o.type.name == "Texture2D"}

    def materials_of(obj):
        """(material name, shader name, {slot: texture-or-EXT}) for a renderer."""
        for mp in obj.read_typetree().get("m_Materials", []):
            mo = mat_by_pid.get(mp["m_PathID"])
            if not mo:
                continue
            mtt = mo.read_typetree()
            slots = {}
            for te in mtt.get("m_SavedProperties", {}).get("m_TexEnvs", []):
                slot = te[0] if isinstance(te, (list, tuple)) else te.get("first")
                tenv = te[1] if isinstance(te, (list, tuple)) else te.get("second")
                tref = (tenv or {}).get("m_Texture", {}) or {}
                fid, pid = tref.get("m_FileID", 0), tref.get("m_PathID", 0)
                if pid:
                    slots[slot] = tex_name.get(pid, f"EXT(fid{fid})") if fid == 0 else f"EXT(fid{fid})"
            return (mtt.get("m_Name"),
                    shader_name.get(mtt.get("m_Shader", {}).get("m_PathID")), slots)
        return (None, None, {})

    # ---- head mesh + skeleton
    _, smr, mesh, obj = head[0]
    m = _read_smr_mesh(mesh)
    skin_pids = [b.path_id for b in smr.m_Bones]
    skin_names = [trans.get(p, {}).get("name") for p in skin_pids]

    # The skeleton must cover EVERY submesh's skin bones too (the eyes ride their own bones), or
    # those submeshes can't be posed later — o_head's bone set alone is not enough.
    need = set()
    for _, smr_i, _, _ in found:
        for b in smr_i.m_Bones:
            cur = b.path_id
            while cur is not None and cur in trans and cur not in need:  # + all ancestors, for FK
                need.add(cur)
                cur = trans[cur]["parent"]
    skel = {}
    for pid in need:
        b = dict(trans[pid])
        b["parent"] = str(b["parent"]) if b["parent"] is not None else None
        skel[str(pid)] = b

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "o_head_mesh.npz"), **m)
    with open(os.path.join(out_dir, "skeleton.json"), "w", encoding="utf-8") as f:
        json.dump({"skin_bone_pids": [str(p) for p in skin_pids],
                   "skin_bone_names": skin_names, "bones": skel}, f)
    mat, shader, slots = materials_of(obj)
    print(f"[head] {prefab}: {len(m['verts'])} verts {len(m['faces'])} tris, "
          f"{len(skin_names)} skin bones, {len(skel)} skeleton bones")
    print(f"[head] material={mat} shader={shader}")

    # ---- submeshes (eyes / lashes / tear / eyeshadow / teeth / tongue)
    sub_dir = os.path.join(out_dir, "submeshes")
    os.makedirs(sub_dir, exist_ok=True)
    manifest, seen = [], set()
    for name, smr_i, mesh_i, obj_i in found:
        if name == "o_head" or name in seen:
            continue
        try:
            sm = _read_smr_mesh(mesh_i)
        except Exception as exc:  # noqa: BLE001 - odd vertex layouts: log & skip
            print(f"[sub] {name}: mesh read failed ({type(exc).__name__}: {exc}) -> skip")
            continue
        np.savez(os.path.join(sub_dir, f"{name}.npz"), **sm,
                 skin_bone_pids=np.asarray([str(b.path_id) for b in smr_i.m_Bones]))
        mat_i, shader_i, slots_i = materials_of(obj_i)
        manifest.append({"mesh": name, "material": mat_i, "shader": shader_i, "tex": slots_i})
        seen.add(name)
        print(f"[sub] {name}: {len(sm['verts'])} verts {len(sm['faces'])} tris  "
              f"mat={mat_i} shader={shader_i}")
    manifest.insert(0, {"mesh": "o_head", "material": mat, "shader": shader, "tex": slots})
    with open(os.path.join(sub_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return len(m["verts"])


def _kv(x):
    return (x[0], x[1]) if isinstance(x, (list, tuple)) else (x.get("first"), x.get("second"))


def _material_props(tt, tex_name=None):
    """A material's saved properties as plain JSON: colors / floats / texture slots.

    NB: Unity stores Vector4 material properties in `m_Colors`, so the layout vectors the
    composition shader reads (`_Texture9UV` = lip placement, etc.) come out under "colors".
    """
    sp = tt.get("m_SavedProperties", {})
    colors = {k: [c.get("r"), c.get("g"), c.get("b"), c.get("a")]
              for k, c in (_kv(x) for x in sp.get("m_Colors", []))}
    floats = {k: float(v) for k, v in (_kv(x) for x in sp.get("m_Floats", []))}
    tex = {}
    for x in sp.get("m_TexEnvs", []):
        k, v = _kv(x)
        ref = (v or {}).get("m_Texture", {}) or {}
        pid, fid = ref.get("m_PathID", 0), ref.get("m_FileID", 0)
        name = (tex_name or {}).get(pid) if fid == 0 else None
        tex[k] = {"asset": name, "external": bool(pid) and name is None,
                  "scale": [(v.get("m_Scale") or {}).get("x", 1), (v.get("m_Scale") or {}).get("y", 1)],
                  "offset": [(v.get("m_Offset") or {}).get("x", 0), (v.get("m_Offset") or {}).get("y", 0)]}
    return {"name": tt.get("m_Name"), "colors": colors, "floats": floats, "tex": tex}


def extract_materials(bundle: str, out_dir: str):
    """Dump every material's REAL shading params -> materials/<name>.json."""
    env = UnityPy.load(bundle)
    mat_dir = os.path.join(out_dir, "materials")
    os.makedirs(mat_dir, exist_ok=True)
    tex_name = {o.path_id: o.read().m_Name for o in env.objects if o.type.name == "Texture2D"}

    n = 0
    for o in env.objects:
        if o.type.name != "Material":
            continue
        d = _material_props(o.read_typetree(), tex_name)
        with open(os.path.join(mat_dir, f"{d['name']}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        n += 1
    print(f"[mat] {n} materials -> {mat_dir}")


COMPOSITE_BUNDLE = os.path.join(AB, "chara", "mm_base.unity3d")
COMPOSITE_MATS = ("create_skin_face", "create_skin detail_face")


def extract_composite_assets(out_dir: str) -> dict:
    """The runtime face-texture composition inputs (`Create/skin color` / `Create/skin detail`).

    `AIT/Skin True Face`'s `_MainTex` is NOT a file — ChaControl blits these two materials into a
    2048^2 RenderTexture at load (see docs/hs2-shader-recipe.md). Their saved properties carry the
    layer placements the card never overrides (e.g. `_Texture9UV` = where lipstick lands), and
    their in-bundle masks (`paint_mask_face`) gate the paint layers.
    """
    env = UnityPy.load(COMPOSITE_BUNDLE)
    objs = list(env.objects)
    tex_name = {o.path_id: o.read().m_Name for o in objs if o.type.name == "Texture2D"}
    comp_dir = os.path.join(out_dir, "composite")
    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)

    out, wanted = {}, set()
    for o in objs:
        if o.type.name != "Material":
            continue
        tt = o.read_typetree()
        if tt.get("m_Name") not in COMPOSITE_MATS:
            continue
        d = _material_props(tt, tex_name)
        with open(os.path.join(comp_dir, f"{d['name']}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        out[d["name"]] = d
        wanted |= {t["asset"] for t in d["tex"].values() if t["asset"]}
        print(f"[comp] {d['name']}: {len(d['tex'])} slots, masks={sorted(t['asset'] for t in d['tex'].values() if t['asset'])}")

    saved = _save_textures_from(COMPOSITE_BUNDLE, wanted, tex_dir)
    # black2048 is the detail pass's base texture (ChaControl loads it from chara/etc.unity3d)
    etc = os.path.join(AB, "chara", "etc.unity3d")
    if os.path.exists(etc):
        saved.update(_save_textures_from(etc, {"black2048"}, tex_dir))
    merge_pool(out_dir, saved)
    print(f"[comp] {len(saved)} composition texture(s) -> {tex_dir}")
    return out


# ---------------------------------------------------------------- textures
_TEX_CACHE: dict[str, dict] = {}


def _save_textures_from(bundle: str, wanted: set, tex_dir: str) -> dict:
    """Export the named Texture2Ds from one bundle -> tex_dir/<name>.png. Returns {name: info}.

    Records `m_ColorSpace` (1 = sRGB, 0 = linear/non-colour). The game renders in LINEAR space —
    `CustomTextureCreate` explicitly sets `GL.sRGBWrite = true`, which is a no-op in gamma space —
    so an sRGB texture is linearised on sample while a linear one is not. The face composition
    mixes both (skin = sRGB, lipstick and masks = linear), so consumers must honour this per
    texture rather than assume.
    """
    out = {}
    env = UnityPy.load(bundle)
    for o in env.objects:
        if o.type.name != "Texture2D":
            continue
        try:
            t = o.read()
        except Exception:  # noqa: BLE001
            continue
        if t.m_Name not in wanted:
            continue
        try:
            img = t.image
            img.save(os.path.join(tex_dir, f"{t.m_Name}.png"))
        except Exception as exc:  # noqa: BLE001 - undecodable format
            print(f"[tex]   {t.m_Name}: decode failed ({type(exc).__name__}) -> skip")
            continue
        try:
            cs = int(o.read_typetree().get("m_ColorSpace", 1))
        except Exception:  # noqa: BLE001
            cs = 1
        out[t.m_Name] = {"file": f"textures/{t.m_Name}.png", "w": img.width, "h": img.height,
                         "bundle": os.path.basename(bundle), "srgb": cs == 1}
    return out


def extract_card_textures(cl: ChaList, ids: dict, out_dir: str) -> dict:
    """Export every card-selected texture (skin/eyes/brow/lash/makeup) into the shared pool."""
    tex_dir = os.path.join(out_dir, "textures")
    os.makedirs(tex_dir, exist_ok=True)

    by_bundle: dict[str, set] = {}
    slots: dict[str, dict] = {}
    for id_key, cat, col in TEXTURE_IDS:
        try:
            row = cl.resolve(cat, ids[id_key])
        except KeyError as exc:
            print(f"[tex] {cat} id={ids[id_key]}: {exc}")
            continue
        asset = row.get(col)
        if not asset or asset in ("0", "none"):        # id 0 in these lists means "none"
            continue
        try:
            path = cl.bundle_path(row)
        except (KeyError, FileNotFoundError) as exc:
            print(f"[tex] {cat}.{col}={asset}: {exc}")
            continue
        by_bundle.setdefault(path, set()).add(asset)
        slots[f"{cat}.{col}"] = {"id": ids[id_key], "asset": asset}

    saved = {}
    for path, wanted in by_bundle.items():
        got = _save_textures_from(path, wanted, tex_dir)
        saved.update(got)
        missing = wanted - set(got)
        print(f"[tex] {os.path.basename(path)}: {len(got)}/{len(wanted)}"
              f"{' MISSING ' + str(sorted(missing)) if missing else ''}")
    for key, info in slots.items():
        info.update(saved.get(info["asset"], {}))
    return slots


def merge_pool(out_dir: str, infos: dict):
    """Merge texture infos into textures/pool.json (name -> file/size/srgb) for the renderer."""
    path = os.path.join(out_dir, "textures", "pool.json")
    pool = {}
    if os.path.exists(path):
        try:
            pool = json.load(open(path, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pool = {}
    pool.update(infos)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pool, f, indent=2)
    return pool


def extract_bundle_textures(bundle: str, out_dir: str) -> dict:
    """Export every in-bundle Texture2D of the head bundle (detail/mask/eye/tooth layers)."""
    env = UnityPy.load(bundle)
    names = set()
    for o in env.objects:
        if o.type.name == "Texture2D":
            try:
                names.add(o.read().m_Name)
            except Exception:  # noqa: BLE001
                continue
    infos = _save_textures_from(bundle, names, os.path.join(out_dir, "textures"))
    print(f"[tex] {os.path.basename(bundle)}: {len(infos)} in-bundle textures")
    return infos


# ---------------------------------------------------------------- TextAsset tables
def _read_dotnet_string(br):
    """.NET BinaryReader.ReadString: 7-bit-encoded length prefix, then UTF-8 bytes."""
    length, shift = 0, 0
    while True:
        b = br.read(1)[0]
        length |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return br.read(length).decode("utf-8")


def parse_anm_shape(bundle: str, asset_name: str):
    """cf_anmShapeHead_NN (binary TextAsset) -> {bone: [{no,pos,rot,scl}, ...]} keyframe table."""
    env = UnityPy.load(bundle)
    data = None
    for o in env.objects:
        if o.type.name != "TextAsset":
            continue
        tt = o.read_typetree()
        if tt.get("m_Name") == asset_name:
            s = tt.get("m_Script")
            data = s.encode("utf-8", "surrogateescape") if isinstance(s, str) else bytes(s)
            break
    if data is None:
        raise SystemExit(f"{asset_name} not found in {bundle}")
    br = io.BytesIO(data)
    out = {}
    for _ in range(struct.unpack("<i", br.read(4))[0]):
        name = _read_dotnet_string(br)
        frames = []
        for _ in range(struct.unpack("<i", br.read(4))[0]):
            no = struct.unpack("<i", br.read(4))[0]
            v = struct.unpack("<9f", br.read(36))
            frames.append({"no": no, "pos": v[0:3], "rot": v[3:6], "scl": v[6:9]})
        out[name] = frames
    print(f"[anm] {asset_name}: {len(out)} bones, "
          f"{len(next(iter(out.values())))} keyframes/bone")
    return out


def parse_customhead():
    """cf_customhead (CSV TextAsset) -> [{category, bone, use[9]}] slider -> cf_s_ bone + DOF flags."""
    env = UnityPy.load(CUSTOMSHAPE)
    txt = None
    for o in env.objects:
        if o.type.name != "TextAsset":
            continue
        tt = o.read_typetree()
        if tt.get("m_Name") == "cf_customhead":
            s = tt.get("m_Script")
            txt = s if isinstance(s, str) else bytes(s).decode("utf-8", "replace")
            break
    if txt is None:
        raise SystemExit("cf_customhead not found")
    rows = []
    for line in txt.splitlines():
        if not line.strip():
            continue
        c = line.rstrip("\r").split("\t")
        rows.append({"category": int(c[0]), "bone": c[1],
                     "use": [int(x != "0") for x in c[2:11]]})  # pos.xyz, rot.xyz, scl.xyz
    print(f"[customhead] {len(rows)} rows")
    return rows


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--no-cache", action="store_true", help="rescan the game's asset lists")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    cl = ChaList(use_cache=not args.no_cache)
    ids = face_ids(args.card)
    head_row = cl.resolve("fo_head", ids["headId"])
    head_bundle = cl.bundle_path(head_row)
    prefab, anm_name = head_row["MainData"], head_row["ShapeAnime"]
    print(f"[card] {args.card}")
    print(f"[card] headId={ids['headId']} -> {os.path.relpath(head_bundle, AB)} :: {prefab}  "
          f"(anm={anm_name}, mat={head_row['MatData']})")

    head_dir = os.path.join(OUT, f"head_{ids['headId']}")
    nverts = extract_head_rig(head_bundle, prefab, head_dir)
    extract_materials(head_bundle, head_dir)

    with open(os.path.join(head_dir, "anmShapeHead.json"), "w", encoding="utf-8") as f:
        json.dump(parse_anm_shape(head_bundle, anm_name), f)
    with open(os.path.join(OUT, "customhead.json"), "w", encoding="utf-8") as f:
        json.dump(parse_customhead(), f)

    merge_pool(OUT, extract_bundle_textures(head_bundle, OUT))
    slots = extract_card_textures(cl, ids, OUT)
    extract_composite_assets(OUT)

    cards_dir = os.path.join(OUT, "cards")
    os.makedirs(cards_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.card))[0]
    card_manifest = {"card": args.card, "ids": ids, "head_dir": f"head_{ids['headId']}",
                     "prefab": prefab, "shape_anime": anm_name,
                     "head_material": head_row["MatData"], "verts": nverts, "textures": slots}
    with open(os.path.join(cards_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump(card_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[done] rig -> {head_dir}")
    print(f"[done] card manifest -> {os.path.join(cards_dir, stem + '.json')}")


if __name__ == "__main__":
    main()
