"""Offline ChaListControl — resolve a card's *IDs* to the game's actual asset bundles/assets.

The game never hardcodes "the head is fo_head_00.unity3d". `ChaListControl` merges every
`abdata/list/characustom/*.unity3d` list into `categoryNo -> id -> row`, and each row names the
bundle + asset to load. Reproducing that lookup offline is what lets us extract *the assets this
card actually uses* instead of guessing (see docs/hs2-renderer-and-mesh.md).

Each list TextAsset is a msgpack `ChaListData`:
    {mark, categoryNo, distributionNo, filePath, lstKey: [col names], dictList: {id: [values]}}

Example — for tests/HS2ChaF_20240901192905747.png (headId=2, skinId=20):
    fo_head    id=2  -> chara/38/fo_head_38.unity3d  p_cf_head_02  (ShapeAnime cf_anmShapeHead_02)
    ft_skin_f  id=20 -> chara/30/ft_skin_f_30.unity3d  cf_head_02_00_t

    from src.hs2_assets import ChaList
    cl = ChaList()
    row = cl.resolve("fo_head", 2)              # dict keyed by lstKey
    path = cl.bundle_path(row)                  # abs path to the .unity3d

Note: sideloader zipmods (`mods/**.zipmod`) also register list entries; only `abdata/` is scanned
here, so a card using a zipmod-only item will raise KeyError (loudly, not silently wrong).
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

import msgpack
import UnityPy

HS2 = os.environ.get("HS2_DIR", r"E:\HoneySelect2_ArcticFox")
AB = os.path.join(HS2, "abdata")
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "data", "hs2_head", "chalist_cache.json")

# ChaListDefine.CategoryNo — the female-face subset the renderer needs. Values verified by
# scanning every list bundle in abdata/list/characustom/ (the categoryNo field of each list).
CATEGORY = {
    "fo_head": 210,        # headId    -> MainAB/MainData (prefab) + ShapeAnime + MatData
    "ft_skin_f": 211,      # skinId    -> MainTex / OcclusionMapTex / NormalMapTex  (+HeadID)
    "ft_detail_f": 212,    # detailId  -> AddTex (skin detail / wrinkles)
    "ft_sunburn": 233,     # sunburnId -> AddTex
    "st_paint": 313,       # face/body paint (tattoo) -> AddTex + GlossTex
    "st_eyebrow": 314,     # eyebrowId    -> AddTex
    "st_eyelash": 315,     # eyelashesId  -> AddTex
    "st_eyeshadow": 316,   # makeup.eyeshadowId -> AddTex + GlossTex
    "st_eye": 317,         # pupil[].pupilId    -> AddTex (iris)
    "st_eyeblack": 318,    # pupil[].blackId    -> AddTex (pupil/black)
    "st_eye_hl": 319,      # hlId               -> AddTex (highlight)
    "st_cheek": 320,       # makeup.cheekId     -> AddTex + GlossTex
    "st_lip": 322,         # makeup.lipId       -> AddTex + GlossTex
    "st_mole": 323,        # moleId             -> AddTex + GlossTex
    "mole_layout": 6,      # moleLayout preset  -> PosX / PosY / Scale
    "facepaint_layout": 7,
}


def _script_bytes(tt):
    """TextAsset m_Script -> bytes (UnityPy hands back a surrogate-escaped str for binary assets)."""
    s = tt.get("m_Script")
    return s.encode("utf-8", "surrogateescape") if isinstance(s, str) else bytes(s)


class ChaList:
    """Merged view of every `abdata/list/characustom/*.unity3d` list, keyed by category name."""

    def __init__(self, ab_dir: str = AB, use_cache: bool = True):
        self.ab_dir = ab_dir
        self.by_cat: dict[int, dict[int, dict]] = {}
        if use_cache and self._load_cache():
            return
        self._scan()
        if use_cache:
            self._save_cache()

    # ---------- build ----------
    def _scan(self):
        t0 = time.time()
        pattern = os.path.join(self.ab_dir, "list", "characustom", "*.unity3d")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"no list bundles under {pattern} (is HS2_DIR right? got {HS2})")
        for path in files:
            env = UnityPy.load(path)
            for obj in env.objects:
                if obj.type.name != "TextAsset":
                    continue
                tt = obj.read_typetree()
                try:
                    d = msgpack.unpackb(_script_bytes(tt), raw=False, strict_map_key=False)
                except Exception:  # noqa: BLE001 - not every TextAsset is a ChaListData
                    continue
                if not isinstance(d, dict) or "lstKey" not in d or "dictList" not in d:
                    continue
                keys = d["lstKey"]
                cat = self.by_cat.setdefault(int(d["categoryNo"]), {})
                for id_, row in d["dictList"].items():
                    r = dict(zip(keys, row))
                    r["_list"] = tt.get("m_Name")
                    cat[int(id_)] = r
        n = sum(len(v) for v in self.by_cat.values())
        print(f"[chalist] {len(files)} bundles, {len(self.by_cat)} categories, {n} entries "
              f"({time.time() - t0:.1f}s)")

    def _load_cache(self) -> bool:
        if not os.path.exists(_CACHE):
            return False
        try:
            with open(_CACHE, encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:  # noqa: BLE001 - corrupt cache: just rebuild
            return False
        if raw.get("ab_dir") != self.ab_dir:
            return False
        self.by_cat = {int(c): {int(i): r for i, r in rows.items()}
                       for c, rows in raw["by_cat"].items()}
        return True

    def _save_cache(self):
        os.makedirs(os.path.dirname(_CACHE), exist_ok=True)
        with open(_CACHE, "w", encoding="utf-8") as f:
            json.dump({"ab_dir": self.ab_dir, "by_cat": self.by_cat}, f)

    # ---------- query ----------
    def resolve(self, category, id_: int) -> dict:
        """Row dict for (category, id). `category` is a CATEGORY name or a raw categoryNo."""
        cat_no = CATEGORY[category] if isinstance(category, str) else int(category)
        rows = self.by_cat.get(cat_no)
        if not rows:
            raise KeyError(f"category {category} ({cat_no}) not present in the merged lists")
        if int(id_) not in rows:
            avail = sorted(rows)
            raise KeyError(f"{category} id={id_} not found "
                           f"(have {len(avail)}: {avail[:12]}{'...' if len(avail) > 12 else ''}) "
                           f"— a sideloader zipmod item? only abdata/ is scanned")
        return rows[int(id_)]

    def bundle_path(self, row: dict, key: str = "MainAB") -> str:
        """Absolute path of the row's bundle. Tries `abdata/<AB>` then `abdata/<Manifest>/<AB>`."""
        rel = row.get(key)
        if not rel:
            raise KeyError(f"row has no {key}: {sorted(row)}")
        cands = [os.path.join(self.ab_dir, rel.replace("/", os.sep))]
        manifest = row.get("MainManifest") or row.get("TexManifest")
        if manifest and manifest != "abdata":
            cands.append(os.path.join(self.ab_dir, manifest, rel.replace("/", os.sep)))
        for c in cands:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(f"bundle not found for {key}={rel}: tried {cands}")


# ---------- card -> the ids the renderer cares about ----------
def face_ids(card_path: str) -> dict:
    """Every appearance-selecting id/colour a face render needs, as a plain dict.

    Mostly `card_data.Custom["face"]`, but a few properties the FACE material consumes live in the
    BODY block — notably the skin tone: `ChaControl.CreateFaceTexture` does
    `customTextureCreate.SetColor(ChaShader.SkinColor, fileBody.skinColor)`. Reading `face.skinId`
    for the tone would be wrong (that selects the *texture*, not the colour).
    """
    from .face_data_utils.utils import AiSyoujyoCharaData
    custom = AiSyoujyoCharaData.load(card_path, True).Custom
    face, body = custom["face"], custom["body"]
    pupil = (face.get("pupil") or [{}])[0]
    makeup = face.get("makeup") or {}
    paint = makeup.get("paintInfo") or []
    return {
        # --- from the BODY block, but applied to the face material ---
        "skinColor": list(body.get("skinColor", [1, 1, 1, 1])),
        "skinGlossPower": float(body.get("skinGlossPower", 0.0)),
        "skinMetallicPower": float(body.get("skinMetallicPower", 0.0)),
        "sunburnId": int(body.get("sunburnId", 0)),
        "sunburnColor": list(body.get("sunburnColor", [1, 1, 1, 0])),
        # --- makeup extras ---
        "eyeshadowGloss": float(makeup.get("eyeshadowGloss", 0.0)),
        "cheekGloss": float(makeup.get("cheekGloss", 0.0)),
        "lipGloss": float(makeup.get("lipGloss", 0.0)),
        "paintInfo": [{"id": int(p.get("id", 0)), "color": list(p.get("color", [1, 1, 1, 1])),
                       "layout": list(p.get("layout", [0.5, 0.5, 0.5, 0.5])),
                       "rotation": float(p.get("rotation", 0.5)),
                       "glossPower": float(p.get("glossPower", 0.0)),
                       "metallicPower": float(p.get("metallicPower", 0.0))} for p in paint],
        "headId": int(face["headId"]),
        "skinId": int(face["skinId"]),
        "detailId": int(face["detailId"]),
        "detailPower": float(face.get("detailPower", 1.0)),
        "eyebrowId": int(face["eyebrowId"]),
        "eyebrowColor": list(face.get("eyebrowColor", [0, 0, 0, 1])),
        "eyebrowLayout": list(face.get("eyebrowLayout", [0.5, 0.5, 0.5, 0.5])),
        "eyebrowTilt": float(face.get("eyebrowTilt", 0.5)),
        "pupilId": int(pupil.get("pupilId", 0)),
        "pupilColor": list(pupil.get("pupilColor", [0, 0, 0, 1])),
        "whiteColor": list(pupil.get("whiteColor", [1, 1, 1, 1])),
        "blackId": int(pupil.get("blackId", 0)),
        "hlId": int(face.get("hlId", 0)),
        "hlColor": list(face.get("hlColor", [1, 1, 1, 1])),
        "eyelashesId": int(face["eyelashesId"]),
        "eyelashesColor": list(face.get("eyelashesColor", [0, 0, 0, 1])),
        "moleId": int(face["moleId"]),
        "moleColor": list(face.get("moleColor", [0, 0, 0, 1])),
        "moleLayout": list(face.get("moleLayout", [0.5, 0.5, 0.5, 0.5])),
        "eyeshadowId": int(makeup.get("eyeshadowId", 0)),
        "eyeshadowColor": list(makeup.get("eyeshadowColor", [0, 0, 0, 0])),
        "cheekId": int(makeup.get("cheekId", 0)),
        "cheekColor": list(makeup.get("cheekColor", [0, 0, 0, 0])),
        "lipId": int(makeup.get("lipId", 0)),
        "lipColor": list(makeup.get("lipColor", [0, 0, 0, 0])),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", default="tests/HS2ChaF_20240901192905747.png")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cl = ChaList(use_cache=not args.no_cache)
    ids = face_ids(args.card)
    print(f"card: {args.card}")
    print("  " + json.dumps({k: v for k, v in ids.items() if not isinstance(v, list)}, indent=2)
          .replace("\n", "\n  "))

    # the resolutions the extractor depends on
    for cat, key, id_key in [("fo_head", "MainData", "headId"),
                             ("ft_skin_f", "MainTex", "skinId"),
                             ("ft_detail_f", "AddTex", "detailId"),
                             ("st_eyebrow", "AddTex", "eyebrowId"),
                             ("st_eyelash", "AddTex", "eyelashesId"),
                             ("st_eye", "AddTex", "pupilId"),
                             ("st_eyeblack", "AddTex", "blackId"),
                             ("st_eye_hl", "AddTex", "hlId"),
                             ("st_eyeshadow", "AddTex", "eyeshadowId"),
                             ("st_cheek", "AddTex", "cheekId"),
                             ("st_lip", "AddTex", "lipId"),
                             ("st_mole", "AddTex", "moleId")]:
        try:
            row = cl.resolve(cat, ids[id_key])
            extra = f"  anm={row['ShapeAnime']} mat={row['MatData']}" if "ShapeAnime" in row else ""
            print(f"  {cat:14s} id={ids[id_key]:<4} -> {row.get('MainAB')}  {row.get(key)}{extra}")
        except (KeyError, FileNotFoundError) as exc:
            print(f"  {cat:14s} id={ids[id_key]:<4} -> UNRESOLVED ({exc})")
