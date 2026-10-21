"""Which installed file a Sideloader-era HS2 asset load actually reads.

Reimplements, from the installed ``HS2_Sideloader.dll`` (BepisPlugins; SHA-256 in
``SIDELOADER_DLL_SHA256``), the parts of mod loading that decide asset identity:

* ``Sideloader.LoadModsFromDirectories``: every ``*.zip``/``*.zipmod`` below the
  mods directory (recursive); an archive is skipped without ``manifest.xml``, with a
  ``schema-ver`` other than ``1``, without ``<guid>``, or with ``<game>`` entries none
  of which (lower-cased) is ``hs2``/``honeyselect2``/``honey select 2``.  Archives
  are grouped by GUID; within a group the first of
  ``OrderByDescending(Version, ManifestVersionComparer).ThenByDescending(name
  length)`` is loaded when every member has a version, otherwise
  ``OrderByDescending(File.GetLastWriteTime)``.  Groups are loaded in
  ``orderby GUID`` order.
* ``LoadAllUnityArchives`` -> ``BundleManager.AddBundleLoader``: each ``.unity3d``
  member is registered under its path without the first component
  (``abdata/chara/x.unity3d`` -> ``chara/x.unity3d``), appended in load order.
* ``RedirectHook`` -> ``BundleManager.TryGetObjectFromName``: an asset load from
  bundle path *B* returns the asset from the first registered zipmod bundle of *B*
  that ``Contains`` the name; only when none does, the vanilla file is read.  For
  Texture types a zipmod ``abdata/<B without .unity3d>/<name>.png`` wins first
  (``GetPng``: runtime ``Texture2D.LoadImage``, i.e. a runtime-decoded texture).

Ordering caveat (explicit): ``orderby x.Key`` uses ``Comparer<string>.Default``,
which is culture-sensitive in Mono (never ordinal: ``com.x`` sorts before ``HS2.y``).
The collation tables are not reimplemented; ``culture_order`` accepts an order only
where several case-insensitive collation proxies agree, and ``bundle_sources``
raises for a contested bundle path whose order they do not determine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SIDELOADER_DLL = 'BepInEx/Plugins/HS2_BepisPlugins/HS2_Sideloader.dll'
SIDELOADER_DLL_SHA256 = None  # filled by scan() from the installed file
GAME_NAMES = ('hs2', 'honeyselect2', 'honey select 2')
SCAN_SCHEMA = 'hs2pipe.sideloader_scan/1'


class SideloaderUnresolved(RuntimeError):
    pass


def _file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def _scan_one(path):
    rec = {'path': path, 'mtime_ns': os.stat(path).st_mtime_ns, 'size': os.stat(path).st_size}
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            info = {i.filename: i for i in z.infolist()}
            if 'manifest.xml' not in info:
                rec['skip'] = 'no manifest.xml'
                return rec
            text = z.read('manifest.xml')
    except (zipfile.BadZipFile, OSError) as e:
        rec['skip'] = 'archive error %r' % (e,)
        return rec
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        rec['skip'] = 'manifest parse error %r' % (e,)
        return rec
    rec['schema_ver'] = root.attrib.get('schema-ver')
    g = root.find('guid')
    rec['guid'] = g.text if g is not None else None
    v = root.find('version')
    rec['version'] = v.text if v is not None else None
    rec['games'] = [e.text for e in root.findall('game') if e.text and e.text.strip()]
    rec['unity3d'] = [n for n in names if n.lower().endswith('.unity3d')]
    rec['png'] = [n for n in names if n.lower().startswith('abdata/') and n.lower().endswith('.png')]
    rec['uncensor_selector'] = root.find('KK_UncensorSelector') is not None
    rec['manifest_sha256'] = hashlib.sha256(text).hexdigest()
    if rec['uncensor_selector']:
        rec['manifest_xml'] = text.decode('utf-8', 'replace')
    return rec


def scan(game_root, out_json, workers=8):
    """Scan every archive below ``<game>/mods`` (and the configured additional mods
    directory, which is empty in this install).  Writes a cache JSON."""
    import concurrent.futures
    g = Path(game_root)
    cfg = (g / 'BepInEx/config/com.bepis.bepinex.sideloader.cfg').read_text(encoding='utf-8', errors='replace')
    m = re.search(r'^Additional mods directory\s*=\s*(.*)$', cfg, re.M)
    extra = (m.group(1).strip() if m else '')
    dirs = [str(g / 'mods')] + ([extra] if extra else [])
    paths = []
    for d in dirs:
        for dp, _, fn in os.walk(d):
            for f in fn:
                if f.lower().endswith(('.zip', '.zipmod')):
                    paths.append(os.path.join(dp, f))
    with concurrent.futures.ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(_scan_one, paths))
    doc = {'schema': SCAN_SCHEMA, 'game_root': str(g), 'mods_directories': dirs,
           'sideloader_dll_sha256': _file_sha256(g / SIDELOADER_DLL), 'archives': recs}
    Path(out_json).write_text(json.dumps(doc), encoding='utf-8')
    return doc


# ------------------------------------------------- ManifestVersionComparer
def _version_parts(v):
    out = []
    for part in re.split(r"[. \-,_]", v.strip()):
        try:
            if re.fullmatch(r'[+-]?\d+', part.strip()) and -2**31 <= int(part) < 2**31:
                out.append(int(part))
            else:
                out.append(part)
        except ValueError:
            out.append(part)
    return out


def _cmp_component(a, b):
    """IComparable.CompareTo of int/string; ArgumentException (int vs string) falls
    back to ordinal only when both are strings, otherwise the component is skipped."""
    if isinstance(a, int) and isinstance(b, int):
        return (a > b) - (a < b)
    if isinstance(a, str) and isinstance(b, str):
        # String.CompareTo is culture-sensitive; ordinal used only when culture and
        # ordinal agree on the pair (checked by the caller via raise).
        o = (a > b) - (a < b)
        c = (a.lower() > b.lower()) - (a.lower() < b.lower())
        if o != c and a.lower() != b.lower():
            raise SideloaderUnresolved('culture-sensitive version component comparison %r vs %r' % (a, b))
        return c if a.lower() != b.lower() else o
    return 0  # ArgumentException with a non-string side: no decision from this component


def compare_versions(x, y):
    if x == y:
        return 0
    f, s = _version_parts(x), _version_parts(y)
    for i in range(max(len(f), len(s))):
        a = f[i] if i < len(f) else ''
        b = s[i] if i < len(s) else ''
        r = _cmp_component(a, b)
        if r:
            return r
    return (len(f) > len(s)) - (len(f) < len(s))


# Mono ``Comparer<string>.Default`` (culture-sensitive CompareInfo.Compare, CompareOptions.None).
# Its exact collation tables are not reimplemented.  Every proxy below orders Latin letters
# case-insensitively (as any culture collation does); they differ only in how punctuation is
# weighted (ignored hyphen/apostrophe "word sort", all punctuation ignored, punctuation
# before alphanumerics).  An order is accepted only where all proxies agree.
COLLATION_PROXIES = (
    lambda x: (re.sub(r"[-']", '', x.lower()), x),
    lambda x: (re.sub(r'[^0-9a-z]', '', x.lower()), x),
    lambda x: (''.join(chr(1) + c if not c.isalnum() else c for c in x.lower()), x),
)


def culture_order(guids):
    """GUIDs in Mono culture-sensitive order, or SideloaderUnresolved when the order of
    these strings depends on collation details not reproduced here."""
    g = list(guids)
    orders = {tuple(sorted(g, key=k)) for k in COLLATION_PROXIES}
    if len(orders) != 1:
        raise SideloaderUnresolved('culture-sensitive GUID order of %r is undetermined' % (g,))
    return list(next(iter(orders)))


class SideloaderState:
    """Loaded mods and bundle registrations reproduced from a scan."""

    def __init__(self, doc):
        if doc.get('schema') != SCAN_SCHEMA:
            raise SideloaderUnresolved('scan schema mismatch')
        self.doc = doc
        self.skipped = []
        groups = {}
        for r in doc['archives']:
            if r.get('skip'):
                self.skipped.append((r['path'], r['skip']))
                continue
            if r.get('schema_ver') != '1' or not r.get('guid'):
                self.skipped.append((r['path'], 'invalid manifest (schema-ver/guid)'))
                continue
            if r['games'] and not any(gm.lower() in GAME_NAMES for gm in r['games']):
                self.skipped.append((r['path'], 'other game %r' % r['games']))
                continue
            groups.setdefault(r['guid'], []).append(r)
        self.duplicates = []
        self.loaded = []
        import functools
        for guid in sorted(groups, key=COLLATION_PROXIES[0]):
            items = groups[guid]
            if all(x.get('version') for x in items):
                items = sorted(items, key=lambda x: len(x['path']), reverse=True)
                items = sorted(items, key=functools.cmp_to_key(lambda a, b: compare_versions(a['version'], b['version'])),
                               reverse=True)
            else:
                items = sorted(items, key=lambda x: x['mtime_ns'], reverse=True)
            self.loaded.append(items[0])
            if len(items) > 1:
                self.duplicates.append({'guid': guid, 'loaded': items[0]['path'], 'skipped': [x['path'] for x in items[1:]]})
        self.bundles = {}
        for r in self.loaded:
            for n in r['unity3d']:
                key = n[n.index('/') + 1:] if '/' in n else n
                self.bundles.setdefault(key, []).append((r['path'], n, r['guid']))
        self.png = {}
        for r in self.loaded:
            for n in r['png']:
                self.png.setdefault(n, (r['path'], r['guid']))   # first registration wins (PngList.Add)

    def bundle_sources(self, bundle_path):
        """[(zip path, member, guid)] registered for a bundle path, in TryGetObjectFromName order.

        Raises when the GUID order of the competing mods could depend on Mono's
        culture-sensitive comparison (ordinal, case-folded and punctuation-free
        orders disagree)."""
        rows = list(self.bundles.get(bundle_path, []))
        if len(rows) > 1:
            culture_order(r[2] for r in rows)       # raises if the collation proxies disagree
        return rows

    def png_override(self, bundle_path, asset_name):
        key = 'abdata/' + re.sub(r'\.unity3d$', '', bundle_path, flags=re.I) + '/' + asset_name + '.png'
        return self.png.get(key)

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))
