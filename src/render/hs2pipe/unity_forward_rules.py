"""Unity 2018.4.11f1 built-in FORWARD loop rules recovered from UnityPlayer.dll.

Source: ``E:/HoneySelect2_ArcticFox/UnityPlayer.dll`` (SHA-256 ``5034cdc2bff7…``,
CodeView GUID ``BFA91A8C44734BB4AABC5A975B304CFE`` age 1), disassembled with
``tools/offline_renderer/unityplayer_disasm.py`` against Unity's public-symbol
PDB.  Dumps: ``outputs/offline_renderer_unity_forward_20261020/disasm/forward``.
The specification with every RVA, struct offset and explicit gap is
``outputs/offline_renderer_unity_forward_20261020/forward_rules_spec.md``.

Contents
  1. Render-object extraction, ForwardRenderPassData construction, the two
     ``ForwardShaderRenderLoop::RenderObjectSorter<bool>`` comparators and the
     exact ``qsort_internal::QSort`` algorithm that orders the draws.
  2. Per-object forward light selection: ``CullAllPerObjectLights`` importance
     + MSVC ``std::sort``, ``FindForwardLightsForObject`` (main light, ForwardAdd
     list, four vertex lights, SH), the last-light blend factors, the SH
     projection of non-important lights and the ForwardBase vertex-light
     constants.
  3. Camera culling (frustum / layer / layer cull distance) and visible-list
     order.

Float arithmetic is IEEE binary32 with the operand order of the machine code
(numpy ``float32`` scalars).  Functions take plain dicts/lists so they stay
pure.  Anything the binary does that is not transcribed raises
``NotImplementedError`` naming the gap instead of approximating it.
"""
from __future__ import annotations

import struct

import numpy as np

F = np.float32
_ZERO = F(0.0)
_ONE = F(1.0)


def _fbits(bits):
    return np.frombuffer(struct.pack('<I', bits), dtype=np.float32)[0]


def f32_bits(x):
    """IEEE bit pattern of a float32 (``movss [mem]; mov eax,[mem]``)."""
    return int(np.float32(x).view(np.uint32))


def _u32(x):
    return int(x) & 0xFFFFFFFF


def _s32(x):
    x = int(x) & 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def _s16(x):
    x = int(x) & 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def _u16(x):
    return int(x) & 0xFFFF


def _gt(a, b):
    """``comiss a, b; seta`` -> False when unordered."""
    return bool(F(a) > F(b))


def _min_ss(a, b):
    """``minss a, b``: a < b ? a : b (returns b when unordered)."""
    a, b = F(a), F(b)
    return a if a < b else b


def _max_ss(a, b):
    """``maxss a, b``: a > b ? a : b (returns b when unordered)."""
    a, b = F(a), F(b)
    return a if a > b else b


def _clamp01_ss(x):
    """``xorps z; comiss z, x; jbe -> minss(1, x) ; else 0`` used repeatedly."""
    x = F(x)
    if _ZERO > x:
        return _ZERO
    return _min_ss(_ONE, x)


# Constants (bit-exact from .rdata).
K_LUM_R = _fbits(0x3E99999A)        # 0.3
K_LUM_G = _fbits(0x3F170A3D)        # 0.59
K_LUM_B = _fbits(0x3DE147AE)        # 0.11
K_SHADOW_DIR_BOOST = _fbits(0x41800000)   # 16.0
K_ATTEN = _fbits(0x41C80000)        # 25.0
K_BLEND_EPS = _fbits(0x3A83126F)    # 0.001
K_BLEND_MAX = _fbits(0x447A0000)    # 1000.0  ([14ae784])
RENDER_MODE_SORT_BIAS = (_fbits(0x00000000), _fbits(0x447A0000), _fbits(0xC47A0000))  # GetRenderModeSortBias table 0x14ae780
QUEUE_SPLIT = 2500                  # 0x9c4 at 0x6b692a: queue <= 2500 -> opaque list

FIRST_PASS_FLAG = 0x1000000         # kPackFirstPassFlag
MULTI_PASS_FLAG = 0x2000000
GRAB_PASS_FLAG = 0x4000000

LIGHT_SPOT, LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_RECTANGLE, LIGHT_DISC = 0, 1, 2, 3, 4
RENDER_MODE_AUTO, RENDER_MODE_FORCE_PIXEL, RENDER_MODE_FORCE_VERTEX = 0, 1, 2
SORT_DEFAULT, SORT_PERSPECTIVE, SORT_ORTHOGRAPHIC, SORT_CUSTOM_AXIS = 0, 1, 2, 3
OPAQUE_SORT_DEFAULT, OPAQUE_SORT_FRONT_TO_BACK, OPAQUE_SORT_NONE = 0, 1, 2
SHADOWS_ONLY = 3

PASS_NORMAL, PASS_VERTEX, PASS_VERTEX_LM, PASS_FORWARD_BASE, PASS_FORWARD_ADD = 0, 1, 2, 4, 5


# =========================================================================
# 1. Render objects, pass list and draw sorting
# =========================================================================
def resolve_transparency_sort(camera, graphics_settings):
    """DoRenderLoop 0x6b9a83..0x6b9b2f -> (mode, axis) stored at ctx+0x7b4/0x7a4.

    camera keys: ``implicit_transparency_sort`` (Camera+0x4e0; set by
    ResetTransparencySortSettings, cleared by the transparencySortMode setter),
    ``transparency_sort_mode`` (+0x4d0), ``transparency_sort_axis`` (+0x4d4),
    ``orthographic`` (+0x4fb).  graphics_settings keys:
    ``transparency_sort_mode`` (+0x108), ``transparency_sort_axis`` (+0x10c).
    """
    if camera['implicit_transparency_sort']:
        mode = int(graphics_settings['transparency_sort_mode'])
        axis = graphics_settings['transparency_sort_axis']
    else:
        mode = int(camera['transparency_sort_mode'])
        axis = camera['transparency_sort_axis']
    if mode == SORT_DEFAULT:
        mode = SORT_ORTHOGRAPHIC if camera['orthographic'] else SORT_PERSPECTIVE
    return mode, tuple(F(a) for a in axis)


def opaque_distance_bucket_enabled(camera, has_hidden_surface_removal_on_gpu):
    """DoForwardShaderRenderLoop 0x6c9a00..0x6c9a33: RenderObjectSorter byte +8.

    True when opaqueSortMode (Camera+0x4cc) is FrontToBack, or Default on a GPU
    without hidden-surface removal (GraphicsCaps+0x3a2).  The same byte is passed
    to the transparent sorter.
    """
    mode = int(camera['opaque_sort_mode'])
    if mode == OPAQUE_SORT_DEFAULT and not has_hidden_surface_removal_on_gpu:
        return True
    return mode == OPAQUE_SORT_FRONT_TO_BACK


def sort_distances(center, sort_fudge, ctx):
    """Render-object extraction 0x6b671c..0x6b68a0 -> (RO+0x20 distance, RO+0x24).

    ``ctx``: ``world_to_camera`` (16 floats column-major, ctx+0x748),
    ``position`` (camera Transform position, ctx+0x798), ``sort_mode`` and
    ``sort_axis`` from :func:`resolve_transparency_sort`.
    The opaque *and* transparent sorters both read RO+0x20.
    """
    x, y, z = (F(c) for c in center)
    fudge = F(sort_fudge)
    m = [F(v) for v in ctx['world_to_camera']]
    view_z = ((x * m[2] + y * m[6]) + z * m[10]) + m[14]
    mode = ctx['sort_mode']
    if mode == SORT_PERSPECTIVE:
        cx, cy, cz = (F(c) for c in ctx['position'])
        dy, dx, dz = y - cy, x - cx, z - cz
        sq = (dy * dy + dx * dx) + dz * dz
        if fudge == _ZERO:              # ucomiss fudge,0: equal and ordered
            dist = -sq
        else:
            d = F(np.sqrt(sq)) + fudge
            sign = F(-1.0) if _ZERO > d else _ONE
            dist = -((d * d) * sign)
    elif mode == SORT_ORTHOGRAPHIC:
        dist = view_z - fudge
    elif mode == SORT_CUSTOM_AXIS:
        ax, ay, az = ctx['sort_axis']
        dist = -(((y * F(ay) + x * F(ax)) + z * F(az)) + fudge)
    else:
        raise ValueError('sort mode %r leaves RO+0x20 undefined (0x6b673c)' % (mode,))
    return F(dist), F(-view_z)


def layering_key(sorting_layer_value, sorting_order):
    """Renderer::SetupSortingOverride 0x7400b0 -> GlobalLayeringData.layerAndOrder."""
    return _u32(((int(sorting_layer_value) + 0x8000) << 16) | (int(sorting_order) + 0x8000))


NO_SORTING_GROUP = 0xFFFFF000


def material_key(material_instance_id, custom_props_hash, enable_instancing):
    """RO+8 written at 0x6b6974..0x6b69ba.

    (SharedMaterialData+0xdc = Material instance ID) << 20 | (hash & 0xfffff);
    the hash part is omitted for materials with enableInstancing
    (SharedMaterialData+0xd8 bit 1, Material::SetEnableInstancingVariants).
    ``custom_props_hash`` is ShaderPropertySheet+0x88 of the renderer's
    MaterialPropertyBlock (0 when none).
    """
    hi = _u32(int(material_instance_id) << 20)
    if enable_instancing:
        return hi
    return _u32(hi | (int(custom_props_hash) & 0xFFFFF))


def extract_render_objects(nodes, ctx, replacement_shader=None):
    """Unnamed extraction routine 0x6b65c0 (called from DoRenderLoop 0x6b9b58).

    ``nodes`` is the RenderNodeQueue in index order.  Node keys:
      center (node+0x80 world AABB centre), sort_fudge (+0x10c, 0 for mesh
      renderers: FlattenBasicData 0x650b84), layering (+0xbc),
      sorting_group (+0xc0, NO_SORTING_GROUP if none), sorting_group_center and
      sorting_group_layering (SortingGroupData entry, needed only when a group is
      set), shadow_casting_mode (+0xe8 bits 6-7), lightmap_static / lightmap_realtime
      (+0xe4/+0xe6), mesh_id (+0x100 = Mesh+0x174), first_subset (+0xb8),
      static_batch_subset_count (+0xba, 0 when not static batched), priority
      (+0x122, 0 except UI canvas batches), custom_props_hash (single or a
      per-material list when FlattenPerMaterialCustomProps was used),
      materials: list of {shader_id, material_instance_id, enable_instancing,
      custom_render_queue (-1 = from shader), shader_render_queue (IntShader+0x4c),
      replacement_subshader / replacement_queue when ``replacement_shader``}.
    Returns (opaque_ros, transparent_ros) in append order (= RO index order).
    """
    opaque, transparent = [], []
    for ni, node in enumerate(nodes):
        if int(node.get('shadow_casting_mode', 0)) == SHADOWS_ONLY:
            continue
        center = node['center']
        layering = _u32(node['layering'])
        group = _u32(node.get('sorting_group', NO_SORTING_GROUP))
        if (group & 0xFFFFF000) != 0xFFFFF000 and node.get('sorting_group_center') is not None:
            center = node['sorting_group_center']
            layering = _u32(node['sorting_group_layering'])
        dist, along = sort_distances(center, node.get('sort_fudge', 0.0), ctx)
        count = int(node.get('static_batch_subset_count', 0))
        props = node.get('custom_props_hash', 0)
        for i, mat in enumerate(node['materials']):
            subshader = -1
            if replacement_shader is not None:
                subshader = int(mat['replacement_subshader'])
                if subshader < 0:
                    continue
            queue = int(mat.get('custom_render_queue', -1))
            if queue < 0:
                queue = int(mat['replacement_queue'] if replacement_shader is not None
                            else mat['shader_render_queue'])
            sub_i = i
            if count != 0 and i >= count:
                sub_i = count - 1
            subset = _u16(int(node.get('first_subset', 0)) + sub_i)
            h = props[i] if isinstance(props, (list, tuple)) else props
            instancing = bool(mat.get('enable_instancing', False))
            ro = {
                'shader_id': _s32(replacement_shader['shader_id'] if replacement_shader is not None
                                  else mat['shader_id']),
                'material_key': material_key(mat['material_instance_id'], h, instancing),
                'node': ni,
                'queue': _s16(queue),
                'subset': subset,
                'subshader': _s16(subshader),
                'material_index': i & 0x7FFF,
                'distance': dist,
                'distance_along_view': along,
                'layering': layering,
                'sorting_group': group,
                'instancing_key': _u32((int(node.get('mesh_id', 0)) << 16) | subset) if instancing else 0,
            }
            (opaque if queue <= QUEUE_SPLIT else transparent).append(ro)
    return opaque, transparent


def forward_pass_entries(ro_index, passes, *, vertex_lit_subshader, lightmapped, add_light_count,
                         render_options, no_add_pass_flag=None, multi_pass, grab_added=()):
    """ForwardRenderPassData for one RO (DoForwardShaderRenderLoop 0x6c9720..0x6c98b7).

    ``passes``: the selected subshader's passes in order, each {type (LightMode
    pass type, Pass+0x2f0), needs_render_option (Pass+0x3b0 bit 0), disabled
    (LightMode tag listed in the material's disabled passes), is_grab}.
    ``multi_pass`` is the kPackMultiPassFlag decision made by predicate 0x6cc280
    (not transcribed: it does not take part in ordering).  ``grab_added`` lists
    the pass indices for which GrabPasses::Add returned true.
    Returns entries {ro, data} in emission order.
    """
    out = []
    first = FIRST_PASS_FLAG
    for idx, p in enumerate(passes):
        t = int(p['type'])
        if (int(p.get('needs_render_option', 0)) & 1) and not (int(render_options) & 1):
            continue
        if vertex_lit_subshader:
            if t not in (PASS_NORMAL, PASS_VERTEX, PASS_VERTEX_LM):
                continue
            if t == PASS_VERTEX and lightmapped:
                continue
            if t == PASS_VERTEX_LM and not lightmapped:
                continue
        else:
            if t & 0xFFFFFFFA or t == PASS_VERTEX:
                continue
            if add_light_count == 0 and t == PASS_FORWARD_ADD:
                continue
        if p.get('disabled', False):
            continue
        data = (t << 8) | (idx & 0xFF) | (MULTI_PASS_FLAG if multi_pass else 0) | first
        if p.get('is_grab', False) and idx in grab_added:
            data |= GRAB_PASS_FLAG
        out.append({'ro': ro_index, 'data': data})
        first = 0
    return out


def sort_key(ro, node):
    """The fields RenderObjectSorter reads for one draw (no total-order tuple exists:
    the comparators contain data-dependent branches, see :func:`compare`)."""
    return {
        'layering': ro['layering'], 'queue': ro['queue'], 'distance': ro['distance'],
        'distance_bucket': f32_bits(ro['distance']) >> 24, 'sorting_group': ro['sorting_group'],
        'priority': _u16(node.get('priority', 0)), 'node': ro['node'],
        'material_index': ro['material_index'], 'static_batch': int(node.get('static_batch_subset_count', 0)) != 0,
        'mesh_id': _s32(node.get('mesh_id', 0)), 'first_subset': _u16(node.get('first_subset', 0)),
        'lightmap': _u32((_u16(node.get('lightmap_realtime', 0xFFFF)) << 16) | _u16(node.get('lightmap_static', 0xFFFF))),
        'shader_id': ro['shader_id'], 'material_key': ro['material_key'],
        'instancing_key': ro['instancing_key'],
    }


def _compare_static_batch(ka, kb):
    """CompareStaticBatch 0x6b7640 -> (decided, result)."""
    sa, sb = ka['static_batch'], kb['static_batch']
    if sa and sb:
        if ka['material_key'] != kb['material_key']:
            return True, ka['material_key'] < kb['material_key']
        if ka['mesh_id'] != kb['mesh_id']:
            return True, ka['mesh_id'] < kb['mesh_id']
        return True, ka['first_subset'] < kb['first_subset']
    if sa != sb:
        return True, sa and not sb
    return False, False


def _compare_sorting_group(ga, gb):
    """CompareSortingGroupData 0x6c87a0."""
    if (ga & 0xFFFFF000) == 0xFFFFF000 and (gb & 0xFFFFF000) == 0xFFFFF000:
        return 0
    return (ga > gb) - (ga < gb)


def compare(a, b, keys, opaque, bucket):
    """RenderObjectSorter<1> 0x6c6b30 (opaque) / <0> 0x6c6d00 (transparent): a < b."""
    ka, kb = keys[a['ro']], keys[b['ro']]
    if ka['layering'] != kb['layering']:
        return ka['layering'] < kb['layering']
    if ka['queue'] != kb['queue']:
        return ka['queue'] < kb['queue']
    if not opaque:
        da, db = ka['distance'], kb['distance']
        if not (da == db):                   # ucomiss; jp/jne
            return _gt(db, da)               # comiss b, a; seta  (back to front)
        c = _compare_sorting_group(ka['sorting_group'], kb['sorting_group'])
        if c != 0:
            return c < 0
        if ka['priority'] != kb['priority']:
            return ka['priority'] < kb['priority']
        if ka['node'] == kb['node'] and ka['material_key'] != kb['material_key']:
            if ka['material_index'] != kb['material_index']:
                return ka['material_index'] < kb['material_index']
    fa, fb = a['data'] & FIRST_PASS_FLAG, b['data'] & FIRST_PASS_FLAG
    if fa != fb:
        return fa > fb
    if bucket and ka['distance_bucket'] != kb['distance_bucket']:
        return ka['distance_bucket'] < kb['distance_bucket']
    decided, r = _compare_static_batch(ka, kb)
    if decided:
        return r
    if ka['lightmap'] != kb['lightmap']:
        return ka['lightmap'] < kb['lightmap']
    if ka['shader_id'] != kb['shader_id']:
        return ka['shader_id'] < kb['shader_id']
    if ka['material_key'] != kb['material_key']:
        return ka['material_key'] < kb['material_key']
    pa, pb = a['data'] & 0xFF, b['data'] & 0xFF
    if pa != pb:
        return pa < pb
    if opaque:
        if ka['instancing_key'] != kb['instancing_key']:
            return ka['instancing_key'] > kb['instancing_key']
        da, db = ka['distance'], kb['distance']
        if not (da == db):
            return _gt(da, db)               # front to back
        if ka['priority'] != kb['priority']:
            return ka['priority'] > kb['priority']
    return _s32(a['ro']) < _s32(b['ro'])


# ---- qsort_internal (Unity) ------------------------------------------------
def _unity_find_pivot(arr, lo, last, n1, less):
    """FindAndMovePivotToLastPosition 0x6c2110 (n1 = count-1)."""
    def cswap(i, j):          # if less(arr[i], arr[j]) swap(i, j)
        if less(arr[i], arr[j]):
            arr[i], arr[j] = arr[j], arr[i]
    mid = lo + (n1 >> 1)
    if n1 <= 64:
        cswap(mid, lo)
        cswap(last, mid)
        cswap(mid, lo)
    else:
        step = n1 >> 3
        two = step * 2
        cswap(lo + step, lo); cswap(lo + two, lo + step); cswap(lo + step, lo)
        cswap(mid, mid - step); cswap(mid + step, mid); cswap(mid, mid - step)
        cswap(last - step, last - two); cswap(last, last - step); cswap(last - step, last - two)
        cswap(mid, lo + step); cswap(last - step, mid); cswap(mid, lo + step)
    arr[mid], arr[last] = arr[last], arr[mid]


def _heap_pop_hole(arr, base, hole, bottom, val, less):
    """std::_Pop_heap_hole_by_index (0x6c6020) incl. the inlined push."""
    top = hole
    idx = hole
    max_non_leaf = (bottom - 1) >> 1
    while idx < max_non_leaf:
        idx = 2 * idx + 2
        if less(arr[base + idx], arr[base + idx - 1]):
            idx -= 1
        arr[base + hole] = arr[base + idx]
        hole = idx
    if idx == max_non_leaf and bottom % 2 == 0:
        arr[base + hole] = arr[base + bottom - 1]
        hole = bottom - 1
    while top < hole:
        parent = (hole - 1) >> 1
        if not less(arr[base + parent], val):
            break
        arr[base + hole] = arr[base + parent]
        hole = parent
    arr[base + hole] = val


def _heap_sort(arr, lo, hi, less):
    n = hi - lo
    h = n >> 1
    while h > 0:
        h -= 1
        _heap_pop_hole(arr, lo, h, n, arr[lo + h], less)
    while n >= 2:
        val = arr[lo + n - 1]
        arr[lo + n - 1] = arr[lo]
        _heap_pop_hole(arr, lo, 0, n - 1, val, less)
        n -= 1


def _unity_insertion(arr, lo, hi, less):
    """InsertionSort 0x6c2670 (adjacent swaps)."""
    for i in range(lo + 1, hi):
        j = i
        while j > lo and less(arr[j], arr[j - 1]):
            arr[j], arr[j - 1] = arr[j - 1], arr[j]
            j -= 1


def unity_qsort(arr, less, lo=0, hi=None, depth=None):
    """qsort_internal::QSort 0x6c4750/0x6c4b40; SortJob passes depth = count."""
    if hi is None:
        hi = len(arr)
    if depth is None:
        depth = hi - lo
    count = hi - lo
    while count >= 32:
        if depth <= 0:
            _heap_sort(arr, lo, hi, less)
            return arr
        last = hi - 1
        _unity_find_pivot(arr, lo, last, count - 1, less)
        i, j = -1, count - 1
        while True:
            i += 1
            if less(arr[lo + i], arr[last]):
                while i != count - 1:
                    i += 1
                    if not less(arr[lo + i], arr[last]):
                        break
            j -= 1
            if less(arr[last], arr[lo + j]):
                while j != 0:
                    j -= 1
                    if not less(arr[last], arr[lo + j]):
                        break
            if i < j:
                arr[lo + i], arr[lo + j] = arr[lo + j], arr[lo + i]
                continue
            break
        arr[lo + i], arr[last] = arr[last], arr[lo + i]
        left, right = i, count - (i + 1)
        depth = depth // 2 + depth // 4
        if left < right:
            unity_qsort(arr, less, lo, lo + i, left)
            lo = lo + i + 1
        else:
            unity_qsort(arr, less, lo + i + 1, hi, right)
            hi = lo + i
        count = hi - lo
    if count > 1:
        _unity_insertion(arr, lo, hi, less)
    return arr


def sort_draws(draws, camera, ros, nodes, *, opaque, has_hidden_surface_removal_on_gpu=False):
    """Order ForwardRenderPassData like DoForwardShaderRenderLoop's sort job.

    ``draws``: entries from :func:`forward_pass_entries` in emission order
    (RO order, then pass order).  Returns a new list in draw order.
    """
    bucket = opaque_distance_bucket_enabled(camera, has_hidden_surface_removal_on_gpu)
    keys = [sort_key(ro, nodes[ro['node']]) for ro in ros]
    arr = list(draws)
    unity_qsort(arr, lambda a, b: compare(a, b, keys, opaque, bucket))
    return arr


# =========================================================================
# 2. Forward light classification
# =========================================================================
def light_luminance_intensity(color, intensity):
    """ActiveLight+0xc4 (0x643f71 directional, 0x644164 local)."""
    r, g, b = F(color[0]), F(color[1]), F(color[2])
    return ((g * K_LUM_G + r * K_LUM_R) + b * K_LUM_B) * F(intensity)


def find_main_directional_light(dir_lights):
    """0x641680 (called by AddDirectionalLights 0x640170): index or -1."""
    best, best_i = F(-1.0), -1
    for i, l in enumerate(dir_lights):
        v = light_luminance_intensity(l['color'], l['intensity'])
        if int(l.get('shadows', 0)) != 0:
            v = v * K_SHADOW_DIR_BOOST
        if int(l['render_mode']) != RENDER_MODE_FORCE_VERTEX and _gt(v, best):
            best, best_i = v, i
    return best_i


def order_directional_lights(dir_lights):
    """AddDirectionalLights 0x640170: main first (hasMainLight), rest in order."""
    m = find_main_directional_light(dir_lights)
    if m < 0:
        return list(dir_lights), False
    return [dir_lights[m]] + [l for i, l in enumerate(dir_lights) if i != m], True


def _attenuate_approx(d2, r2):
    """LightAttenuateApprox 0x67ea80: 1 / (1 + (25 / r2) * d2)."""
    return _ONE / ((K_ATTEN / F(r2)) * F(d2) + _ONE)


def _light_range(l):
    t = int(l['type'])
    return F(l['area_range'] if t in (LIGHT_RECTANGLE, LIGHT_DISC) else l['range'])


def light_importance(light, center):
    """CulledLight.key computed in 0x6aa330 for one light and node+0x80 centre."""
    base = F(light['luminance_intensity'])
    if int(light['type']) == LIGHT_DIRECTIONAL:
        imp = base * K_SHADOW_DIR_BOOST if int(light.get('shadows', 0)) != 0 else base
    else:
        cx, cy, cz = (F(c) for c in center)
        px, py, pz = (F(c) for c in light['position'])
        dx, dy, dz = cx - px, cy - py, cz - pz
        d2 = (dx * dx + dy * dy) + dz * dz
        r = _light_range(light)
        imp = _attenuate_approx(d2, r * r) * base
    return RENDER_MODE_SORT_BIAS[int(light['render_mode'])] + imp


def intersect_aabb_aabb(a_center, a_extent, b_center, b_extent):
    """IntersectAABBAABB 0x6ea760 (strict on every axis)."""
    for k in range(3):
        diff = abs(F(b_center[k]) - F(a_center[k]))
        if not _gt(F(a_extent[k]) + F(b_extent[k]), diff):
            return False
    return True


def _hsum3_lane0(a, b, c):
    """pshufd 0x39 / 0x4e horizontal sum with w=0, lane 0: (a+b)+c."""
    return (a + b) + c


def _hsum3_lane1(a, b, c):
    return a + (b + c)


def intersect_aabb_sphere(center, extent, sphere_center, radius):
    """IntersectAABBSphere 0x6eaca0 (mask 0x1543ef0 = xyz lanes, see spec)."""
    q = []
    for k in range(3):
        d = abs(F(sphere_center[k]) - F(center[k]))
        e = F(extent[k])
        q.append(_max_ss(d, e) - e)
    s = _hsum3_lane0(q[0] * q[0], q[1] * q[1], q[2] * q[2])
    return bool(s < F(radius) * F(radius))


def intersect_transformed_aabb_sphere(local_center, local_extent, m, transform_type, sphere_center, radius):
    """IntersectTransformedAABBSphere 0x6eb430, uniform-scale branch (type bit 1 clear)."""
    if (int(transform_type) >> 1) & 1:
        raise NotImplementedError('gap: IntersectTransformedAABBSphere non-uniform branch 0x6eb5ef')
    m = [F(v) for v in m]
    if int(transform_type) == 0:
        inv = _ONE
    else:
        inv = _ONE / F(np.sqrt((m[1] * m[1] + m[0] * m[0]) + m[2] * m[2]))
    sx, sy, sz = (F(c) for c in sphere_center)
    dx, dy, dz = sx - m[12], sy - m[13], sz - m[14]
    lx = ((m[1] * dy + m[0] * dx) + m[2] * dz) * inv * inv
    ly = ((dy * m[5] + dx * m[4]) + dz * m[6]) * inv * inv
    lz = ((dy * m[9] + dx * m[8]) + dz * m[10]) * inv * inv
    r = inv * F(radius)
    return intersect_aabb_sphere(local_center, local_extent, (lx, ly, lz), r)


def spot_light_rejects_object(light, local_center, local_extent, object_to_world, aabb_frustum_test):
    """0x6abc20: True when the object's local AABB is outside the spot frustum.

    Builds P = SetPerspectiveCotan(cotan=SharedLightData+0x3c,
    near=max(range*1e-5, 1e-4), far=max(range, near+1e-4)), M = P*(S(1,1,-1)*
    worldToLight*objectToWorld), extracts planes and tests the local AABB.
    ``aabb_frustum_test(center, extent, matrix)`` must implement
    ExtractProjectionPlanes 0x68b480 + IntersectAABBFrustumFull 0x6ea930.
    """
    from .unity_math import multiply, set_scale
    r = _light_range(light)
    near = _max_ss(r * _fbits(0x3727C5AC), _fbits(0x38D1B717))
    far = _max_ss(r, near + _fbits(0x38D1B717))
    cot = F(light['spot_cotan'])
    p = [_ZERO] * 16
    p[0] = cot
    p[5] = cot
    p[10] = (near + far) / (near - far)
    p[14] = ((near + near) * far) / (near - far)
    p[11] = F(-1.0)
    m1 = multiply(set_scale(1.0, 1.0, -1.0), [F(v) for v in light['world_to_light']])
    m2 = multiply(m1, [F(v) for v in object_to_world])
    return not aabb_frustum_test(local_center, local_extent, multiply(p, m2))


def object_needs_forward_lights(node, rendering_path, max_caps_value, main_light_lightmap_mode_is_one):
    """Per-node bitset of CullAllPerObjectLights 0x6a95c0..0x6a970b."""
    if rendering_path < 2:
        return True
    lm = _u16(node.get('lightmap_static', 0xFFFF))
    if max_caps_value <= 4 and lm != 0xFFFF:
        return True
    if main_light_lightmap_mode_is_one and (lm & 0xFFFE) != 0xFFFE:
        return True
    if int(node.get('light_probe_usage', 0)) != 0:
        return True
    for mat in node['materials']:
        q = int(mat.get('custom_render_queue', -1))
        if q < 0:
            q = int(mat['shader_render_queue'])
        if q > QUEUE_SPLIT:
            return True
        if not mat['has_deferred_subshader'][rendering_path]:
            return True
    return False


def cull_lights_for_object(active, node, aabb_frustum_test=None):
    """0x6aa330: per-object CulledLight list, sorted by MSVC std::sort (key desc).

    ``active``: {'lights': ActiveLights order, 'counts': (dir, spot, point,
    offscreen_spot, offscreen_point)} (ActiveLights+0x28..+0x48).  Light keys:
    type, render_mode, culling_mask, lightmap_mode_for_render (+0x24),
    luminance_intensity, shadows, position, range/area_range, bounds_center /
    bounds_extent (+0x30, spot), world_to_light / spot_cotan (spot), sphere is
    (position, range) (+0xb4).
    Node keys: layer, lightmap_static, center/extent (node+0x80), local_center /
    local_extent (+0x98), object_to_world (+0), transform_type (+0xb0).
    Returns (sorted [(light_index, key)], has_offscreen).
    """
    layer_bit = 1 << int(node['layer'])
    lightmapped = (_u16(node.get('lightmap_static', 0xFFFF)) & 0xFFFE) != 0xFFFE
    lights = active['lights']
    counts = active['counts']
    out = []
    groups = ('dir', 'spot', 'point', 'spot', 'point')
    size_before_offscreen = 0
    start = 0
    for gi, (g, c) in enumerate(zip(groups, counts)):
        end = start + int(c)
        if gi == 3:
            size_before_offscreen = len(out)      # [rsp+0x30] at 0x6aa880
        for li in range(start, end):
            l = lights[li]
            if lightmapped and int(l['lightmap_mode_for_render']) != 0:
                continue
            if not (int(l['culling_mask']) & layer_bit):
                continue
            if g == 'spot':
                if not intersect_aabb_aabb(node['center'], node['extent'], l['bounds_center'], l['bounds_extent']):
                    continue
                if spot_light_rejects_object(l, node['local_center'], node['local_extent'],
                                             node['object_to_world'],
                                             aabb_frustum_test or aabb_in_projection_frustum):
                    continue
            elif g == 'point':
                if not intersect_aabb_sphere(node['center'], node['extent'], l['position'], _light_range(l)):
                    continue
                if not intersect_transformed_aabb_sphere(node['local_center'], node['local_extent'],
                                                         node['object_to_world'], node['transform_type'],
                                                         l['position'], _light_range(l)):
                    continue
            out.append((li, light_importance(l, node['center'])))
        start = end
    if len(counts) < 4:
        size_before_offscreen = len(out)
    has_offscreen = len(out) != size_before_offscreen
    if out:
        msvc_sort(out, lambda a, b: _gt(a[1], b[1]))
    return out, has_offscreen


# ---- MSVC std::sort (VS2017 STL), as compiled at 0x6a8910 -----------------
def _msvc_med3(a, f, m, l, less):
    if less(a[m], a[f]):
        a[m], a[f] = a[f], a[m]
    if less(a[l], a[m]):
        a[l], a[m] = a[m], a[l]
        if less(a[m], a[f]):
            a[m], a[f] = a[f], a[m]


def _msvc_guess_median(a, f, m, l, less):
    count = l - f
    if count > 40:
        step = (count + 1) >> 3
        two = step << 1
        _msvc_med3(a, f, f + step, f + two, less)
        _msvc_med3(a, m - step, m, m + step, less)
        _msvc_med3(a, l - two, l - step, l, less)
        _msvc_med3(a, f + step, m, l - step, less)
    else:
        _msvc_med3(a, f, m, l, less)


def _msvc_partition(a, first, last, less):
    mid = first + ((last - first) >> 1)
    _msvc_guess_median(a, first, mid, last - 1, less)
    pf, pl = mid, mid + 1
    while first < pf and not less(a[pf - 1], a[pf]) and not less(a[pf], a[pf - 1]):
        pf -= 1
    while pl < last and not less(a[pl], a[pf]) and not less(a[pf], a[pl]):
        pl += 1
    gf, gl = pl, pf
    while True:
        while gf < last:
            if less(a[pf], a[gf]):
                pass
            elif less(a[gf], a[pf]):
                break
            elif pl != gf:
                a[pl], a[gf] = a[gf], a[pl]
                pl += 1
            else:
                pl += 1
            gf += 1
        while first < gl:
            if less(a[gl - 1], a[pf]):
                pass
            elif less(a[pf], a[gl - 1]):
                break
            else:
                pf -= 1
                if pf != gl - 1:
                    a[pf], a[gl - 1] = a[gl - 1], a[pf]
            gl -= 1
        if gl == first and gf == last:
            return pf, pl
        if gl == first:
            if pl != gf:
                a[pf], a[pl] = a[pl], a[pf]
            pl += 1
            a[pf], a[gf] = a[gf], a[pf]
            pf += 1
            gf += 1
        elif gf == last:
            gl -= 1
            pf -= 1
            if gl != pf:
                a[gl], a[pf] = a[pf], a[gl]
            pl -= 1
            a[pf], a[pl] = a[pl], a[pf]
        else:
            gl -= 1
            a[gf], a[gl] = a[gl], a[gf]
            gf += 1


def _msvc_insertion(a, first, last, less):
    for nxt in range(first + 1, last):
        val = a[nxt]
        if less(val, a[first]):
            a[first + 1:nxt + 1] = a[first:nxt]
            a[first] = val
        else:
            hole = nxt
            while less(val, a[hole - 1]):
                a[hole] = a[hole - 1]
                hole -= 1
            a[hole] = val


def msvc_sort(a, less, first=0, last=None, ideal=None):
    """std::_Sort_unchecked (ISORT_MAX 32, ideal = count, heap fallback)."""
    if last is None:
        last = len(a)
    if ideal is None:
        ideal = last - first
    while True:
        if last - first <= 32:
            if last - first >= 2:
                _msvc_insertion(a, first, last, less)
            return a
        if ideal <= 0:
            _heap_sort(a, first, last, less)
            return a
        mf, ms = _msvc_partition(a, first, last, less)
        ideal = (ideal >> 1) + (ideal >> 2)
        if mf - first < last - ms:
            msvc_sort(a, less, first, mf, ideal)
            first = ms
        else:
            msvc_sort(a, less, ms, last, ideal)
            last = mf


# ---- SH projection ---------------------------------------------------------
def _sh_norm_constants():
    """Dynamic initializer 0x83260 of SphericalHarmonicsL2::kNormalizationConstants."""
    sp = F(np.sqrt(F(3.14159274)))
    k0 = _ONE / (sp + sp)
    k2 = F(np.sqrt(F(3.0))) / (sp * F(3.0))
    k4 = F(np.sqrt(F(15.0))) / (sp * F(8.0))
    k6 = F(np.sqrt(F(5.0))) / (sp * F(16.0))
    return (k0, -k2, k2, -k2, k4, -k4, k6, -k4, k4 * F(0.5))


SH_NORM = _sh_norm_constants()
_SH_C0 = _fbits(0x403D3C19)
_SH_A0, _SH_A1 = _fbits(0x3E906EBB), _fbits(0x3EFA2A1C)
_SH_B0, _SH_B2 = _fbits(0x3F8BD8A1), _fbits(0x3F723881)
_SH_C1 = _fbits(0x3F0BD8A1)
_THIRD = _fbits(0x3EAAAAAB)


def sh_add_directional_light(sh, direction, color):
    """SphericalHarmonicsL2::AddDirectionalLight 0x7d3950 (sh: 27 floats, [ch*9+k])."""
    x, y, z = (F(c) for c in direction[:3])
    nx, ny, nz = -x, -y, -z
    k = SH_NORM
    v = [
        (_ONE * _SH_A0) * k[0],
        (ny * _SH_A1) * k[1],
        (z * _SH_A1) * k[2],
        (nx * _SH_A1) * k[3],
        ((x * y) * _SH_B0) * k[4],
        ((z * ny) * _SH_B0) * k[5],
        ((z * z - _THIRD) * _SH_B2) * k[6],
        ((nx * z) * _SH_B0) * k[7],
        ((x * x - y * y) * _SH_C1) * k[8],
    ]
    out = [F(s) for s in sh]
    for c in range(3):
        cc = _SH_C0 * F(color[c])
        for j in range(9):
            out[c * 9 + j] = cc * v[j] + out[c * 9 + j]
    return out


def light_color_for_sh(light, center, extent):
    """0x678ae0: SharedLightData+0x110 times the box-aware attenuation, per lane."""
    col = [F(c) for c in light['final_color']]
    if int(light['type']) == LIGHT_DIRECTIONAL:
        return col
    ex, ey, ez = (F(e) for e in extent)
    e2 = (_hsum3_lane0(ex * ex, ey * ey, ez * ez), _hsum3_lane1(ex * ex, ey * ey, ez * ez))
    r = _light_range(light)
    r2 = r * r
    px, py, pz = (F(p) for p in light['position'])
    cx, cy, cz = (F(c) for c in center)
    dx, dy, dz = cx - px, cy - py, cz - pz
    d2 = (_hsum3_lane0(dx * dx, dy * dy, dz * dz), _hsum3_lane1(dx * dx, dy * dy, dz * dz))
    k = K_ATTEN / r2
    shrink = bool(r2 < e2[0])
    out = []
    for lane in range(4):
        s, e = d2[lane & 1], e2[lane & 1]
        att = _ONE / (_max_ss(s, e) * k + _ONE)
        if shrink:
            att = att * (r2 / e)
        out.append(col[lane] * att)
    return out


def _rsqrtps(x, rsqrt_estimate):
    if rsqrt_estimate is None:
        raise NotImplementedError('gap rsqrtps: hardware-defined estimate (AddLightToSH 0x677c17); pass rsqrt_estimate')
    return F(rsqrt_estimate(F(x)))


def sh_light_direction(light, center, rsqrt_estimate=None):
    """Direction argument built by 0x677b80 before AddDirectionalLight."""
    if int(light['type']) == LIGHT_DIRECTIONAL:
        fx, fy, fz = (F(c) for c in light['forward'])
        return (-fx, -fy, -fz)
    px, py, pz = (F(p) for p in light['position'])
    cx, cy, cz = (F(c) for c in center)
    d = (px - cx, py - cy, pz - cz)
    s = (_hsum3_lane0(d[0] * d[0], d[1] * d[1], d[2] * d[2]), _hsum3_lane1(d[0] * d[0], d[1] * d[1], d[2] * d[2]))
    out = []
    for lane in range(3):
        sv = s[lane & 1]
        y = _min_ss(_fbits(0x3F7FFFFD) * _rsqrtps(sv, rsqrt_estimate), _fbits(0x7F7FFFFF))
        rr = (_fbits(0x40400001) - (y * sv) * y) * (_fbits(0x3F000000) * y)
        out.append(rr * d[lane] if _gt(sv, _fbits(0x0DA24260)) else _ZERO)
    return tuple(out)


def add_light_to_sh(sh, light, center, extent, weight, rsqrt_estimate=None):
    """0x677b80: SH += weight * light colour along the light direction."""
    direction = sh_light_direction(light, center, rsqrt_estimate)
    color = [F(weight) * c for c in light_color_for_sh(light, center, extent)]
    return sh_add_directional_light(sh, direction, color)


# ---- FindForwardLightsForObject --------------------------------------------
def _blend(keys, i):
    """Shared formula of 0x67a937..0x67a983 / 0x67aa7d..0x67aac0."""
    k_next = F(keys[i + 1])
    span = F(keys[i - 1]) - k_next
    if not (span < K_BLEND_MAX):                  # comiss span, 1000; jae skip (NaN proceeds)
        if span == span:
            return None
    return _clamp01_ss((F(keys[i]) - k_next) / (span + K_BLEND_EPS))


def classify_lights(obj, lights, quality, rsqrt_estimate=None):
    """FindForwardLightsForObject 0x67c440 (+ last-light blend 0x67a8d0).

    obj keys: culled (list of (light_index, key) from cull_lights_for_object),
      has_offscreen, layer, center, extent (node+0x80), flags dict
      {lightmapped, vertex_lit_subshader, coefficient_type, no_add_pass,
      no_lights}, ambient_sh (27 floats, LightProbeContext ambient),
      probe_sh (27 floats, for coefficient types 2..4), occlusion
      (callable(light_index)->float, only for types >1 with probe occlusion).
    lights: {'lights': ActiveLights list, 'has_main_light': bool}.
    quality: {'pixel_light_count', 'rendering_path'}.
    Returns the ForwardLightsBlock as a dict.
    """
    fl = obj['flags']
    act = lights['lights']
    culled = list(obj['culled'])
    pixel_count = int(quality['pixel_light_count']) if int(quality['rendering_path']) != 0 else 0
    if not fl['vertex_lit_subshader'] and obj.get('has_offscreen', False):
        culled = [c for c in culled if not act[c[0]].get('offscreen', False)]
    block = {'main': None, 'add': [], 'vertex': [], 'main_occlusion': _ONE,
             'last_add_light_blend': _ONE, 'last_vertex_light_blend': _ONE,
             'coefficient_type': int(fl['coefficient_type']), 'sh': [_ZERO] * 27}
    if fl['vertex_lit_subshader']:
        # 0x67c5b2..0x67c63d: up to 8 lights in list order, then jump to the SH
        # output at 0x67c9c5 with the SH still zero (SetZero 0x67c5a1).
        block['vertex'] = [c[0] for c in culled[:8]]
        if block['coefficient_type'] == 5:
            block['sh_source'] = 'custom provided (ExtractSHCoefficientsFromPropertySheet 0x7d7560)'
        _occlusion(block, obj)
        return block
    if lights.get('has_main_light') and act:
        l0 = act[0]
        if (int(l0['culling_mask']) & (1 << int(obj['layer']))) and not l0.get('has_cookie', False):
            block['main'] = 0
    sh = [F(v) for v in obj['ambient_sh']] if block['coefficient_type'] <= 1 else [_ZERO] * 27
    count = 0 if fl['no_lights'] else len(culled)
    exhaust = -1
    keys = [c[1] for c in culled]
    for i in range(count):
        li = culled[i][0]
        l = act[li]
        mode = int(l['render_mode'])
        if block['main'] is None:
            if int(l['type']) == LIGHT_DIRECTIONAL and mode != RENDER_MODE_FORCE_VERTEX and not l.get('has_cookie', False):
                block['main'] = li
                continue
        elif li == 0:
            continue
        if mode == RENDER_MODE_FORCE_PIXEL:
            pixel = True
        elif mode == RENDER_MODE_FORCE_VERTEX:
            pixel = False
        else:
            pixel = len(block['add']) < pixel_count
        if pixel and not fl['no_add_pass']:
            block['add'].append(li)
            if len(block['add']) == pixel_count and mode != RENDER_MODE_FORCE_PIXEL:
                exhaust = i
            continue
        if fl['lightmapped']:
            continue
        if int(l['type']) == LIGHT_DIRECTIONAL or len(block['vertex']) >= 4:
            sh = add_light_to_sh(sh, l, obj['center'], obj['extent'], _ONE, rsqrt_estimate)
        else:
            block['vertex'].append(li)
    if not fl['no_lights']:
        sh = _last_light_blend(block, culled, keys, count, exhaust, act, obj, sh, rsqrt_estimate)
    ct = block['coefficient_type']
    if 2 <= ct <= 4 and not obj.get('lppv_sample_invalid', False):
        if obj.get('probe_sh') is None:
            raise NotImplementedError('probe SH input required (CalculateLightProbeCoefficients 0x6b6e10)')
        sh = [a + F(b) for a, b in zip(sh, obj['probe_sh'])]
    block['sh'] = sh
    if ct == 5:
        block['sh_source'] = 'custom provided (ExtractSHCoefficientsFromPropertySheet 0x7d7560); accumulated SH discarded'
    _occlusion(block, obj)
    return block


def _last_light_blend(block, culled, keys, count, exhaust, act, obj, sh, rsqrt_estimate):
    """0x67a8d0."""
    add, vert = block['add'], block['vertex']
    vidx = (len(vert) - 1) + len(add)
    if 0 < exhaust < count - 1:
        b = _blend(keys, exhaust)
        if b is not None:
            block['last_add_light_blend'] = b
            if not obj['flags']['lightmapped']:
                l = act[culled[exhaust][0]]
                if int(l['type']) == LIGHT_DIRECTIONAL:
                    sh = add_light_to_sh(sh, l, obj['center'], obj['extent'], _ONE - b, rsqrt_estimate)
                else:
                    vert.insert(0, add[-1])     # duplicates pointer slot add-1 (see spec)
                    if len(vert) > 4:
                        del vert[4:]
                        vidx -= 1
    if len(vert) > 0:
        block['last_vertex_light_blend'] = _ONE
        if 0 < vidx < count - 1:
            b = _blend(keys, vidx)
            if b is not None:
                block['last_vertex_light_blend'] = b
    return sh


def _occlusion(block, obj):
    """Per-light occlusion floats 0x67ca13..0x67ce55 (1.0 unless probe occlusion)."""
    ids = block['add'] + block['vertex']
    if block['coefficient_type'] <= 1 or not obj['flags'].get('probe_occlusion', False):
        block['occlusion'] = [_ONE] * len(ids)
        return
    occ = obj.get('occlusion')
    fade = F(obj.get('shadow_fade', 0.0))
    if occ is None:
        raise NotImplementedError('gap: CalculateLightOcclusion 0x6b6b30 inputs required')
    block['main_occlusion'] = _clamp01_ss(F(occ(block['main'])) + fade)
    block['occlusion'] = [_clamp01_ss(F(occ(i)) + fade) for i in ids]


def forward_base_vertex_constants(block, lights):
    """SetupForwardBaseLights 0x680cc0: unity_4LightPos*, unity_4LightAtten0, unity_LightColor[0..3].

    Light keys: position (+0x170), range/area_range, vertex_color (+0xe8, the
    GfxVertexLight colour from CalculateVertexLightData 0x6795c0).
    """
    act = lights['lights']
    vert = block['vertex']
    n = len(vert)
    pos = [[_ZERO] * 4 for _ in range(3)]
    atten = [_ONE] * 4
    color = [[_ZERO] * 4 for _ in range(4)]
    for j in range(n):
        l = act[vert[j]]
        f = _ONE
        lab = F(block['last_add_light_blend'])
        if j == 0 and not (lab == _ONE):
            f = _ONE - lab
        elif j == n - 1:
            f = F(block['last_vertex_light_blend'])
        f = f * F(block['occlusion'][len(block['add']) + j])
        color[j] = [f * F(c) for c in l['vertex_color']]
        r = _light_range(l)
        atten[j] = K_ATTEN / (r * r)
        for a in range(3):
            pos[a][j] = F(l['position'][a])
    return {'unity_4LightPosX0': pos[0], 'unity_4LightPosY0': pos[1], 'unity_4LightPosZ0': pos[2],
            'unity_4LightAtten0': atten, 'unity_LightColor': color, 'VERTEXLIGHT_ON': n != 0}


# =========================================================================
# 3. Camera culling (non-Umbra CustomCull path)
# =========================================================================
EPS_V3 = _fbits(0x3727C5AC)          # Vector3f::epsilon (.rdata 0x1257060) = 1e-5
SCENE_MASK_DEFAULT = 0x8000000000000000
FLAG_SHADOWS_ONLY = 0xC0000000       # SceneNode+0x1c bits 30-31 == 3
FLAG_PENDING_REMOVAL = 0x20000000    # SceneNode+0x1c bit 29
LOD_INDEX_MASK = 0x0FFFFFFF


def _sqrtf(x):
    return F(np.sqrt(F(x)))


def extract_projection_planes(m):
    """ExtractProjectionPlanes 0x68dc40 -> Helper<0>::Apply 0x68b480.

    Planes [a,b,c,d] in order left, right, bottom, top, near, far; a point is
    inside when a*x+b*y+c*z+d >= 0.  No zero-length guard.
    """
    m = [F(v) for v in m]
    row = [[m[0 + r], m[4 + r], m[8 + r], m[12 + r]] for r in range(4)]
    raw = [[row[0][k] + row[3][k] for k in range(4)], [row[3][k] - row[0][k] for k in range(4)],
           [row[1][k] + row[3][k] for k in range(4)], [row[3][k] - row[1][k] for k in range(4)],
           [row[2][k] + row[3][k] for k in range(4)], [row[3][k] - row[2][k] for k in range(4)]]
    out = []
    for a, b, c, d in raw:
        inv = _ONE / _sqrtf((a * a + b * b) + c * c)
        out.append([a * inv, inv * b, inv * c, inv * d])
    return out


def calculate_culling_parameters(camera):
    """Camera::CalculateCullingParameters 0x65e1e0 + CalculateFarCullDistances 0x65e5e0.

    camera keys: world_to_camera, projection (16 floats column-major), position
    (Transform position, CP+0x194), near (+0x43c), far (+0x440), culling_mask
    (+0x414), layer_cull_distances[32] (+0x448), layer_cull_spherical (+0x4fd),
    implicit_culling_matrix (+0x4f9), culling_matrix (+0x2e0, when not implicit),
    implicit_world_to_camera (+0x4ea).
    """
    from .unity_math import invert_full, multiply
    near, far = F(camera['near']), F(camera['far'])
    if camera['implicit_culling_matrix']:
        cm = multiply([F(v) for v in camera['projection']], [F(v) for v in camera['world_to_camera']])
    else:
        cm = [F(v) for v in camera['culling_matrix']]
    planes = extract_projection_planes(cm)
    if camera['implicit_world_to_camera']:
        _, inv = invert_full([F(v) for v in camera['world_to_camera']])
        cx, cy, cz = inv[8], inv[9], inv[10]
        px, py, pz = inv[12], inv[13], inv[14]
        ln = _sqrtf((cy * cy + cx * cx) + cz * cz)
        if ln > EPS_V3:
            nx, ny, nz = cx / ln, cy / ln, cz / ln
        else:
            nx = ny = nz = _ZERO
        fx, fy, fz = -nx, -ny, -nz
        dot = ((fy * py) + (fx * px)) + (fz * pz)
        planes[4] = [fx, fy, fz, (-dot) - near]
        base = -((((-fy) * py) + ((-fx) * px)) + ((-fz) * pz))
        planes[5] = [-fx, -fy, -fz, base + far]
    else:
        base = planes[5][3] - far
    lcd = [F(v) for v in camera['layer_cull_distances']]
    if camera['layer_cull_spherical']:
        dist, mode = list(lcd), 2
    else:
        dist, mode = [base + (d if not (d == _ZERO) else far) for d in lcd], 1
    return {'planes': planes, 'culling_mask': int(camera['culling_mask']) & 0xFFFFFFFF,
            'scene_mask': SCENE_MASK_DEFAULT, 'layer_far': dist, 'layer_mode': mode,
            'position': [F(v) for v in camera['position']], 'culling_matrix': cm}


def aabb_visible(center, extent, planes):
    """CullObjectsWithoutUmbra 0x6aa260 -> IntersectAABBPlaneBoundsOptimized 0x6eab70.

    Per plane dist = ((a*cx + d) + b*cy) + c*cz, rad = (|b|*ey + |a|*ex) + |c|*ez;
    rejected iff (rad + dist) < 0 (strict, no epsilon, NaN never rejects).
    """
    cx, cy, cz = (F(v) for v in center)
    ex, ey, ez = (F(v) for v in extent)
    for a, b, c, d in planes:
        dist = ((a * cx + d) + b * cy) + c * cz
        rad = (abs(b) * ey + abs(a) * ex) + abs(c) * ez
        if (rad + dist) < _ZERO:
            return False
    return True


def intersect_aabb_plane_bounds(center, extent, planes):
    """IntersectAABBPlaneBounds 0x6eaa50 and IntersectAABBFrustumFull 0x6ea930 (6 planes).

    dist = ((a*cx + b*cy) + c*cz) + d; rejected iff 0 > rad + dist.
    """
    cx, cy, cz = (F(v) for v in center)
    ex, ey, ez = (F(v) for v in extent)
    for a, b, c, d in planes:
        dist = ((a * cx + b * cy) + c * cz) + d
        rad = (abs(b) * ey + abs(a) * ex) + abs(c) * ez
        if _ZERO > (rad + dist):
            return False
    return True


def aabb_in_projection_frustum(center, extent, matrix):
    """ExtractProjectionPlanes + IntersectAABBFrustumFull, as used by the spot test 0x6abc20."""
    return intersect_aabb_plane_bounds(center, extent, extract_projection_planes(matrix))


def layer_cull_distance_test(center, extent, layer, cp):
    """ProcessIndexListIsNodeVisibleStep2 0x6ac180. True = keep."""
    if cp['layer_mode'] == 1:
        a, b, c, _ = cp['planes'][5]
        return intersect_aabb_plane_bounds(center, extent, [[a, b, c, cp['layer_far'][layer]]])
    if cp['layer_mode'] == 2:
        lcd = cp['layer_far'][layer]
        if lcd == _ZERO:
            return True
        px, py, pz = cp['position']
        cx, cy, cz = (F(v) for v in center)
        dy, dx, dz = cy - py, cx - px, cz - pz
        return not ((((dy * dy) + (dx * dx)) + (dz * dz)) > lcd * lcd)
    return True


def node_passes_step1(r, cp, lod_masks=None):
    """ProcessCameraIndexListIsNodeVisibleStep1 0x6ac0a0 (checks in binary order)."""
    flags = int(r.get('flags', 0)) & 0xFFFFFFFF
    if flags & FLAG_SHADOWS_ONLY == FLAG_SHADOWS_ONLY:
        return False
    if not (cp['culling_mask'] >> (int(r['layer']) & 31)) & 1:
        return False
    if not (int(r.get('scene_mask', SCENE_MASK_DEFAULT)) & cp['scene_mask']):
        return False
    if not r.get('has_renderer', True):
        return False
    if flags & FLAG_PENDING_REMOVAL:
        return False
    li = flags & LOD_INDEX_MASK
    if li:
        if lod_masks is None:
            raise NotImplementedError('LOD masks required (LODGroupManager::CalculateLODDataArray 0x771440)')
        if not (int(lod_masks[int(r.get('lod_slot', 0))][li]) & int(r.get('lod_mask', 0)) & 0xFF):
            return False
    return True


def shadow_casting_flags(mode):
    """SceneNode+0x1c bits 30-31 for a ShadowCastingMode."""
    return (int(mode) & 3) << 30


def cull(renderers, camera, lod_masks=None):
    """Non-Umbra camera culling (CullScene 0x68d2f0 -> CullDynamicScene 0x6505e0).

    renderers: dicts {list (0 static, 1 dynamic, 2/3 intermediate, 4/5 terrain),
    center, extent (world AABB), layer, flags (SceneNode+0x1c), scene_mask,
    has_renderer, lod_slot, lod_mask}; order within a list = scene node index.
    Returns {list_id: [renderer indices in visible order]} (ascending node order,
    independent of the job split).
    """
    cp = calculate_culling_parameters(camera)
    out = {k: [] for k in range(6)}
    for lid in range(6):
        for i, r in enumerate(renderers):
            if r.get('list', 1) != lid:
                continue
            if not node_passes_step1(r, cp, lod_masks):
                continue
            if not aabb_visible(r['center'], r['extent'], cp['planes']):
                continue
            if not layer_cull_distance_test(r['center'], r['extent'], int(r['layer']) & 31, cp):
                continue
            out[lid].append(i)
    return out


def configure_block_ranges_with_min(count, min_per_job, worker_threads):
    """ConfigureBlockRangesWithMinIndicesPerJob 0x7c00f0 -> [(start, size)] (count >= 1)."""
    if worker_threads == 0:
        jobs = 1
    else:
        jobs = min((min_per_job - 1 + count) // min_per_job, worker_threads * 2 + 2, 16)
    per = (count - 1 + jobs) // jobs
    n = (count - 1 + per) // per
    return [(i * per, (count - i * per) if i == n - 1 else per) for i in range(n)]


def render_node_queue_order(visible, renderers, worker_threads):
    """RenderNodeQueue order: BeginRenderQueueExtraction 0x68bc00, block jobs,
    compaction 0x68c7b0 (PopLastNodeIndex 0x7c0580), main-thread nodes 0x6909c0.

    renderers[i] keys: job_prepared (MeshRenderer True; SkinnedMeshRenderer has no
    PrepareRenderNodes callback -> False), produces_node, flags.
    Custom-cull, terrain-list-5 and projector nodes are not modelled (gap).
    """
    seq = [i for lid in range(5) for i in visible.get(lid, [])]
    if not seq:
        return []
    ranges = configure_block_ranges_with_min(len(seq), 128, worker_threads)
    slots = [None] * len(seq)
    written, queued = [], []
    for start, size in ranges:
        w, q = 0, []
        for i in seq[start:start + size]:
            r = renderers[i]
            if int(r.get('flags', 0)) & FLAG_PENDING_REMOVAL:
                continue
            if r.get('job_prepared', True):
                if r.get('produces_node', True):
                    slots[start + w] = i
                    w += 1
            else:
                q.append(i)
        written.append(w)
        queued.append(q)
    n = len(ranges)
    for k in range(n - 1):
        slot = ranges[k][0] + written[k]
        while slot < ranges[k + 1][0]:
            src = -1
            for j in range(n - 1, k, -1):
                if written[j]:
                    written[j] -= 1
                    src = ranges[j][0] + written[j]
                    break
            if src == -1:
                break
            slots[slot] = slots[src]
            written[k] += 1
            slot += 1
    total = next((ranges[j][0] + written[j] for j in range(n - 1, -1, -1) if written[j]), 0)
    order = slots[:total]
    for q in queued:
        order += [i for i in q if renderers[i].get('produces_node', True)]
    return order
