"""AIT/Liquid (and other GrabPass shaders) in Unity 2018.4.11f1's forward loop.

The body liquid layer uses ``AIT/Liquid`` from ``abdata/chara/oo_base.unity3d``
(path ID ``5561447335612986018``): pass 0 is an *unnamed* ``GrabPass {}``,
pass 1 ``ForwardBase`` (queue ``Transparent-1``), pass 2 ``ForwardAdd``.
``AIT/main namida`` (tears) has the same unnamed GrabPass as pass 0.

GrabPass semantics are taken from the installed ``UnityPlayer.dll``
(2018.4.11f1, CodeView GUID BFA91A8C…, public PDB), disassembled read-only with
``tools/offline_renderer/unityplayer_disasm.py`` (evidence in the ignored
``outputs/offline_renderer_plugins_20261020/unity_disasm``):

* ``DoForwardShaderRenderLoop`` (RVA 0x6c8d60) calls
  ``ShaderLab::GrabPasses::Add(Pass*, Camera*)`` (0x859740) for every pass of
  type 2 while building the pass list.  ``Add`` keys a map by texture name;
  ``Pass+0x3c8 == -1`` (no name) selects ``_GrabTexture`` and marks the entry
  *unnamed* (flag bit 0).  A new entry gets its RT from 0x85b500 and publishes
  it as a global texture with auxiliary properties.
* 0x85b500 (grab texture acquisition): the descriptor (0x85ad60) copies width,
  height and format of ``RenderTexture::GetActive`` (camera rect if none),
  depth 0; ``GetTempBuffer`` named ``"GrabPass Temp"``; filter Bilinear.
  Unnamed: one static temp, released and re-acquired.  Named: looked up in a
  per-render-loop list; when already present the entry is flagged *grabbed*.
* ``GrabPasses::ApplyGrabPass`` (0x859da0), called from ``Pass::ApplyPass``
  when the grab pass is applied: unnamed -> ``RenderTexture::GrabPixels``
  every time; named -> only if not yet grabbed, then flag grabbed.
* ``ClearGrabPassFrameState`` (0x859f30) runs in ``CleanupAfterRenderLoop``:
  named grabs are therefore once per camera render loop (not per frame).
* ``GrabPixels`` -> ``GfxDeviceD3D11Base::GrabIntoRenderTexture``: in the
  capture a full-rect ``CopySubresourceRegion`` from the camera colour target.
* ``StartRenderJobs`` (0x6d05d0) uses a single job when the loop has grab
  passes (flag 0x10), so order is the sorted object order.  ``BatchRenderer``
  defers a ``ForwardBase`` draw (``Add``) until a later flush; neither
  ``ApplyShaderPass`` nor ``ApplyGrabPass`` flushes.  Hence an object's grab
  executes *before* the still-pending draw of the preceding object.  The
  capture shows exactly this: namida's grab (1891) precedes the eyelash draw
  (1910), so namida refracts a colour buffer without eyelashes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .plugin_common import F32, Op, PluginStateError, UnimplementedBranch, UnityRenderTexture, pack_variables

GRAB_TEXTURE = '_GrabTexture'
GRAB_TEMP_NAME = 'GrabPass Temp'
LIQUID_SHADER = 'AIT/Liquid'
LIQUID_BUNDLE = 'abdata/chara/oo_base.unity3d'
LIQUID_PATH_ID = 5561447335612986018
PASS_TYPE_GRAB = 2


@dataclass(frozen=True)
class ForwardObject:
    """One renderer/material of the forward loop, in Unity's sorted order.

    ``passes``: the shader passes Unity runs for this object in this loop, in
    shader order: ``{'pass': i, 'kind': 'grab'|'forward_base'|'forward_add',
    'grab_name': str|None}``; ``forward_add_lights``: additional pixel lights.
    """
    name: str
    shader: str
    queue: int
    passes: Sequence[dict]
    forward_add_lights: int = 0


def grab_passes_of(manifests, shader_name: str) -> List[dict]:
    """GrabPass entries of an installed shader (type 2 passes, name '' = unnamed)."""
    out = []
    for p in manifests.shader(shader_name)['passes']:
        if p['type'] == PASS_TYPE_GRAB:
            out.append({'subshader': p['subshader'], 'pass': p['pass'], 'grab_name': p['texture_name'] or None})
    return out


def forward_passes(manifests, shader_name: str) -> List[dict]:
    """Shader pass list for the forward loop (Grab / ForwardBase / ForwardAdd)."""
    rows = []
    for p in manifests.shader(shader_name)['passes']:
        if p['type'] == PASS_TYPE_GRAB:
            rows.append({'pass': p['pass'], 'kind': 'grab', 'grab_name': p['texture_name'] or None})
            continue
        tags = dict((k.upper(), v) for k, v in p['state']['m_Tags']['tags'])
        mode = tags.get('LIGHTMODE')
        if mode == 'FORWARDBASE':
            rows.append({'pass': p['pass'], 'kind': 'forward_base', 'grab_name': None})
        elif mode in (None, 'ALWAYS'):
            # Untagged ("Always") passes run in the base-pass loop of the forward renderer.
            rows.append({'pass': p['pass'], 'kind': 'forward_base', 'grab_name': None, 'light_mode': 'Always'})
        elif mode == 'FORWARDADD':
            rows.append({'pass': p['pass'], 'kind': 'forward_add', 'grab_name': None})
    return rows


def grab_rt(active: UnityRenderTexture) -> UnityRenderTexture:
    """0x85ad60: grab descriptor from the active colour target, depth 0, Bilinear."""
    if active.msaa != 1:
        raise UnimplementedBranch('GrabPass from an MSAA target (resolve path) is not lowered')
    return UnityRenderTexture(GRAB_TEMP_NAME, active.width, active.height, 0, active.resolved_format(),
                              active.read_write, filter_mode='Bilinear', wrap_mode='Clamp',
                              source='ShaderLab GrabPasses (UnityPlayer 0x85b500)')


@dataclass
class _Entry:
    name: str
    unnamed: bool
    grabbed: bool
    handle: str


def schedule_forward_loop(objects: Sequence[ForwardObject], active_target: UnityRenderTexture) -> List[Op]:
    """Ordered draws and grab copies of one forward render loop (one camera).

    Draw ops are requests to ``unity_forward`` (it lowers the pass with its
    lights/material state); grab copies are lowered here.  The pending-batch
    rule above determines where a grab lands relative to the previous draw.
    """
    src = 'Unity ForwardShaderRenderLoop + ShaderLab::GrabPasses'
    ops: List[Op] = []
    entries: Dict[str, _Entry] = {}
    named_frame_state: Dict[str, str] = {}
    unnamed_handle: Optional[str] = None
    # Prepare: GrabPasses::Add during pass-list construction.
    for obj in objects:
        for p in obj.passes:
            if p['kind'] != 'grab':
                continue
            name = p['grab_name'] or GRAB_TEXTURE
            if name in entries:
                continue
            unnamed = p['grab_name'] is None
            if unnamed:
                if unnamed_handle is not None:
                    ops.append(Op('release_temporary', 'plugin', src, {'handle': unnamed_handle}))
                unnamed_handle = 'grab:' + GRAB_TEXTURE
                handle, already = unnamed_handle, False
            elif name in named_frame_state:
                handle, already = named_frame_state[name], True
            else:
                handle, already = 'grab:' + name, False
                named_frame_state[name] = handle
            if not already:
                ops.append(Op('get_temporary', 'plugin', src, {'handle': handle, 'rt': grab_rt(active_target)}))
            ops.append(Op('set_global_texture', 'unity_forward', src, {'name': name, 'handle': handle,
                                                                       'auxiliary': ['_TexelSize', '_ST', '_HDR']}))
            entries[name] = _Entry(name, unnamed, already, handle)
    pending: Optional[Op] = None
    for obj in objects:
        for p in obj.passes:
            if p['kind'] == 'grab':
                e = entries[p['grab_name'] or GRAB_TEXTURE]
                if e.unnamed or not e.grabbed:
                    ops.append(Op('grab_copy', 'plugin', src, {
                        'object': obj.name, 'shader': obj.shader, 'pass': p['pass'], 'texture': e.name,
                        'dest': e.handle, 'source': 'active colour target',
                        'box': {'left': 0, 'top': 0, 'front': 0, 'right': active_target.width,
                                'bottom': active_target.height, 'back': 1},
                        'before_pending_draw_of': pending.args['object'] if pending is not None else None}))
                    e.grabbed = True
            elif p['kind'] == 'forward_base':
                if pending is not None:
                    ops.append(pending)
                pending = Op('draw_pass', 'unity_forward', src, {'object': obj.name, 'shader': obj.shader,
                                                                  'pass': p['pass'], 'light_mode': 'ForwardBase',
                                                                  'path': 'BatchRenderer.Add (deferred until flush)'})
            elif p['kind'] == 'forward_add':
                if obj.forward_add_lights == 0:
                    continue          # no additional pixel light: the pass is not run
                if pending is not None:
                    ops.append(pending)
                    pending = None
                for light in range(obj.forward_add_lights):
                    ops.append(Op('draw_pass', 'unity_forward', src, {'object': obj.name, 'shader': obj.shader,
                                                                       'pass': p['pass'], 'light_mode': 'ForwardAdd',
                                                                       'light_index': light,
                                                                       'path': 'BatchRenderer.RenderSingleWithPass (immediate)'}))
    if pending is not None:
        ops.append(pending)   # BatchRenderer::EndLoopFlush
    return ops


def end_render_loop() -> List[Op]:
    """``CleanupAfterRenderLoop`` -> ``ClearGrabPassFrameState``."""
    return [Op('release_grab_frame_state', 'plugin', 'ShaderLab::ClearGrabPassFrameState', {})]


def grab_copy_command(builder, *, source_resource: str, dest_resource: str, width: int, height: int) -> int:
    """Contract command of one ``GrabIntoRenderTexture`` (non-MSAA, same format)."""
    return builder.copy_subresource_region(dest_resource, 0, (0, 0, 0), source_resource, 0,
                                           src_box={'left': 0, 'top': 0, 'front': 0, 'right': width, 'bottom': height, 'back': 1})


def liquid_material_globals(manifests, material_values: Dict[str, object], keywords: Sequence[str],
                            engine_values: Dict[str, object], pass_index: int = 1) -> Dict[str, dict]:
    """PS ``$Globals`` bytes of the Liquid ForwardBase pass.

    ``material_values``: the Liquid material's saved property values (scene
    state; ``Material.GetFloat/GetVector`` of every ``AIT/Liquid`` property).
    ``engine_values``: engine-set uniforms in the same cbuffer (``_LightColor0``
    from the forward light setup).  Nothing is defaulted.
    """
    program = manifests.program(LIQUID_SHADER, pass_index, 'progFragment', list(keywords))
    values = dict(material_values)
    values.update(engine_values)
    return pack_variables(program, values)
