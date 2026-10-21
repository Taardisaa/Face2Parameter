"""Pass plan of one Unity 2018.4.11f1 built-in forward camera render.

Builds the ordered passes and their draws (renderer, submesh, material, shader
pass, requested keywords) and attaches the built-in constant values from
``unity_forward``.  Shader/pass/variant selection is delegated to
``shader_library`` (sibling); culling, draw sorting and light classification
to ``unity_forward_rules`` (RVA-cited transcription).  The plan is the input of
``emit_draw`` which turns one planned draw plus a resource resolver into a
``frame_program`` draw command.

Pass order of the forward path (DoForwardShaderRenderLoop RVA 0x6c8d60 and the
CameraEvent order; confirmed pass by pass by the frame graphs of the retry and
paired captures): depth texture -> depth normals -> shadow maps of shadowed
lights -> screen-space shadow collector -> opaque (ForwardBase, ForwardAdd) ->
skybox (clearFlags Skybox) -> image effects opaque (command buffers, PPv2 sibling)
-> motion vectors (camera pass, object pass) -> transparent -> image effects.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import unity_forward as uf

OPAQUE_MAX_QUEUE = 2500        # Unity: render queue <= 2500 is opaque (geometry + alpha test)

PASS_KEYWORDS = {
    # Requested pass-context keywords set by the engine for each pass kind
    # (SetShadowsKeywords RVA 0x6a3d30 for the shadow keywords; light keywords
    # per light type).  Material keywords are OR-ed by shader_library.
    'depth_texture': ('SHADOWS_DEPTH',),
    'shadow_map': ('SHADOWS_DEPTH',),
}


@dataclass
class PlannedDraw:
    pass_kind: str
    renderer: dict
    submesh: int
    material: dict
    light_mode: str
    keywords: tuple
    builtin: dict = field(default_factory=dict)      # cbuffer variable -> value (unity_forward encodings)
    notes: list = field(default_factory=list)


@dataclass
class PlannedPass:
    kind: str
    draws: list = field(default_factory=list)
    builtin: dict = field(default_factory=dict)      # per-pass values (camera/light constants)
    notes: list = field(default_factory=list)


def material_queue(material):
    q = material.get('render_queue')
    if q is None:
        raise uf.GeneratorGap('material render_queue missing')
    return int(q)


def split_queues(items):
    opaque, transparent = [], []
    for it in items:
        (opaque if material_queue(it['material']) <= OPAQUE_MAX_QUEUE else transparent).append(it)
    return opaque, transparent


def renderer_items(renderers):
    out = []
    for r in renderers:
        for i, m in enumerate(r.get('materials', [])):
            out.append({'renderer': r, 'submesh': i, 'material': m})
    return out


def plan_camera_render(cam, setup, graphics, renderers, lights, rules, *, time=None):
    """Ordered PlannedPass list for one camera render.

    ``rules`` must provide ``cull(renderers, camera)``, ``sort_draws(items,
    camera, transparent)`` and ``classify_lights(renderer, lights, quality)``
    (unity_forward_rules).  Missing rules raise GeneratorGap: nothing is
    approximated."""
    for name in ('cull', 'sort_draws', 'classify_lights'):
        if not hasattr(rules, name):
            raise uf.GeneratorGap('forward rule %s not available' % name)
    camera_values = uf.camera_shader_props(cam, setup, graphics)
    if time is not None:
        camera_values.update(uf.time_shader_props(time))
    visible = rules.cull(renderers, cam)
    items = renderer_items(visible)
    opaque, transparent = split_queues(items)
    mode = int(cam.get('depth_texture_mode'))
    passes = []
    if mode & 1:
        p = PlannedPass('depth_texture', builtin=dict(camera_values, unity_LightShadowBias=[uf.F(0.0)] * 4))
        for it in rules.sort_draws(opaque, cam, False):
            p.draws.append(PlannedDraw('depth_texture', it['renderer'], it['submesh'], it['material'], 'SHADOWCASTER',
                                       PASS_KEYWORDS['depth_texture'], uf.per_draw_values(it['renderer'])))
        p.notes.append('unity_LightShadowBias (0,0,0,0) is capture_observed for the camera depth pass')
        passes.append(p)
    if mode & 2:
        p = PlannedPass('depth_normals', builtin=dict(camera_values))
        for it in rules.sort_draws(opaque, cam, False):
            p.draws.append(PlannedDraw('depth_normals', it['renderer'], it['submesh'], it['material'],
                                       'REPLACEMENT:Hidden/Internal-DepthNormalsTexture:RenderType', (),
                                       uf.per_draw_values(it['renderer'])))
        passes.append(p)
    shadowed = [l for l in lights if l.get('type') == 'Directional' and l.get('shadows') not in (None, 'None')]
    if shadowed:
        passes.append(PlannedPass('shadow_map', notes=['cascade fit: unity_shadows.directional_cascade; casters and '
                                                       'MinMaxAABBs: unity_shadow_bounds']))
        passes.append(PlannedPass('screenspace_shadows', builtin=dict(camera_values)))
    p = PlannedPass('opaque', builtin=dict(camera_values))
    for it in rules.sort_draws(opaque, cam, False):
        cls = rules.classify_lights(it['renderer'], lights, None)
        p.draws.append(PlannedDraw('opaque', it['renderer'], it['submesh'], it['material'], 'FORWARDBASE',
                                   tuple(cls.get('base_keywords', ())), uf.per_draw_values(it['renderer'])))
        for add in cls.get('add_lights', []):
            p.draws.append(PlannedDraw('opaque', it['renderer'], it['submesh'], it['material'], 'FORWARDADD',
                                       tuple(add.get('keywords', ())), uf.per_draw_values(it['renderer'])))
    passes.append(p)
    if cam.get('clear_flags') == 'Skybox':
        sk = uf.skybox_matrices(cam, setup, graphics)
        passes.append(PlannedPass('skybox', builtin={'unity_MatrixVP': sk['unity_MatrixVP'],
                                                     'unity_ObjectToWorld': sk['unity_ObjectToWorld']}))
    if mode & 4:
        p = PlannedPass('motion_vectors', builtin=dict(camera_values))
        p.notes.append('camera pass: LoadFullScreenOrthoMatrix(0, projectionFar); object pass: '
                       'renderers with Object motion mode (MotionVectorRenderLoop sorter RVA 0x6c6f20)')
        passes.append(p)
    p = PlannedPass('transparent', builtin=dict(camera_values))
    for it in rules.sort_draws(transparent, cam, True):
        cls = rules.classify_lights(it['renderer'], lights, None)
        p.draws.append(PlannedDraw('transparent', it['renderer'], it['submesh'], it['material'], 'FORWARDBASE',
                                   tuple(cls.get('base_keywords', ())), uf.per_draw_values(it['renderer'])))
    passes.append(p)
    return passes


def draw_cbuffers(draw, pass_values, layouts, shadow_copies, pack_cbuffer):
    """Exact cbuffer bytes for one draw: for each Unity cbuffer layout of the
    bound program (shader_library ``Subprogram.constant_buffers()``), write the
    variables the generator produced into that (name, size) shadow copy and
    return the whole buffer, as BeforeDrawCall + Map(WRITE_DISCARD) do.
    ``$Globals`` values (material properties) must already be in pass_values
    (shader-library sibling); a missing variable is a gap, not a zero."""
    values = dict(pass_values)
    values.update(draw.builtin)
    out = {}
    missing = {}
    for layout in layouts:
        key = (layout.name, layout.size)
        base = shadow_copies.get(key, bytes(layout.size))
        names = {v.name for v in layout.variables}
        have = {k: (v[1] if isinstance(v, tuple) else v) for k, v in values.items() if k in names}
        lacking = sorted(names - set(have))
        if lacking:
            missing[layout.name] = lacking
        data = pack_cbuffer(layout, have, base=base, allow_missing=True)
        shadow_copies[key] = data
        out[layout.slot] = {'name': layout.name, 'bytes': data}
    return out, missing
