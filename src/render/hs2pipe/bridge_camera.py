"""Camera pose of ``MakerRenderStyleService.Capture`` for any yaw, and everything it moves.

Bridge source (HS2Mod ``plugins/HS2_McpBridge/MakerRenderStyleService.cs``)::

    var distance = size / Mathf.Tan(cam.fieldOfView * Mathf.Deg2Rad * .5f);
    cam.transform.position = target + Quaternion.Euler(0, yaw, 0) * new Vector3(0, 0, distance);
    cam.transform.LookAt(target);

Every step is the installed code, run or transcribed exactly:

* ``Quaternion.Euler(0, yaw, 0)``: managed ``new Vector3(x, y, z) * Deg2Rad`` (binary32
  products) -> native ``Internal_FromEulerRad`` (``unity_skin_emu.quaternion_euler_degrees``,
  emulated 0x9541d0);
* ``Quaternion * Vector3``: UnityEngine.CoreModule ``Quaternion.op_Multiply(Quaternion,
  Vector3)`` IL (RVA 0x12d3c, 313 bytes): twelve ``stloc`` float32 locals, then three
  ``stfld`` sums; Mono evaluates the expression stack in double and rounds at each store
  (``plugin_common.mono_f32`` convention); ``Vector3.op_Addition`` likewise;
* ``Transform.position`` setter: ``Transform::SetPosition`` 0xc01f60 -> ``SetGlobalT``
  0x2030a0: local T = ``InverseTransformPosition`` 0x1fff50 over the parent (emulated);
* ``Transform.LookAt(target)`` -> ``Internal_LookAt(target, Vector3.up)`` 0x870400:
  ``GetPosition`` (CalculateGlobalPosition 0x1dd810), forward = target - position (binary32),
  ``LookRotationToQuaternion`` 0x7ce350 (initial result identity); when it fails, the
  normalise + ``Matrix3x3f::SetFromToRotation(zAxis, dir)`` 0x7da4c0 + ``MatrixToQuaternion``
  0x7ce390 path (0x87049c..0x870542); then ``Transform::SetRotation`` 0xc02230: local R =
  ``InverseTransformRotation`` 0x200180 over the parent, normalised by ``SetLocalR``
  (0x1b130f..0x1b1350) -- all emulated;
* camera matrices (implicit view and culling, Camera+0x4ea / +0x4f9 set):
  ``GetWorldToCameraMatrix`` 0x6647b0 = Scale(1, 1, -1) * ``GetWorldToLocalMatrixNoScale``
  (``MultiplyMatrices4x4`` 0xe48d0); ``GetCameraToWorldMatrix`` 0x662dc0 =
  ``InvertMatrix4x4_Full`` 0x7cd530; ``GetCullingMatrix`` 0x662ea0 = projection *
  worldToCamera (the CTAA-jittered projection of that render);
* every transform under the camera (the camera-attached lights ``DHH_Light``,
  ``Lights Custom/Directional Light Key|Back``, the DHH SSS cameras at local identity):
  world position / rotation by ``CalculateGlobalPositionAndRotation`` 0xdda60 over the new
  camera local; their local TRS do not change.

Not yaw-dependent (source): ``_PreviousVP`` is Camera+0x598, copied from the last-used VP
+0x5d8 only by ``RenderManager::UpdatePreviousFrameMatrices`` once per frame (0x66e4ce), so all
16 bridge renders see frame N-1's normal-view VP (constant across the 16 records); CTAA jitter
offsets the projection only; DOF focus = distance (yaw-free); shadow cascade fit and culling
are computed from these camera matrices and the generated light rotation by the existing
transcribed passes.
"""
from __future__ import annotations

import struct

import numpy as np

from .plugin_common import d, mono_f32

F = np.float32
INVERSE_TRANSFORM_POSITION = 0x1fff50
INVERSE_TRANSFORM_ROTATION = 0x200180
CALCULATE_GLOBAL_POSITION = 0x1dd810
LOOK_ROTATION_TO_QUATERNION = 0x7ce350
SET_FROM_TO_ROTATION = 0x7da4c0
MATRIX_TO_QUATERNION = 0x7ce390
MULTIPLY_MATRICES = 0xe48d0
INVERT_FULL = 0x7cd530
SQRTF = 0x116c720
VECTOR3_EPSILON_RVA = 0x1257060       # ?epsilon@Vector3f@@2MB (comiss at 0x8704ec, next 0x8704f3 + 0x9e6b6d)
Z_AXIS_RVA = 0x14b5248                # ?zAxis@Vector3f@@2V1@B (lea at 0x8704fe, next 0x870505 + 0xc44d43)


# ------------------------------------------------------------------ managed (Mono) parts
def quaternion_mul_vector3(q, p):
    """``Quaternion.op_Multiply(Quaternion, Vector3)`` IL, Mono double stack / float32 stores."""
    x, y, z, w = (d(v) for v in q)
    px, py, pz = (d(v) for v in p)
    n0 = d(mono_f32(x * 2.0))
    n1 = d(mono_f32(y * 2.0))
    n2 = d(mono_f32(z * 2.0))
    n3 = d(mono_f32(x * n0))
    n4 = d(mono_f32(y * n1))
    n5 = d(mono_f32(z * n2))
    n6 = d(mono_f32(x * n1))
    n7 = d(mono_f32(x * n2))
    n8 = d(mono_f32(y * n2))
    n9 = d(mono_f32(w * n0))
    n10 = d(mono_f32(w * n1))
    n11 = d(mono_f32(w * n2))
    rx = mono_f32((1.0 - (n4 + n5)) * px + (n6 - n11) * py + (n7 + n10) * pz)
    ry = mono_f32((n6 + n11) * px + (1.0 - (n3 + n5)) * py + (n8 - n9) * pz)
    rz = mono_f32((n7 - n10) * px + (n8 + n9) * py + (1.0 - (n3 + n4)) * pz)
    return np.array([rx, ry, rz], F)


def vector3_add(a, b):
    """``Vector3.op_Addition``: ``new Vector3(a.x + b.x, a.y + b.y, a.z + b.z)``."""
    return np.array([mono_f32(d(a[i]) + d(b[i])) for i in range(3)], F)


# ------------------------------------------------------------------ native (emulated) parts
def _hierarchy(emu, chain):
    """Fake TransformHierarchy (layout of unity_skin_emu.global_position_rotation):
    chain[0] = the transform, chain[i + 1] = parent of chain[i]."""
    trs = b''.join(np.asarray(list(t) + [0.0], F).tobytes() + np.asarray(q, F).tobytes()
                   + np.asarray(list(s) + [0.0], F).tobytes() for t, q, s in chain)
    parents = list(range(1, len(chain))) + [-1]
    p_trs = emu.alloc(trs)
    p_par = emu.alloc(np.asarray(parents, np.int32).tobytes())
    return emu.alloc(struct.pack('<5Q', 0, 0, 0, p_trs, p_par))


def _access(emu, hier, index):
    return emu.alloc(struct.pack('<QI4x', hier, index))


def inverse_transform_position(emu, parent_chain, world):
    emu.reset_heap()
    acc = _access(emu, _hierarchy(emu, parent_chain), 0)
    p = emu.alloc(np.asarray(list(world) + [0.0], F).tobytes())
    emu.call(INVERSE_TRANSFORM_POSITION, (acc, p))
    return np.frombuffer(emu.read(p, 12), F).copy()


def inverse_transform_rotation(emu, parent_chain, world_q):
    emu.reset_heap()
    acc = _access(emu, _hierarchy(emu, parent_chain), 0)
    p = emu.alloc(np.asarray(world_q, F).tobytes())
    emu.call(INVERSE_TRANSFORM_ROTATION, (acc, p))
    return np.frombuffer(emu.read(p, 16), F).copy()


def calculate_global_position(emu, chain):
    emu.reset_heap()
    acc = _access(emu, _hierarchy(emu, chain), 0)
    out = emu.alloc(bytes(16))
    emu.call(CALCULATE_GLOBAL_POSITION, (out, acc))
    return np.frombuffer(emu.read(out, 12), F).copy()


def look_rotation(emu, forward, up=(0.0, 1.0, 0.0)):
    """``Internal_LookAt`` 0x870400 rotation part (identity seed, fallback included)."""
    from unicorn import x86_const as x
    emu.reset_heap()
    pf = emu.alloc(np.asarray(list(forward) + [0.0], F).tobytes())
    pu = emu.alloc(np.asarray(list(up) + [0.0], F).tobytes())
    pq = emu.alloc(np.asarray([0.0, 0.0, 0.0, 1.0], F).tobytes())
    emu.call(LOOK_ROTATION_TO_QUATERNION, (pf, pu, pq))
    if emu.uc.reg_read(x.UC_X86_REG_RAX) & 0xFF:
        return np.frombuffer(emu.read(pq, 16), F).copy(), 'LookRotationToQuaternion'
    # 0x87049c..0x870542: |forward| by sqrtf; <= Vector3f::epsilon -> rotation unchanged
    fx, fy, fz = (F(v) for v in forward)
    sq = F(F(F(fy * fy) + F(fx * fx)) + F(fz * fz))
    emu.reset_heap()
    from unicorn import x86_const as x2
    emu.uc.reg_write(x2.UC_X86_REG_XMM0, int.from_bytes(np.asarray([sq, 0, 0, 0], F).tobytes(), 'little'))
    emu.call(SQRTF, ())
    n = np.frombuffer(emu.uc.reg_read(x2.UC_X86_REG_XMM0).to_bytes(16, 'little'), F)[0]
    eps = np.frombuffer(emu.read(emu.base + VECTOR3_EPSILON_RVA, 4), F)[0]
    if not (n > eps):
        return None, 'degenerate forward: rotation unchanged (0x8704f3)'
    v = np.array([F(fx / n), F(fy / n), F(fz / n)], F)
    emu.reset_heap()
    pm = emu.alloc(bytes(48))
    pv = emu.alloc(v.tobytes() + bytes(4))
    emu.call(SET_FROM_TO_ROTATION, (pm, emu.base + Z_AXIS_RVA, pv))
    pq = emu.alloc(np.asarray([0.0, 0.0, 0.0, 1.0], F).tobytes())
    emu.call(MATRIX_TO_QUATERNION, (pm, pq))
    return np.frombuffer(emu.read(pq, 16), F).copy(), 'SetFromToRotation + MatrixToQuaternion'


def multiply(emu, a, b):
    """MultiplyMatrices4x4(a, b) (column-major 16 floats each)."""
    emu.reset_heap()
    pa = emu.alloc(np.asarray(a, F).tobytes())
    pb = emu.alloc(np.asarray(b, F).tobytes())
    po = emu.alloc(bytes(64))
    emu.call(MULTIPLY_MATRICES, (pa, pb, po))
    return np.frombuffer(emu.read(po, 64), F).copy()


def invert_full(emu, m):
    emu.reset_heap()
    pa = emu.alloc(np.asarray(m, F).tobytes())
    po = emu.alloc(bytes(64))
    emu.call(INVERT_FULL, (pa, po))
    return np.frombuffer(emu.read(po, 64), F).copy()


# ------------------------------------------------------------------ the capture pose
def capture_camera(emu, *, target, ortho_size, yaw, field_of_view, camera_local_scale, parent_chain):
    """Camera local / world TRS after ``position = ...; LookAt(target)``.

    parent_chain: [(t, q, s) of the camera's parent, its parent, ...]; camera_local_scale:
    the camera's (unchanged) local scale."""
    from src.render.unity_skin_emu import quaternion_euler_degrees, emulated_normalize, global_position_rotation
    from .capture_loop import capture_distance
    distance = capture_distance(ortho_size, field_of_view)
    q_yaw = quaternion_euler_degrees(emu, [0.0, float(F(yaw)), 0.0])
    world_pos = vector3_add(np.asarray(target, F), quaternion_mul_vector3(q_yaw, [0.0, 0.0, distance]))
    local_t = inverse_transform_position(emu, parent_chain, world_pos) if parent_chain else world_pos
    s = np.asarray(camera_local_scale, F)
    # the rotation before LookAt only enters GetPosition through scale-free translation terms
    pos_now = calculate_global_position(emu, [(local_t, np.array([0, 0, 0, 1], F), s)] + list(parent_chain))
    tgt = np.asarray(target, F)
    forward = np.array([F(tgt[0] - pos_now[0]), F(tgt[1] - pos_now[1]), F(tgt[2] - pos_now[2])], F)
    q_world, how = look_rotation(emu, forward)
    if q_world is None:
        raise ValueError('camera at the look-at target (degenerate LookAt)')
    q_local = inverse_transform_rotation(emu, parent_chain, q_world) if parent_chain else q_world
    q_local = emulated_normalize(emu, q_local)
    chain = [(local_t, q_local, s)] + list(parent_chain)
    w_pos, w_rot = global_position_rotation(emu, chain)
    return {'distance': distance, 'euler_quaternion': q_yaw, 'world_position_set': world_pos,
            'local_position': local_t, 'local_rotation': q_local, 'world_position': w_pos, 'world_rotation': w_rot,
            'look_rotation_path': how, 'chain': chain}


def camera_matrices(emu, pose, projection):
    """worldToCamera / cameraToWorld / culling (column-major) of the posed camera."""
    w2l, _ = emu.world_to_local_noscale(pose['world_position'], pose['world_rotation'])
    w2l = np.asarray(w2l, F).reshape(4, 4).T.ravel() if np.asarray(w2l).shape == (4, 4) else np.asarray(w2l, F)
    scale = np.diag(np.array([1.0, 1.0, -1.0, 1.0], F)).T.ravel()
    w2c = multiply(emu, scale, w2l)
    c2w = invert_full(emu, w2c)
    cull = multiply(emu, np.asarray(projection, F), w2c)
    return {'world_to_camera': w2c, 'camera_to_world': c2w, 'culling': cull}


def attached_world(emu, local_chain_below_camera, pose):
    """World position / rotation of a transform under the camera: local_chain_below_camera =
    [(t, q, s) of the transform, of its parent, ... up to the camera's direct child]."""
    from src.render.unity_skin_emu import global_position_rotation
    return global_position_rotation(emu, list(local_chain_below_camera) + list(pose['chain']))
