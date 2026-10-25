"""Bridge-render (pose-frozen) renderer matrices and bounds generated from local TRS.

During ``MakerRenderStyleService.Capture`` the pose is frozen
(``MakerRenderService.PoseFreeze.Apply``: every Animator ``Play(hash, 0, 0)`` +
``Update(0)``), then the camera renders 16 times.  Probe v6.1 exports the character
local transforms at every bridge render's ``pre_cull`` callback.

For a SkinnedMeshRenderer whose bounds come from its local AABB
(``updateWhenOffscreen`` false; prepared flag +4 clear), the per-render values are
(UnityPlayer.dll 2018.4.11f1, sha 5034cdc2...):

* ``unity_ObjectToWorld`` = TRS(rootBone world position, world rotation, 1)
  (CalculateGlobalPositionAndRotation 0xdda60, the same as
  ``unity_skin_emu.skin_from_locals``);
* world AABB: ``SkinnedMeshRendererManager::CalculateTransformInfo_Prepared``
  0xbdaca0 -> CalculateGlobalMatrix 0xdd620 of the root bone (affineX columns
  C0..C3, scale included), then the inline non-bone path 0xbdbe7f..0xbdc057 on the
  prepared local AABB (+0x58 centre, +0x64 extent; copied by InitializePreparedInfo
  0xbdce90 from SetLocalAABB 0x788010):

      centre = ((C1*cy + C2*cz) + C0*cx) + C3
      extent = (|C0*ex| + |C1*ey|) + |C2*ez|        (float32 lanes, abs after multiply)

``global_matrix`` runs CalculateGlobalMatrix itself in the emulator on a fake
TransformHierarchy (the layout ``unity_skin_emu.global_position_rotation`` uses).
"""
from __future__ import annotations

import struct

import numpy as np

F = np.float32
GLOBAL_MATRIX = 0xdd620   # affineX CalculateGlobalMatrix(TransformAccessReadOnly) (hidden return in rcx)


def global_matrix(emu, chain):
    """Columns C0..C3 (float32 x4 each) of the root's affineX world matrix.

    chain: local (t, q, s) of the transform, then its parent, ... up to the top."""
    emu.reset_heap()
    trs = b''.join(np.asarray(list(t) + [0.0], F).tobytes() + np.asarray(q, F).tobytes()
                   + np.asarray(list(s) + [0.0], F).tobytes() for t, q, s in chain)
    parents = list(range(1, len(chain))) + [-1]
    p_trs = emu.alloc(trs)
    p_par = emu.alloc(np.asarray(parents, np.int32).tobytes())
    hier = emu.alloc(struct.pack('<5Q', 0, 0, 0, p_trs, p_par))
    access = emu.alloc(struct.pack('<QI4x', hier, 0))
    out = emu.alloc(bytes(64))
    emu.call(GLOBAL_MATRIX, (out, access))
    return np.frombuffer(emu.read(out, 64), F).reshape(4, 4).copy()


def world_aabb(cols, local_center, local_extent):
    """Inline non-bone path of CalculateTransformInfo_Prepared (0xbdbe7f..0xbdbf5c)."""
    c0, c1, c2, c3 = (np.asarray(c, F) for c in cols)
    cx, cy, cz = (F(v) for v in local_center)
    ex, ey, ez = (F(v) for v in local_extent)
    centre = ((c1 * cy + c2 * cz) + c0 * cx) + c3
    extent = (np.abs(c0 * ex) + np.abs(c1 * ey)) + np.abs(c2 * ez)
    return centre[:3].astype(F), extent[:3].astype(F)


def chain_of(locals_by_id, parent_of, tid):
    """[(t, q, s) of tid, of its parent, ...] from {id: (t, q, s)} and {id: parent id | None}."""
    out = []
    while tid is not None:
        out.append(locals_by_id[tid])
        tid = parent_of[tid]
    return out
