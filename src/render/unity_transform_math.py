"""Float32 Unity 2018.4.11f1 transform-setter arithmetic (pure numpy ports).

Each function is a line-by-line port of an installed UnityPlayer.dll routine,
cross-checked bit for bit against the emulated original in
tools/offline_renderer/test_paired_skin.py (UnityEmuTests).
"""
from __future__ import annotations

import numpy as np

F = np.float32
# SetLocalR (RVA 0x1b12f0): cmpltps against [0x1b1334+0x101f00c] = 1e-30f x4;
# fallback lanes from [0x1b134d+0x101f103] = (0,0,0,1).
NORM_EPS = F(1e-30)
NORM_FALLBACK = np.array([0, 0, 0, 1], F)


def normalize_quaternion(q):
    """Transform::SetLocalRotation / set_localRotation storage normalization."""
    q = np.asarray(q, F)
    v = (q * q).astype(F)
    t = (v + np.roll(v, -1)).astype(F)
    s = (t + np.roll(t, -2)).astype(F)
    out = (q / np.sqrt(s)).astype(F)
    return np.where(s > NORM_EPS, out, NORM_FALLBACK).astype(F)


def managed_quaternion_mul(a, b):
    """UnityEngine.Quaternion.operator* as executed by HS2's Mono JIT.

    The managed expression order is the C# source order; Mono evaluates the
    float32 arithmetic in double precision and rounds once when storing each
    Quaternion field (evidence: paired frame 26060, cf_J_NoseBase_s, where the
    per-operation float32 order differs by 1 ulp and this form is bit-equal).
    """
    ax, ay, az, aw = (float(v) for v in np.asarray(a, F))
    bx, by, bz, bw = (float(v) for v in np.asarray(b, F))
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by + ay * bw + az * bx - ax * bz,
                     aw * bz + az * bw + ax * by - ay * bx,
                     aw * bw - ax * bx - ay * by - az * bz], np.float64).astype(F)


# ---- UnityEngine managed math as executed by HS2's Mono JIT ----------------
# Mono evaluates float32 C# expressions in double precision; register-resident
# locals stay double. Values are rounded to float32 when stored to memory
# (fields, struct members, array elements), passed as float32 arguments or
# returned (evidence: paired frame 26060, GetInfo `num` local, cf_J_Eye01_L). Each helper follows the UnityCsReference
# 2018.4 source expression and rounds exactly at those points.

def _r(x):
    return F(x)


def clamp01(v):
    v = _r(v)
    return F(0) if v < 0 else (F(1) if v > 1 else v)


def clamp(v, lo, hi):
    v = _r(v)
    return _r(lo) if v < lo else (_r(hi) if v > hi else v)


def floor(v):                       # Mathf.Floor(f) => (float)Math.Floor(f)
    return _r(np.floor(float(_r(v))))


def floor_to_int(v):                # Mathf.FloorToInt(f) => (int)Math.Floor(f)
    return int(np.floor(float(_r(v))))


def repeat(t, length):              # Clamp(t - Floor(t / length) * length, 0, length)
    t, length = _r(t), _r(length)
    fl = floor(_r(float(t) / float(length)))
    return clamp(float(t) - float(fl) * float(length), 0.0, length)


def lerp_angle(a, b, t):            # Repeat(b - a, 360); if > 180 -= 360; a + num * Clamp01(t)
    a, b = _r(a), _r(b)
    num = float(repeat(_r(float(b) - float(a)), 360.0))   # local stays double
    if num > 180.0:
        num = num - 360.0
    return _r(float(a) + num * float(clamp01(t)))


def vector3_lerp(a, b, t):          # t = Clamp01(t); a + (b - a) * t per component
    t = clamp01(t)
    a = np.asarray(a, F)
    b = np.asarray(b, F)
    return np.array([float(a[i]) + (float(b[i]) - float(a[i])) * float(t) for i in range(3)], np.float64).astype(F)
