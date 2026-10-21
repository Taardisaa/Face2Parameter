"""Run HS2's own managed driver code on the game's own Mono JIT, offline.

The per-frame face drivers (EyeLookCalc, NeckLookCalcVer2, FaceBlendShape and
its FBS controllers) are managed code in the installed IL.dll. Their float32
results depend on how HS2's Mono JIT evaluates C# float arithmetic (double
evaluation, rounding at stores/args/returns, register allocation across basic
blocks). Instead of transcribing that, this module embeds the installed
runtime itself:

  * E:/HoneySelect2_ArcticFox/MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll,
    initialised exactly like UnityPlayer's InitializeMonoFromMain:
    mono_set_defaults(0, mono_parse_default_optimizations(NULL)) (RVA 0x8e99d5).
  * The installed HoneySelect2_Data/Managed assemblies (IL.dll, CoreModule...);
    their IL is JIT-compiled here exactly as in the game process.
  * UnityPlayer.dll loaded in-process; every native method the drivers reach
    is either registered directly as the Mono internal call (the original
    Quaternion/Vector3/Random binding functions) or, for Transform, served from
    a scene model whose arithmetic is UnityPlayer's own hierarchy routines
    (CalculateGlobalPositionAndRotation, TransformPoint, InverseTransformPosition,
    InverseTransformRotation) plus the validated SetLocalR normalisation port.

Mono can be initialised only once per process; use one HostedRuntime per
process (tools run it in a subprocess).
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import struct
from pathlib import Path

import numpy as np

from .unity_transform_math import normalize_quaternion

GAME = Path('E:/HoneySelect2_ArcticFox')
MANAGED = GAME / 'HoneySelect2_Data' / 'Managed'
MONO_DLL = GAME / 'MonoBleedingEdge' / 'EmbedRuntime' / 'mono-2.0-bdwgc.dll'
UNITY_DLL = GAME / 'UnityPlayer.dll'
UNITY_SHA = '5034cdc2bff71dc8d9e038cb71059f8965d0d879f8098af0f3c726ea06e13923'

# UnityPlayer.dll RVAs (matched PDB BFA91A8C44734BB4AABC5A975B304CFE age 1).
RVA = {
    'Quaternion_Inverse': 0x954310, 'Quaternion_LookRotation': 0x9543d0,
    'Quaternion_AngleAxis': 0x954080, 'Quaternion_Slerp': 0x954560,
    'Quaternion_SlerpUnclamped': 0x954530, 'Quaternion_Lerp': 0x954390,
    'Quaternion_LerpUnclamped': 0x954360, 'Quaternion_FromToRotation': 0x9541a0,
    'Quaternion_FromEulerRad': 0x9541d0, 'Quaternion_ToEulerRad': 0x9542d0,
    'Vector3_Slerp': 0x954bd0, 'Vector3_SlerpUnclamped': 0x954b90,
    'Vector3_OrthoNormalize2': 0x954b30, 'Vector3_OrthoNormalize3': 0x954b40,
    'Vector3_RotateTowards': 0x954b50,
    'Random_Range': 0x969de0, 'Random_RandomRangeInt': 0x969d30,
    'Random_get_state': 0x969fd0, 'Random_set_state': 0x969ff0,
    # TransformAccessReadOnly hierarchy routines (fake TransformHierarchy:
    # +0x18 -> 48-byte TRS records [pos4 rot4 scale4], +0x20 -> int32 parents,
    # index 0 = top with parent -1).
    'CalculateGlobalPositionAndRotation': 0xdda60,
    'TransformPoint': 0x22db00,                 # (float3* ret, access*, const float3*)
    'InverseTransformPosition': 0x1fff50,       # (access*, float3* inout)
    'InverseTransformRotation': 0x200180,       # (access* parent, float4* inout)
    # GetScriptingRand() (RVA 0x7da940) is `mov rax, [rip+0xd76f19]; ret`: it returns
    # the Rand* stored in this data slot. The engine fills it at startup; the host
    # points it at its own 16-byte xorshift128 RandState (UnityEngine.Random.state).
    'ScriptingRandSlot': 0x7da947 + 0xd76f19,
}

# Direct registrations: managed extern name -> UnityPlayer binding function.
DIRECT_ICALLS = {
    'UnityEngine.Quaternion::Inverse_Injected': 'Quaternion_Inverse',
    'UnityEngine.Quaternion::LookRotation_Injected': 'Quaternion_LookRotation',
    'UnityEngine.Quaternion::AngleAxis_Injected': 'Quaternion_AngleAxis',
    'UnityEngine.Quaternion::Slerp_Injected': 'Quaternion_Slerp',
    'UnityEngine.Quaternion::SlerpUnclamped_Injected': 'Quaternion_SlerpUnclamped',
    'UnityEngine.Quaternion::Lerp_Injected': 'Quaternion_Lerp',
    'UnityEngine.Quaternion::LerpUnclamped_Injected': 'Quaternion_LerpUnclamped',
    'UnityEngine.Quaternion::FromToRotation_Injected': 'Quaternion_FromToRotation',
    'UnityEngine.Quaternion::Internal_FromEulerRad_Injected': 'Quaternion_FromEulerRad',
    'UnityEngine.Quaternion::Internal_ToEulerRad_Injected': 'Quaternion_ToEulerRad',
    'UnityEngine.Vector3::Slerp_Injected': 'Vector3_Slerp',
    'UnityEngine.Vector3::SlerpUnclamped_Injected': 'Vector3_SlerpUnclamped',
    'UnityEngine.Vector3::OrthoNormalize2': 'Vector3_OrthoNormalize2',
    'UnityEngine.Vector3::OrthoNormalize3': 'Vector3_OrthoNormalize3',
    'UnityEngine.Vector3::RotateTowards_Injected': 'Vector3_RotateTowards',
    'UnityEngine.Random::Range': 'Random_Range',
    'UnityEngine.Random::RandomRangeInt': 'Random_RandomRangeInt',
    'UnityEngine.Random::get_state_Injected': 'Random_get_state',
    'UnityEngine.Random::set_state_Injected': 'Random_set_state',
}

vp = C.c_void_p
F = np.float32


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


class _Mono:
    API = [
        ('mono_set_dirs', None, [C.c_char_p, C.c_char_p]),
        ('mono_set_assemblies_path', None, [C.c_char_p]),
        ('mono_config_parse', None, [C.c_char_p]),
        ('mono_parse_default_optimizations', C.c_int, [C.c_char_p]),
        ('mono_set_defaults', None, [C.c_int, C.c_int]),
        ('mono_jit_init_version', vp, [C.c_char_p, C.c_char_p]),
        ('mono_domain_assembly_open', vp, [vp, C.c_char_p]),
        ('mono_assembly_get_image', vp, [vp]),
        ('mono_class_from_name', vp, [vp, C.c_char_p, C.c_char_p]),
        ('mono_class_get_method_from_name', vp, [vp, C.c_char_p, C.c_int]),
        ('mono_class_get_field_from_name', vp, [vp, C.c_char_p]),
        ('mono_class_get_nested_types', vp, [vp, C.POINTER(vp)]),
        ('mono_class_get_name', C.c_char_p, [vp]),
        ('mono_class_vtable', vp, [vp, vp]),
        ('mono_class_value_size', C.c_int, [vp, C.POINTER(C.c_uint32)]),
        ('mono_class_get_parent', vp, [vp]),
        ('mono_object_new', vp, [vp, vp]),
        ('mono_object_get_class', vp, [vp]),
        ('mono_runtime_object_init', None, [vp]),
        ('mono_runtime_invoke', vp, [vp, vp, C.POINTER(vp), C.POINTER(vp)]),
        ('mono_object_unbox', vp, [vp]),
        ('mono_field_set_value', None, [vp, vp, vp]),
        ('mono_field_get_value', None, [vp, vp, vp]),
        ('mono_field_static_set_value', None, [vp, vp, vp]),
        ('mono_field_static_get_value', None, [vp, vp, vp]),
        ('mono_array_new', vp, [vp, vp, C.c_size_t]),
        ('mono_array_addr_with_size', vp, [vp, C.c_int, C.c_size_t]),
        ('mono_array_length', C.c_size_t, [vp]),
        ('mono_gchandle_new', C.c_uint32, [vp, C.c_int]),
        ('mono_add_internal_call', None, [C.c_char_p, vp]),
        ('mono_get_int32_class', vp, []),
        ('mono_get_single_class', vp, []),
        ('mono_get_byte_class', vp, []),
        ('mono_string_new', vp, [vp, C.c_char_p]),
        ('mono_object_to_string', vp, [vp, C.POINTER(vp)]),
        ('mono_string_to_utf8', C.c_char_p, [vp]),
    ]

    def __init__(self, dll):
        os.add_dll_directory(str(Path(dll).parent))
        self.lib = C.CDLL(str(dll))
        for name, res, args in self.API:
            fn = getattr(self.lib, name)
            fn.restype, fn.argtypes = res, args
            setattr(self, name[5:], fn)


class HostedRuntime:
    """Installed Mono + installed managed assemblies + in-process UnityPlayer."""

    def __init__(self, assemblies=('UnityEngine.CoreModule.dll', 'IL.dll')):
        if _sha256(UNITY_DLL) != UNITY_SHA:
            raise RuntimeError('Installed UnityPlayer.dll differs from the audited binary')
        os.add_dll_directory(str(GAME))
        self.unity = C.WinDLL(str(UNITY_DLL))
        self.ubase = self.unity._handle
        self.m = _Mono(MONO_DLL)
        m = self.m
        m.set_dirs(str(GAME / 'MonoBleedingEdge' / 'lib').encode(), str(GAME / 'MonoBleedingEdge' / 'etc').encode())
        m.set_assemblies_path(str(MANAGED).encode())
        m.config_parse(None)
        m.set_defaults(0, m.parse_default_optimizations(None))
        self._keep = []
        self.icall_errors = []
        self.scene = None
        self.transforms = {}            # Transform MonoObject -> (TransformScene, SceneTransform)
        self.components = {}            # Component MonoObject -> (TransformScene, SceneTransform)
        self.transform_icalls = False
        self.domain = m.jit_init_version(b'HS2HostedDrivers', b'v4.0.30319')
        self._rand = C.create_string_buffer(64)
        self.rand_addr = (C.addressof(self._rand) + 15) & ~15
        C.c_void_p.from_address(self.ubase + RVA['ScriptingRandSlot']).value = self.rand_addr
        self._register_direct_icalls()
        self.images = {}
        for name in assemblies:
            self.load(name)
        self.time = {'deltaTime': 0.0, 'time': 0.0, 'frameCount': 0}
        tm = self.time
        self.icall('UnityEngine.Time::get_deltaTime', C.c_float, [], lambda: tm['deltaTime'])
        self.icall('UnityEngine.Time::get_time', C.c_float, [], lambda: tm['time'])
        self.icall('UnityEngine.Time::get_frameCount', C.c_int, [], lambda: tm['frameCount'])

    # ---------------------------------------------------------------- native
    @property
    def random_state(self):
        return np.frombuffer(C.string_at(self.rand_addr, 16), np.uint32).copy()

    @random_state.setter
    def random_state(self, words):
        C.memmove(self.rand_addr, np.asarray(words, np.uint32).tobytes(), 16)

    def native(self, key, restype, *argtypes):
        return C.CFUNCTYPE(restype, *argtypes)(self.ubase + RVA[key])

    def _register_direct_icalls(self):
        for managed, key in DIRECT_ICALLS.items():
            self.m.add_internal_call(managed.encode(), vp(self.ubase + RVA[key]))

    def icall(self, name, restype, argtypes, fn):
        def guarded(*a):
            try:
                return fn(*a)
            except BaseException as exc:  # a Python error must not become a silent null/zero
                self.icall_errors.append((name, repr(exc)))
                raise
        cb = C.CFUNCTYPE(restype, *argtypes)(guarded)
        self._keep.append(cb)
        self.m.add_internal_call(name.encode(), C.cast(cb, vp))

    # ---------------------------------------------------------------- managed
    def load(self, name):
        asm = self.m.domain_assembly_open(self.domain, str(MANAGED / name).encode())
        if not asm:
            raise RuntimeError(f'cannot open {name}')
        self.images[name] = self.m.assembly_get_image(asm)
        return self.images[name]

    def klass(self, image, name, namespace=''):
        k = self.m.class_from_name(self.images[image], namespace.encode(), name.encode())
        if not k:
            raise KeyError(f'{namespace}.{name} in {image}')
        return k

    def nested(self, outer, name):
        it = vp()
        while True:
            k = self.m.class_get_nested_types(outer, C.byref(it))
            if not k:
                raise KeyError(name)
            if self.m.class_get_name(k).decode() == name:
                return k

    def new(self, klass, init=True):
        obj = self.m.object_new(self.domain, klass)
        self.m.gchandle_new(obj, 1)
        if init:
            self.m.runtime_object_init(obj)
        return obj

    def field(self, klass, name):
        k = klass
        while k:
            f = self.m.class_get_field_from_name(k, name.encode())
            if f:
                return f
            k = self.m.class_get_parent(k)
        raise KeyError(name)

    def set(self, obj, name, value, ctype=None):
        f = self.field(self.m.object_get_class(obj), name)
        self.m.field_set_value(obj, f, self._ptr(value, ctype))

    def set_ref(self, obj, name, ref):
        f = self.field(self.m.object_get_class(obj), name)
        # mono_field_set_value copies reference-type values from the pointer
        # argument itself (mono_copy_value with deref_pointer=FALSE).
        self.m.field_set_value(obj, f, vp(ref or None))

    def get(self, obj, name, ctype):
        f = self.field(self.m.object_get_class(obj), name)
        out = ctype()
        self.m.field_get_value(obj, f, C.addressof(out))
        return out

    def get_ref(self, obj, name):
        return self.get(obj, name, vp).value

    def set_static(self, klass, name, value, ctype=None):
        vt = self.m.class_vtable(self.domain, klass)
        self.m.field_static_set_value(vt, self.field(klass, name), self._ptr(value, ctype))

    @staticmethod
    def _ptr(value, ctype):
        if ctype is not None:
            buf = ctype(value) if not isinstance(value, C.Array) else value
            return C.cast(C.pointer(buf), vp) if not isinstance(buf, C.Array) else C.cast(buf, vp)
        arr = np.ascontiguousarray(value)
        return C.cast(arr.ctypes.data, vp) if arr.size else None

    def floats(self, obj, name, n):
        return np.frombuffer(bytes(self.get(obj, name, C.c_float * n)), F).copy()

    def array(self, eklass, items, elem_size=8, refs=True):
        arr = self.m.array_new(self.domain, eklass, len(items))
        self.m.gchandle_new(arr, 1)
        for i, it in enumerate(items):
            addr = self.m.array_addr_with_size(arr, elem_size, i)
            if refs:
                C.c_void_p.from_address(addr).value = it
            else:
                C.memmove(addr, bytes(it), elem_size)
        return arr

    def array_bytes(self, arr, elem_size):
        n = self.m.array_length(arr)
        return C.string_at(self.m.array_addr_with_size(arr, elem_size, 0), n * elem_size) if n else b''

    def method(self, klass, name, nargs):
        k = klass
        while k:
            mt = self.m.class_get_method_from_name(k, name.encode(), nargs)
            if mt:
                return mt
            k = self.m.class_get_parent(k)
        raise KeyError(name)

    def invoke(self, method, obj, *args):
        params = (vp * max(1, len(args)))(*[self._arg(a) for a in args])
        exc = vp()
        r = self.m.runtime_invoke(method, obj, params, C.pointer(exc))
        if self.icall_errors:
            errors, self.icall_errors = self.icall_errors, []
            raise RuntimeError(f'internal call failed during managed execution: {errors}')
        if exc.value:
            s = self.m.object_to_string(exc.value, None)
            raise RuntimeError('managed exception: ' + self.m.string_to_utf8(s).decode(errors='replace'))
        return r

    def _arg(self, a):
        if isinstance(a, int):            # object reference
            return a
        buf = np.ascontiguousarray(a)
        self._keep.append(buf)
        return buf.ctypes.data

    def unbox(self, obj, ctype):
        return ctype.from_address(self.m.object_unbox(obj)).value


class SceneTransform:
    __slots__ = ('index', 'name', 'parent', 'pos', 'rot', 'scl', 'obj', 'children')

    def __init__(self, index, name, parent, pos, rot, scl):
        self.index, self.name, self.parent = index, name, parent
        self.pos = np.asarray(pos, F).copy()
        self.rot = np.asarray(rot, F).copy()
        self.scl = np.asarray(scl, F).copy()
        self.obj = None
        self.children = []


class TransformScene:
    """Transform hierarchy served to the managed code through internal calls.

    Local TRS are float32 records; world-space queries and setters run
    UnityPlayer's own TransformAccess routines on a fake TransformHierarchy
    built from the queried transform's ancestor chain.
    """

    def __init__(self, rt: HostedRuntime):
        self.rt = rt
        self.items: list[SceneTransform] = []
        self.by_obj = {}
        self.k_transform = rt.klass('UnityEngine.CoreModule.dll', 'Transform', 'UnityEngine')
        self.k_gameobject = rt.klass('UnityEngine.CoreModule.dll', 'GameObject', 'UnityEngine')
        self.components = {}            # component MonoObject -> SceneTransform
        self.log = []
        n = rt.native
        self._gpr = n('CalculateGlobalPositionAndRotation', None, vp, vp, vp)
        self._tp = n('TransformPoint', vp, vp, vp, vp)
        self._itp = n('InverseTransformPosition', None, vp, vp)
        self._itr = n('InverseTransformRotation', None, vp, vp)
        rt.scene = self
        if not rt.transform_icalls:
            self._install()
            rt.transform_icalls = True

    # ----------------------------------------------------------- building
    def add(self, name, parent, pos, rot, scl):
        t = SceneTransform(len(self.items), name, parent, pos, rot, scl)
        self.items.append(t)
        if parent is not None:
            self.items[parent].children.append(t.index)
        t.obj = self.rt.new(self.k_transform, init=False)
        self.rt.set(t.obj, 'm_CachedPtr', np.array([t.index + 1], np.int64))
        self.by_obj[t.obj] = t
        self.rt.transforms[t.obj] = (self, t)
        return t

    def attach(self, component_obj, transform: SceneTransform):
        """A MonoBehaviour/Component living on `transform` (native object alive)."""
        self.rt.set(component_obj, 'm_CachedPtr', np.array([0x100000 + transform.index], np.int64))
        self.components[component_obj] = transform
        self.rt.components[component_obj] = (self, transform)

    # ----------------------------------------------------------- native math
    def _hierarchy(self, t: SceneTransform):
        chain = []
        i = t.index
        while i is not None:
            chain.append(self.items[i])
            i = self.items[i].parent
        chain.reverse()                       # index 0 = top
        trs = b''.join(np.concatenate([c.pos, [0], c.rot, c.scl, [0]]).astype(F).tobytes() for c in chain)
        parents = np.array([-1] + list(range(len(chain) - 1)), np.int32)
        rec = C.create_string_buffer(len(trs) + 128)
        rec_addr = (C.addressof(rec) + 63) & ~63
        C.memmove(rec_addr, trs, len(trs))
        bufs = [rec, C.create_string_buffer(parents.tobytes(), parents.nbytes + 16)]
        hier = C.create_string_buffer(struct.pack('<5Q', 0, 0, 0, rec_addr, C.addressof(bufs[1])) + bytes(0x100))
        bufs.append(hier)
        access = C.create_string_buffer(struct.pack('<QI4x', C.addressof(hier), len(chain) - 1))
        bufs.append(access)
        return access, bufs

    @staticmethod
    def _vec4(values=(0.0, 0.0, 0.0, 0.0)):
        # The SSE routines use aligned memory operands (mulps/pxor xmm, [mem]):
        # every float4 argument buffer is 16-byte aligned, as in the engine.
        raw = C.create_string_buffer(48)
        addr = (C.addressof(raw) + 15) & ~15
        v = list(values) + [0.0] * (4 - len(values))
        C.memmove(addr, np.asarray(v, F).tobytes(), 16)
        return raw, addr

    @staticmethod
    def _read4(addr):
        return np.frombuffer(C.string_at(addr, 16), F).copy()

    def world(self, t: SceneTransform):
        access, keep = self._hierarchy(t)
        pk, pos = self._vec4()
        rk, rot = self._vec4()
        self._gpr(C.addressof(access), pos, rot)
        return self._read4(pos)[:3], self._read4(rot)

    def transform_point(self, t, p):
        access, keep = self._hierarchy(t)
        sk, src = self._vec4([float(v) for v in p])
        ok, out = self._vec4()
        self._tp(out, C.addressof(access), src)
        return self._read4(out)[:3]

    def inverse_transform_point(self, t, p):
        access, keep = self._hierarchy(t)
        ik, io = self._vec4([float(v) for v in p])
        self._itp(C.addressof(access), io)
        return self._read4(io)[:3]

    def set_world_rotation(self, t, q):
        if t.parent is not None:
            access, keep = self._hierarchy(self.items[t.parent])
            ik, io = self._vec4([float(v) for v in q])
            self._itr(C.addressof(access), io)
            q = self._read4(io)
        t.rot = normalize_quaternion(q)

    # ----------------------------------------------------------- icalls
    def _t(self, obj):
        owner = self.rt.transforms.get(obj)
        if owner is None:
            raise RuntimeError('unknown Transform object')
        return owner

    def _install(self):
        rt = self.rt
        FP = C.POINTER(C.c_float)

        def out3(ptr, v):
            for i in range(3):
                ptr[i] = float(v[i])

        def out4(ptr, v):
            for i in range(4):
                ptr[i] = float(v[i])

        def rd(ptr, n):
            return np.array([ptr[i] for i in range(n)], F)

        def get_position(this, ret):
            sc, t = self._t(this)
            out3(ret, sc.world(t)[0])

        def get_rotation(this, ret):
            sc, t = self._t(this)
            out4(ret, sc.world(t)[1])

        def set_rotation(this, v):
            sc, t = self._t(this)
            sc.set_world_rotation(t, rd(v, 4))
            sc.log.append(('set_rotation', t.name))

        def get_local_rotation(this, ret):
            out4(ret, self._t(this)[1].rot)

        def set_local_rotation(this, v):
            sc, t = self._t(this)
            t.rot = normalize_quaternion(rd(v, 4))
            sc.log.append(('set_localRotation', t.name))

        def get_local_position(this, ret):
            out3(ret, self._t(this)[1].pos)

        def set_local_position(this, v):
            sc, t = self._t(this)
            t.pos = rd(v, 3)
            sc.log.append(('set_localPosition', t.name))

        def get_local_scale(this, ret):
            out3(ret, self._t(this)[1].scl)

        def set_local_scale(this, v):
            sc, t = self._t(this)
            t.scl = rd(v, 3)
            sc.log.append(('set_localScale', t.name))

        def transform_point(this, p, ret):
            sc, t = self._t(this)
            out3(ret, sc.transform_point(t, rd(p, 3)))

        def inverse_transform_point(this, p, ret):
            sc, t = self._t(this)
            out3(ret, sc.inverse_transform_point(t, rd(p, 3)))

        def get_parent(this):
            sc, t = self._t(this)
            return sc.items[t.parent].obj if t.parent is not None else None

        def get_transform(this):
            if this in self.rt.transforms:
                return this
            sc, t = self.rt.components[this]
            return t.obj

        def unsupported(name):
            def fn(*a):
                raise RuntimeError(f'unsupported internal call {name}')
            return fn

        T = 'UnityEngine.Transform::'
        rt.icall(T + 'get_position_Injected', None, [vp, FP], get_position)
        rt.icall(T + 'get_rotation_Injected', None, [vp, FP], get_rotation)
        rt.icall(T + 'set_rotation_Injected', None, [vp, FP], set_rotation)
        rt.icall(T + 'get_localRotation_Injected', None, [vp, FP], get_local_rotation)
        rt.icall(T + 'set_localRotation_Injected', None, [vp, FP], set_local_rotation)
        rt.icall(T + 'get_localPosition_Injected', None, [vp, FP], get_local_position)
        rt.icall(T + 'set_localPosition_Injected', None, [vp, FP], set_local_position)
        rt.icall(T + 'get_localScale_Injected', None, [vp, FP], get_local_scale)
        rt.icall(T + 'set_localScale_Injected', None, [vp, FP], set_local_scale)
        rt.icall(T + 'TransformPoint_Injected', None, [vp, FP, FP], transform_point)
        rt.icall(T + 'InverseTransformPoint_Injected', None, [vp, FP, FP], inverse_transform_point)
        rt.icall(T + 'GetParent', vp, [vp], get_parent)
        rt.icall(T + 'get_parentInternal', vp, [vp], get_parent)
        rt.icall('UnityEngine.Component::get_transform', vp, [vp], get_transform)
        for name in ('set_position_Injected', 'TransformDirection_Injected', 'InverseTransformDirection_Injected',
                     'get_lossyScale_Injected'):
            rt.icall(T + name, None, [vp, FP, FP], unsupported(T + name))
