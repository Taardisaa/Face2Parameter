// Read-only access to UnityPlayer 2018.4.11f1 native state that the managed API
// does not expose (temporary RT pool, shader property sheets, global keyword
// bits, camera previous VP, renderer visibility flags, build/caps settings).
// All reads go through ReadProcessMemory on the own process, so an unexpected
// layout yields an error record instead of a fault. Nothing is written, no
// native function is called. Layout constants and their provenance are in
// ProbeCore.cs (UnityPlayerLayout); they are used only after the loaded
// UnityPlayer.dll hash equals the binary they were recovered from.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using UnityEngine;

namespace HS2Codex.FramePairProbe
{
    internal sealed class ProcessMemory : IMemory
    {
        [DllImport("kernel32.dll")] static extern IntPtr GetCurrentProcess();
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool ReadProcessMemory(IntPtr process, IntPtr address, byte[] buffer, IntPtr size, out IntPtr read);

        readonly IntPtr _process = GetCurrentProcess();

        public bool Read(ulong address, byte[] buffer, int count)
        {
            if (address == 0 || count < 0 || count > buffer.Length) return false;
            IntPtr read;
            return ReadProcessMemory(_process, new IntPtr((long)address), buffer, new IntPtr(count), out read) && read.ToInt64() == count;
        }
    }

    internal sealed class UnityNative
    {
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern IntPtr GetModuleHandleW(string name);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern uint GetModuleFileNameW(IntPtr module, StringBuilder name, uint size);
        [DllImport("kernel32.dll")] static extern IntPtr TlsGetValue(uint index);

        static readonly FieldInfo CachedPtr = typeof(UnityEngine.Object).GetField("m_CachedPtr", BindingFlags.Instance | BindingFlags.NonPublic);
        static readonly FieldInfo BlockPtr = typeof(MaterialPropertyBlock).GetField("m_Ptr", BindingFlags.Instance | BindingFlags.NonPublic);

        public readonly IMemory Memory = new ProcessMemory();
        public readonly ulong Base;
        public readonly string ModulePath, ModuleSha256;
        public string[] PropertyNames, BuiltinVectorNames, BuiltinMatrixNames, BuiltinTexEnvNames;
        public ushort[] BuiltinVectorArraySizes;
        public SortedDictionary<int, string> KeywordNames;
        public readonly List<string> Validation = new List<string>();
        public bool Valid = true;

        UnityNative(ulong b, string path, string sha) { Base = b; ModulePath = path; ModuleSha256 = sha; }

        public static UnityNative Open(string expectedSha)
        {
            IntPtr module = GetModuleHandleW("UnityPlayer.dll");
            if (module == IntPtr.Zero) throw new InvalidOperationException("UnityPlayer.dll is not loaded");
            var name = new StringBuilder(1024);
            GetModuleFileNameW(module, name, 1024);
            string sha;
            using (var h = SHA256.Create()) sha = Mem.Hex(h.ComputeHash(File.ReadAllBytes(name.ToString())));
            if (sha != expectedSha || sha != UnityPlayerLayout.Sha256)
                throw new InvalidOperationException("Loaded UnityPlayer.dll " + sha + " is not the binary the native layouts were recovered from");
            if (CachedPtr == null || BlockPtr == null) throw new InvalidOperationException("UnityEngine m_CachedPtr/m_Ptr fields missing");
            var n = new UnityNative((ulong)module.ToInt64(), name.ToString(), sha);
            n.RefreshNames();
            n.BuiltinVectorNames = n.StaticNames(UnityPlayerLayout.BuiltinVectorNames, UnityPlayerLayout.BuiltinVectorCount);
            n.BuiltinMatrixNames = n.StaticNames(UnityPlayerLayout.BuiltinMatrixNames, UnityPlayerLayout.BuiltinMatrixCount);
            n.BuiltinTexEnvNames = n.StaticNames(UnityPlayerLayout.BuiltinTexEnvNames, UnityPlayerLayout.BuiltinTexEnvCount);
            ulong table = Mem.U64(n.Memory, Mem.U64(n.Memory, n.Base + UnityPlayerLayout.ParamNameLookupsPtr) + UnityPlayerLayout.ParamNameLookupsVectorArraySizes);
            var sb = Mem.Bytes(n.Memory, table, 2 * UnityPlayerLayout.BuiltinVectorCount);
            n.BuiltinVectorArraySizes = new ushort[UnityPlayerLayout.BuiltinVectorCount];
            for (int i = 0; i < n.BuiltinVectorArraySizes.Length; i++) n.BuiltinVectorArraySizes[i] = BitConverter.ToUInt16(sb, 2 * i);
            n.Validate();
            return n;
        }

        public void RefreshNames()
        {
            PropertyNames = NameTables.PropertyNames(Memory, Base + UnityPlayerLayout.PropertyNameTablePtr);
            KeywordNames = NameTables.Keywords(Memory, Base + UnityPlayerLayout.KeywordMapPtr);
        }

        // Cross-checks between native reads and managed getters that are side-effect free
        // for names that already exist (PropertyToID of a registered name only looks it up).
        void Validate()
        {
            Check("property name table: PropertyToID(_MainTex) names _MainTex", () =>
            {
                int id = Shader.PropertyToID("_MainTex");
                return id >= 0 && id < PropertyNames.Length && PropertyNames[id] == "_MainTex";
            });
            // ScriptingGraphicsCaps::GetGraphicsMemorySize 0x2f8750: cvttss2si eax,[caps+0xd8] (a float).
            Check("graphics caps video memory (float, truncated) equals SystemInfo.graphicsMemorySize", () =>
                (int)BitConverter.ToSingle(Mem.Bytes(Memory, Caps() + UnityPlayerLayout.CapsVideoMemoryMB, 4), 0) == SystemInfo.graphicsMemorySize);
            Check("keyword table is non-empty and indices are unique", () => KeywordNames.Count > 0);
            Check("global keyword bits readable", () => { GlobalKeywordBits(); return true; });
            Check("global property sheet readable", () => { GlobalSheet(); return true; });
            Check("temporary RT pool readable", () => { Pool(); return true; });
        }

        // Second validation stage, with the live scene: every native interpretation that also
        // has a side-effect-free managed counterpart must agree with it bit for bit
        // (managed getters are called with native ids or registered names only).
        public void ValidateWithScene()
        {
            var rts = new Dictionary<int, RenderTexture>();
            foreach (var rt in Resources.FindObjectsOfTypeAll<RenderTexture>()) rts[rt.GetInstanceID()] = rt;
            Check("temporary RT pool: every entry names a RenderTexture whose +0x138 points back to it", () =>
            {
                var pool = Pool();
                int n = 0;
                foreach (var e in pool.Keys.SelectMany(k => k.FreeInAcquisitionOrder).Concat(pool.InUse))
                {
                    RenderTexture rt;
                    if (!rts.TryGetValue(e.InstanceId, out rt)) return false;
                    if (Mem.U64(Memory, Ptr(rt) + UnityPlayerLayout.RenderTextureTempEntry) != e.Address) return false;
                    n++;
                }
                Validation.Add("info: pool entries checked " + n);
                return true;
            });
            var renderers = Resources.FindObjectsOfTypeAll<Renderer>().Where(r => r.gameObject.scene.IsValid()).ToArray();
            Check("instance id offset: native +offset equals GetInstanceID for every scene renderer", () =>
                InstanceIdOffset > 0 && renderers.All(r => Mem.I32(Memory, Ptr(r) + (ulong)InstanceIdOffset) == r.GetInstanceID()));
            Check("SharedMaterialData (Material+0xb0) +0xdc equals Material.GetInstanceID for every renderer material", () =>
                renderers.SelectMany(r => r.sharedMaterials).Where(m => m != null).Distinct()
                    .All(m => Mem.I32(Memory, Mem.U64(Memory, Ptr(m) + UnityPlayerLayout.MaterialPropertiesHolder) + UnityPlayerLayout.SmdInstanceId) == m.GetInstanceID()));
            Check("RenderTexture+0xd0 equals RenderTexture.antiAliasing for every RenderTexture", () =>
                rts.Values.All(t => Mem.I32(Memory, Ptr(t) + UnityPlayerLayout.RenderTextureAntiAliasing) == t.antiAliasing));
            // BaseRenderer's matrix is refreshed by RendererUpdateManager at culling, so a MeshRenderer
            // moved by a script after the last cull legitimately differs. A wrong layout matches none;
            // require >= 90 % bit-exact matches and record every mismatch.
            Check("BaseRenderer (Renderer+0x38) +0x08 equals MeshRenderer.localToWorldMatrix bits (>= 90 % of enabled MeshRenderers)", () =>
            {
                int n = 0, equal = 0;
                var mismatched = new List<string>();
                foreach (var r in renderers.OfType<MeshRenderer>())
                {
                    if (!r.enabled || !r.gameObject.activeInHierarchy) continue;
                    var m = r.localToWorldMatrix;
                    var raw = Mem.Bytes(Memory, Ptr(r) + UnityPlayerLayout.RendererBase + UnityPlayerLayout.BaseL2W, 64);
                    bool same = true;
                    for (int i = 0; i < 16; i++) if (!SameBits(m[i], raw, 4 * i)) { same = false; break; }   // Matrix4x4 indexer = column-major memory order
                    n++;
                    if (same) equal++; else mismatched.Add(r.name + "#" + r.GetInstanceID());
                }
                Validation.Add("info: BaseRenderer localToWorld bit-equal for " + equal + "/" + n + " MeshRenderers; mismatched: " + string.Join(", ", mismatched.ToArray()));
                return n > 0 && equal * 10 >= n * 9;
            });
            Check("renderer visibility bit equals Renderer.isVisible", () =>
                renderers.All(r => ((Mem.U32(Memory, Ptr(r) + UnityPlayerLayout.RendererFlags) & UnityPlayerLayout.RendererVisibleBit) != 0
                                    && Mem.I32(Memory, Ptr(r) + UnityPlayerLayout.RendererSceneHandle) != -1) == r.isVisible));
            Check("material sheets: raw floats/vectors (no gamma flag) and texture ids equal Material getters", () =>
            {
                int compared = 0;
                foreach (var m in renderers.SelectMany(r => r.sharedMaterials).Where(m => m != null).Distinct())
                {
                    var sheet = MaterialSheet(m);
                    if (sheet == null) continue;
                    foreach (var e in sheet.Entries)
                    {
                        if (e.ArraySize > 1) continue;
                        if (e.Type == "float" && !e.GammaFlag) { if (!SameBits(m.GetFloat(e.NameId), e.Value, 0)) return false; compared++; }
                        else if (e.Type == "vector" && !e.GammaFlag)
                        {
                            var v = m.GetVector(e.NameId);
                            if (!SameBits(v.x, e.Value, 0) || !SameBits(v.y, e.Value, 4) || !SameBits(v.z, e.Value, 8) || !SameBits(v.w, e.Value, 12)) return false;
                            compared++;
                        }
                        else if (e.Type == "texture")
                        {
                            var t = m.GetTexture(e.NameId);
                            int id = BitConverter.ToInt32(e.Value, 0);
                            if ((t == null) != (id == 0) || (t != null && TextureId(t) != id)) return false;
                            compared++;
                        }
                    }
                }
                Validation.Add("info: material sheet values compared " + compared);
                return compared > 0;
            });
            Check("global sheet: raw floats/vectors equal Shader.GetGlobal* by id", () =>
            {
                int compared = 0;
                foreach (var e in GlobalSheet().Entries)
                {
                    if (e.ArraySize > 1) continue;
                    if (e.Type == "float") { if (!SameBits(Shader.GetGlobalFloat(e.NameId), e.Value, 0)) return false; compared++; }
                    else if (e.Type == "vector")
                    {
                        var v = Shader.GetGlobalVector(e.NameId);
                        if (!SameBits(v.x, e.Value, 0) || !SameBits(v.y, e.Value, 4) || !SameBits(v.z, e.Value, 8) || !SameBits(v.w, e.Value, 12)) return false;
                        compared++;
                    }
                }
                Validation.Add("info: global sheet values compared " + compared);
                return compared > 0;
            });
            Check("built-in vectors/matrices equal Shader.GetGlobalVector/Matrix of built-in ids", () =>
            {
                var v = BuiltinVectors();
                var mm = BuiltinMatrices();
                for (int i = 0; i < UnityPlayerLayout.BuiltinVectorCount; i++)
                {
                    var g = Shader.GetGlobalVector(unchecked((int)0x40000000u | i));
                    if (!SameBits(g.x, v, 16 * i) || !SameBits(g.y, v, 16 * i + 4) || !SameBits(g.z, v, 16 * i + 8) || !SameBits(g.w, v, 16 * i + 12)) return false;
                }
                for (int i = 0; i < UnityPlayerLayout.BuiltinMatrixCount; i++)
                {
                    var g = Shader.GetGlobalMatrix(unchecked((int)0x80000000u | i));
                    for (int c = 0; c < 16; c++) if (!SameBits(g[c], mm, 64 * i + 4 * c)) return false;   // Matrix4x4 indexer = memory order
                }
                int timeId = Shader.PropertyToID("_Time");   // registered name: lookup only
                Validation.Add("info: PropertyToID(_Time)=0x" + timeId.ToString("x8") + " (built-in vector index of _Time: " + Array.IndexOf(BuiltinVectorNames, "_Time") + ")");
                return true;
            });
            Check("global keyword bits equal Shader.IsKeywordEnabled for every registered keyword name", () =>
            {
                var bits = GlobalKeywordBits();
                foreach (var kv in KeywordNames)   // registered names only: keywords::Create finds, never adds
                    if (kv.Key >= 0 && kv.Key < 256 && (((bits[kv.Key >> 6] >> (kv.Key & 63)) & 1) != 0) != Shader.IsKeywordEnabled(kv.Value)) return false;
                return true;
            });
        }

        static bool SameBits(float managed, byte[] raw, int offset)
        {
            return BitConverter.ToInt32(BitConverter.GetBytes(managed), 0) == BitConverter.ToInt32(raw, offset);
        }

        void Check(string what, Func<bool> f)
        {
            try
            {
                bool ok = f();
                Validation.Add((ok ? "ok: " : "FAILED: ") + what);
                if (!ok) Valid = false;
            }
            catch (Exception e) { Validation.Add("FAILED: " + what + ": " + e.Message); Valid = false; }
        }

        string[] StaticNames(ulong rva, int count)
        {
            var ptrs = Mem.Bytes(Memory, Base + rva, 8 * count);
            var names = new string[count];
            for (int i = 0; i < count; i++) names[i] = Mem.CString(Memory, BitConverter.ToUInt64(ptrs, 8 * i), 256);
            return names;
        }

        public string Name(int id)
        {
            uint kind = (uint)id & 0xC0000000u;
            int i = id & 0x3FFFFFFF;
            if (kind == 0x40000000u && i < BuiltinVectorNames.Length) return BuiltinVectorNames[i];
            if (kind == 0x80000000u && i < BuiltinMatrixNames.Length) return BuiltinMatrixNames[i];
            if (kind == 0xC0000000u && i < BuiltinTexEnvNames.Length) return BuiltinTexEnvNames[i];
            return NameTables.Resolve(PropertyNames, id);
        }

        // Main-thread GfxDevice (the client device under multithreaded rendering), as
        // GetGfxDevice returns it to ShaderScripting::GetGlobalVector on this thread.
        ulong GfxDevice()
        {
            uint index = Mem.U32(Memory, Base + UnityPlayerLayout.GfxDeviceTlsIndex);
            ulong p = (ulong)TlsGetValue(index).ToInt64();
            if (p == 0) throw new NativeReadError("GfxDevice TLS slot is empty on this thread");
            return p;
        }

        public byte[] BuiltinVectors() { return Mem.Bytes(Memory, GfxDevice() + UnityPlayerLayout.GfxBuiltinVectors, 16 * UnityPlayerLayout.BuiltinVectorCount); }
        public byte[] BuiltinMatrices() { return Mem.Bytes(Memory, GfxDevice() + UnityPlayerLayout.GfxBuiltinMatrices, 64 * UnityPlayerLayout.BuiltinMatrixCount); }

        public void WriteBuiltins(JsonOut w)
        {
            var v = BuiltinVectors();
            var m = BuiltinMatrices();
            w.Begin('{');
            w.Key("source"); w.Str("main-thread GfxDevice built-in params (+0x11e8 vectors, +0x1978 matrices), what Shader.GetGlobalVector/Matrix return for built-in ids");
            w.Key("vectors"); w.Begin('[');
            for (int i = 0; i < UnityPlayerLayout.BuiltinVectorCount; i++)
            {
                w.Begin('{'); w.Key("index"); w.Int(i); w.Key("name"); w.Str(BuiltinVectorNames[i]); w.Key("array_size"); w.Int(BuiltinVectorArraySizes[i]);
                w.Key("value_f32"); w.Begin('['); for (int c = 0; c < 4; c++) w.Num(BitConverter.ToSingle(v, 16 * i + 4 * c)); w.End(']'); w.End('}');
            }
            w.End(']');
            w.Key("matrices_memory_order"); w.Begin('[');
            for (int i = 0; i < UnityPlayerLayout.BuiltinMatrixCount; i++)
            {
                w.Begin('{'); w.Key("index"); w.Int(i); w.Key("name"); w.Str(BuiltinMatrixNames[i]);
                w.Key("value_f32"); w.Begin('['); for (int c = 0; c < 16; c++) w.Num(BitConverter.ToSingle(m, 64 * i + 4 * c)); w.End(']'); w.End('}');
            }
            w.End(']');
            w.Key("texenv_names_storage_not_read"); SceneState.Names(w, BuiltinTexEnvNames);
            w.End('}');
        }

        public static ulong Ptr(UnityEngine.Object o)
        {
            if ((object)o == null) return 0;
            return (ulong)((IntPtr)CachedPtr.GetValue(o)).ToInt64();
        }

        public static ulong Ptr(MaterialPropertyBlock b) { return b == null ? 0 : (ulong)((IntPtr)BlockPtr.GetValue(b)).ToInt64(); }

        ulong Caps() { return Mem.U64(Memory, Base + UnityPlayerLayout.GraphicsCapsPtr); }
        ulong Manager(int index) { return Mem.U64(Memory, Base + UnityPlayerLayout.ManagerContext + (ulong)(8 * index)); }
        ulong PassContext() { return Mem.U64(Memory, Base + UnityPlayerLayout.SharedPassContextPtr); }

        public PoolSnapshot Pool()
        {
            return PoolWalker.Read(Memory, Mem.U64(Memory, Base + UnityPlayerLayout.RenderBufferManagerPtr));
        }

        public ulong[] GlobalKeywordBits()
        {
            var b = Mem.Bytes(Memory, PassContext() + UnityPlayerLayout.PassContextKeywords, 32);
            return new[] { BitConverter.ToUInt64(b, 0), BitConverter.ToUInt64(b, 8), BitConverter.ToUInt64(b, 16), BitConverter.ToUInt64(b, 24) };
        }

        public SheetSnapshot GlobalSheet() { return SheetReader.Read(Memory, PassContext() + UnityPlayerLayout.PassContextSheet); }

        // null when the runtime sheet has not been built yet (Material::BuildProperties is lazy;
        // the probe never forces it).
        public SheetSnapshot MaterialSheet(Material m)
        {
            ulong holder = Mem.U64(Memory, Ptr(m) + UnityPlayerLayout.MaterialPropertiesHolder);
            bool built = (Mem.U8(Memory, holder + UnityPlayerLayout.HolderBuiltFlags) & 1) != 0 && Mem.U64(Memory, holder + UnityPlayerLayout.HolderBuiltPtr) != 0;
            return built ? SheetReader.Read(Memory, holder + UnityPlayerLayout.HolderSheet) : null;
        }

        public SheetSnapshot BlockSheet(MaterialPropertyBlock b) { return SheetReader.Read(Memory, Ptr(b)); }

        public void WriteKeywords(JsonOut w)
        {
            var bits = GlobalKeywordBits();
            w.Begin('{');
            w.Key("bits_hex"); w.Begin('['); foreach (var x in bits) w.Str("0x" + x.ToString("x16")); w.End(']');
            w.Key("enabled"); w.Begin('[');
            for (int i = 0; i < 256; i++)
                if (((bits[i >> 6] >> (i & 63)) & 1) != 0)
                {
                    string n;
                    w.Begin('{'); w.Key("index"); w.Int(i); w.Key("name"); w.Str(KeywordNames.TryGetValue(i, out n) ? n : null); w.End('}');
                }
            w.End(']');
            w.End('}');
        }

        public void WriteCamera(JsonOut w, Camera c)
        {
            ulong p = Ptr(c);
            w.Begin('{');
            w.Key("implicit_projection_flag_0x4ec"); w.Int(Mem.I32(Memory, p + UnityPlayerLayout.CameraImplicitProjection));
            w.Key("previous_view_projection_0x598"); Matrix(w, p + UnityPlayerLayout.CameraPreviousViewProjection);
            w.Key("view_projection_0x5d8"); Matrix(w, p + UnityPlayerLayout.CameraViewProjection);
            w.Key("matrix_order"); w.Str("Matrix4x4f memory order (column-major, m00,m10,m20,m30,m01,...)");
            ulong target = Mem.U64(Memory, p + UnityPlayerLayout.CameraCurrentTarget);
            w.Key("current_target_0x558"); w.Str("0x" + target.ToString("x"));
            if (target != 0)
            {
                w.Key("current_target_instance_id"); w.Int(Mem.I32(Memory, target + (ulong)InstanceIdOffset));
                w.Key("current_target_antialiasing_0xd0"); w.Int(Mem.I32(Memory, target + UnityPlayerLayout.RenderTextureAntiAliasing));
            }
            w.Key("gfxdevice_single_pass_stereo_0x21e8"); w.Int(Mem.I32(Memory, GfxDevice() + UnityPlayerLayout.GfxSinglePassStereo));
            w.End('}');
        }

        // UnityEngine.Object.GetOffsetOfInstanceIDInCPlusPlusObject (internal call used by
        // GetInstanceID itself) - read once; pure.
        public static readonly int InstanceIdOffset = (int)typeof(UnityEngine.Object)
            .GetMethod("GetOffsetOfInstanceIDInCPlusPlusObject", BindingFlags.Static | BindingFlags.NonPublic).Invoke(null, null);

        void Matrix(JsonOut w, ulong a)
        {
            var b = Mem.Bytes(Memory, a, 64);
            w.Begin('['); for (int i = 0; i < 16; i++) w.Num(BitConverter.ToSingle(b, 4 * i)); w.End(']');
        }

        public void WriteRenderer(JsonOut w, Renderer r)
        {
            ulong p = Ptr(r);
            uint flags = Mem.U32(Memory, p + UnityPlayerLayout.RendererFlags);
            int handle = Mem.I32(Memory, p + UnityPlayerLayout.RendererSceneHandle);
            w.Begin('{');
            w.Key("flags_0x128"); w.Str("0x" + flags.ToString("x8"));
            w.Key("scene_handle_0x160"); w.Int(handle);
            w.Key("is_visible_bit"); w.Bool((flags & UnityPlayerLayout.RendererVisibleBit) != 0 && handle != -1);
            ulong b = p + UnityPlayerLayout.RendererBase;
            w.Key("base_renderer_local_to_world_0x08"); Matrix(w, b + UnityPlayerLayout.BaseL2W);
            w.Key("base_renderer_previous_local_to_world_0x48"); Matrix(w, b + UnityPlayerLayout.BasePrevL2W);
            w.Key("base_renderer_transform_update_frame_0xbc"); w.Raw(Mem.U32(Memory, b + UnityPlayerLayout.BaseUpdateFrame).ToString(CultureInfo.InvariantCulture));
            w.Key("base_renderer_flags_0xf0"); w.Str("0x" + Mem.U32(Memory, b + UnityPlayerLayout.BaseFlags).ToString("x8"));
            w.Key("base_renderer_flatten_block_hex"); w.Str(Mem.Hex(Mem.Bytes(Memory, b + UnityPlayerLayout.BaseBlockStart, UnityPlayerLayout.BaseBlockLength)));
            if (r is SkinnedMeshRenderer) WriteSkinFields(w, r, p);
            w.End('}');
        }

        void WriteSkinFields(JsonOut w, Renderer r, ulong p)
        {
            w.Key("smr_has_last_position_data_0x255"); w.Int(Mem.U8(Memory, p + UnityPlayerLayout.SmrHasLastPosition));
            w.Key("smr_force_matrix_recalc_0x256"); w.Int(Mem.U8(Memory, p + UnityPlayerLayout.SmrForceMatrixRecalc));
            w.Key("smr_dirty_0x28c"); w.Int(Mem.U8(Memory, p + UnityPlayerLayout.SmrDirty));
            ulong s0 = Mem.U64(Memory, p + UnityPlayerLayout.SmrSkinBuffer), s1 = Mem.U64(Memory, p + UnityPlayerLayout.SmrPrevSkinBuffer);
            w.Key("smr_skin_buffer_0x258"); w.Str("0x" + s0.ToString("x"));
            w.Key("smr_prev_skin_buffer_0x260"); w.Str("0x" + s1.ToString("x"));
            if (SkinSlots != null)
            {
                w.Key("skin_buffer_holds_frame"); w.Int(SkinSlots.HoldsFrame(s0));
                w.Key("prev_skin_buffer_holds_frame"); w.Int(SkinSlots.HoldsFrame(s1));
            }
        }

        public SkinSlotTracker SkinSlots;

        public void WriteSkinOnly(JsonOut w, Renderer r) { WriteSkinFields(w, r, Ptr(r)); }

        public ulong SkinBuffer(SkinnedMeshRenderer r) { return Mem.U64(Memory, Ptr(r) + UnityPlayerLayout.SmrSkinBuffer); }

        public void WriteSharedMaterialData(JsonOut w, Material m)
        {
            ulong smd = Mem.U64(Memory, Ptr(m) + UnityPlayerLayout.MaterialPropertiesHolder);
            w.Begin('{');
            w.Key("caster_pass_values_hash_0xd0"); w.Raw(Mem.U32(Memory, smd + UnityPlayerLayout.SmdCasterPassHash).ToString(CultureInfo.InvariantCulture));
            w.Key("state_block_values_hash_0xd4"); w.Raw(Mem.U32(Memory, smd + UnityPlayerLayout.SmdStateBlockHash).ToString(CultureInfo.InvariantCulture));
            w.Key("flags_0xd8"); w.Str("0x" + Mem.U32(Memory, smd + UnityPlayerLayout.SmdFlags).ToString("x8"));
            w.Key("material_instance_id_0xdc"); w.Int(Mem.I32(Memory, smd + UnityPlayerLayout.SmdInstanceId));
            w.End('}');
        }

        public void WriteRenderTextureCompact(JsonOut w, RenderTexture rt)
        {
            ulong p = Ptr(rt);
            w.Begin('{');
            w.Key("id"); w.Int(rt.GetInstanceID()); w.Key("name"); w.Str(rt.name);
            w.Key("is_created"); w.Bool(rt.IsCreated());
            var d = rt.descriptor;
            w.Key("descriptor"); w.Begin('{');
            w.Key("width"); w.Int(d.width); w.Key("height"); w.Int(d.height); w.Key("msaaSamples"); w.Int(d.msaaSamples);
            w.Key("volumeDepth"); w.Int(d.volumeDepth); w.Key("colorFormat"); w.Str(d.colorFormat.ToString()); w.Key("depthBufferBits"); w.Int(d.depthBufferBits);
            w.Key("dimension"); w.Str(d.dimension.ToString()); w.Key("shadowSamplingMode"); w.Str(d.shadowSamplingMode.ToString());
            w.Key("vrUsage"); w.Str(d.vrUsage.ToString()); w.Key("flags"); w.Str(d.flags.ToString()); w.Key("memoryless"); w.Str(d.memoryless.ToString());
            w.Key("sRGB"); w.Bool(d.sRGB); w.Key("useMipMap"); w.Bool(d.useMipMap); w.Key("autoGenerateMips"); w.Bool(d.autoGenerateMips);
            w.Key("enableRandomWrite"); w.Bool(d.enableRandomWrite); w.Key("bindMS"); w.Bool(d.bindMS);
            w.End('}');
            w.Key("filterMode"); w.Str(rt.filterMode.ToString()); w.Key("anisoLevel"); w.Int(rt.anisoLevel); w.Key("mipMapBias"); w.Num(rt.mipMapBias);
            w.Key("wrapModeU"); w.Str(rt.wrapModeU.ToString()); w.Key("wrapModeV"); w.Str(rt.wrapModeV.ToString()); w.Key("wrapModeW"); w.Str(rt.wrapModeW.ToString());
            w.Key("native"); w.Begin('{');
            w.Key("texture_id_0x50"); w.Int(Mem.I32(Memory, p + UnityPlayerLayout.TextureId));
            w.Key("antialiasing_0xd0"); w.Int(Mem.I32(Memory, p + UnityPlayerLayout.RenderTextureAntiAliasing));
            ulong entry = Mem.U64(Memory, p + UnityPlayerLayout.RenderTextureTempEntry);
            w.Key("temp_entry_0x138"); w.Str("0x" + entry.ToString("x"));
            var desc = Mem.Bytes(Memory, p + UnityPlayerLayout.RenderTextureDesc, 4 * UnityPlayerLayout.DescDwords);
            w.Key("desc_key_0xc8"); w.Begin('['); for (int i = 0; i < UnityPlayerLayout.DescDwords; i++) w.Raw(BitConverter.ToUInt32(desc, 4 * i).ToString(CultureInfo.InvariantCulture)); w.End(']');
            w.End('}');
            w.End('}');
        }

        public int TextureId(Texture t) { return Mem.I32(Memory, Ptr(t) + UnityPlayerLayout.TextureId); }

        public void WriteTexture(JsonOut w, Texture t)
        {
            ulong p = Ptr(t);
            w.Begin('{');
            w.Key("texture_id_0x50"); w.Int(Mem.I32(Memory, p + UnityPlayerLayout.TextureId));
            if (t is RenderTexture)
            {
                ulong entry = Mem.U64(Memory, p + UnityPlayerLayout.RenderTextureTempEntry);
                w.Key("temp_entry_0x138"); w.Str("0x" + entry.ToString("x"));
                if (entry != 0)
                {
                    w.Key("temp_entry_instance_id"); w.Int(Mem.I32(Memory, entry + UnityPlayerLayout.EntryInstanceId));
                    w.Key("temp_entry_last_frame"); w.Int(Mem.I32(Memory, entry + UnityPlayerLayout.EntryLastFrame));
                }
                var desc = Mem.Bytes(Memory, p + UnityPlayerLayout.RenderTextureDesc, 4 * UnityPlayerLayout.DescDwords);
                w.Key("desc_key_0xc8"); w.Begin('[');
                for (int i = 0; i < UnityPlayerLayout.DescDwords; i++) w.Raw(BitConverter.ToUInt32(desc, 4 * i).ToString(CultureInfo.InvariantCulture));
                w.End(']');
            }
            w.End('}');
        }

        public void WriteSettings(JsonOut w)
        {
            ulong caps = Caps();
            int tier = Mem.I32(Memory, caps + UnityPlayerLayout.CapsActiveTier);
            ulong build = Manager(UnityPlayerLayout.ManagerBuildSettings);
            ulong graphics = Manager(UnityPlayerLayout.ManagerGraphicsSettings);
            w.Begin('{');
            w.Key("build_settings_0x101"); w.Int(Mem.U8(Memory, build + UnityPlayerLayout.BuildFlag101));
            w.Key("build_settings_0x102"); w.Int(Mem.U8(Memory, build + UnityPlayerLayout.BuildFlag102));
            w.Key("graphics_caps_active_tier_0xd4"); w.Int(tier);
            w.Key("graphics_caps_video_memory_mb_0xd8_f32"); w.Num(BitConverter.ToSingle(Mem.Bytes(Memory, caps + UnityPlayerLayout.CapsVideoMemoryMB, 4), 0));
            w.Key("graphics_caps_max_render_texture_size_0xf8"); w.Int(Mem.I32(Memory, caps + UnityPlayerLayout.CapsMaxRenderTextureSize));
            w.Key("graphics_settings_tier_cascaded_shadow_maps"); w.Int(Mem.U8(Memory, graphics + (ulong)(UnityPlayerLayout.GraphicsSettingsTierStride * tier) + UnityPlayerLayout.GraphicsSettingsTierCascaded));
            uint d0 = Mem.U32(Memory, caps + UnityPlayerLayout.CapsD0);
            w.Key("graphics_caps_0xd0_u32"); w.Str("0x" + d0.ToString("x8"));
            w.Key("graphics_caps_0xd0_bit7"); w.Bool((d0 & 0x80) != 0);
            w.Key("graphics_caps_0xdd_u8"); w.Int(Mem.U8(Memory, caps + UnityPlayerLayout.CapsDD));
            w.Key("graphics_caps_0x3a1_u8"); w.Int(Mem.U8(Memory, caps + UnityPlayerLayout.Caps3A1));
            var c3fc = Mem.Bytes(Memory, caps + UnityPlayerLayout.Caps3FC, 4);
            w.Key("graphics_caps_0x3fc_u32"); w.Raw(BitConverter.ToUInt32(c3fc, 0).ToString(CultureInfo.InvariantCulture));
            w.Key("graphics_caps_0x3fc_f32"); w.Num(BitConverter.ToSingle(c3fc, 0));
            var c404 = Mem.Bytes(Memory, caps + UnityPlayerLayout.Caps404, 4);
            w.Key("graphics_caps_0x404_u32"); w.Raw(BitConverter.ToUInt32(c404, 0).ToString(CultureInfo.InvariantCulture));
            w.Key("graphics_caps_0x404_f32"); w.Num(BitConverter.ToSingle(c404, 0));
            w.Key("build_settings_0xfa"); w.Int(Mem.U8(Memory, build + UnityPlayerLayout.BuildFlagFA));
            w.Key("gfxdevice_single_pass_stereo_0x21e8"); w.Int(Mem.I32(Memory, GfxDevice() + UnityPlayerLayout.GfxSinglePassStereo));
            w.End('}');
        }

        public void WriteIdentity(JsonOut w)
        {
            w.Begin('{');
            w.Key("module"); w.Str(ModulePath); w.Key("sha256"); w.Str(ModuleSha256);
            w.Key("base"); w.Str("0x" + Base.ToString("x"));
            w.Key("layout_valid"); w.Bool(Valid);
            w.Key("validation"); w.Begin('['); foreach (var v in Validation) w.Str(v); w.End(']');
            w.Key("property_name_count"); w.Int(PropertyNames.Length);
            w.Key("keyword_count"); w.Int(KeywordNames.Count);
            w.End('}');
        }
    }
}
