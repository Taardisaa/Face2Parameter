// Unity-free core of the v5 frame-pair probe: request validation, frame plan,
// streaming JSON writer and read-only parsers of UnityPlayer 2018.4.11f1 native
// structures. Everything here is pure (memory is read through IMemory), so it is
// unit-tested on the CPU without the game (tests/ProbeCoreTests).
//
// Native layouts were recovered from the installed UnityPlayer.dll
// (SHA-256 5034cdc2…13923, PDB GUID BFA91A8C44734BB4AABC5A975B304CFE age 1)
// with tools/offline_renderer/unityplayer_disasm.py; each constant names the
// routine and instruction it comes from. They are valid only for that binary,
// which the probe verifies before any native read.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace HS2Codex.FramePairProbe
{
    internal static class UnityPlayerLayout
    {
        public const string Sha256 = "5034cdc2bff71dc8d9e038cb71059f8965d0d879f8098af0f3c726ea06e13923";

        // ?GetRenderBufferManager@@ 0x755ca0: mov rax,[rip+0xdf21c9] (next 0x755ca7).
        // RenderTexture_CUSTOM_GetTemporary_Internal 0x93bea0 passes it unchanged as
        // RenderBufferManager::Textures* (Textures is at offset 0).
        public const ulong RenderBufferManagerPtr = 0x1547E70;
        // RenderBufferManager::Textures::GetTempBuffer(RenderTextureDesc) 0x756540,
        // ReleaseTempBuffer 0x75a610, GarbageCollect 0x754440.
        public const int TexBuckets = 0x00;      // bucket array (0x40-byte buckets)
        public const int TexMask = 0x08;         // byte mask: bucket = (hash & mask)
        public const int TexCount = 0x0C;        // occupied keys (dec when a key empties)
        public const int TexInUseList = 0x20;    // List<TextureEntry> root (push_front on acquire)
        public const int TexNameCounter = 0x34;  // "TempBuffer %d %ix%i" counter
        public const int TexFrame = 0x38;        // GarbageCollect frame counter
        public const int BucketStride = 0x40;
        public const int BucketHash = 0x00;      // 0xFFFFFFFF empty, 0xFFFFFFFE deleted
        public const int BucketKey = 0x08;       // RenderTextureDesc, 0x2c bytes compared (11 dwords)
        public const int BucketHead = 0x38;      // TextureEntry* free-list head (LIFO)
        public const int DescDwords = 11;
        // TextureEntry is a Unity ListNode (+0 m_Prev, +8 m_Next) plus +0x10 last-used
        // frame and +0x14 PPtr<RenderTexture> instance id (GetTempBuffer 0x7568c1/0x756905).
        public const int EntryPrev = 0x00, EntryNext = 0x08, EntryLastFrame = 0x10, EntryInstanceId = 0x14;
        // GetTempBuffer takes head and sets the new head to head->m_Next; Release inserts
        // the released entry before head and makes it the head (LIFO), so acquisition
        // order is head, head->m_Next, ... The in-use List root's m_Next is its front.
        public const int RenderTextureTempEntry = 0x138;  // ReleaseTempBuffer: mov rbx,[rdx+0x138]
        public const int RenderTextureDesc = 0xC8;        // ReleaseTempBuffer: lea r8,[rdx+0xc8]
        public const int TextureId = 0x50;                // Texture::GetNativeTexturePtr 0x755c30: [rbx+0x50]

        // Camera::UpdatePreviousViewProjectionMatrix 0x66e570 (mono path) copies +0x5d8 -> +0x598;
        // SetupMotionMatrices 0x6d0249 reads +0x598 as _PreviousVP; SetCameraShaderProps
        // tests [camera+0x4ec]==1 (implicit projection).
        public const int CameraImplicitProjection = 0x4EC;
        public const int CameraPreviousViewProjection = 0x598;
        public const int CameraViewProjection = 0x5D8;

        // Renderer_Get_Custom_PropIsVisible 0x970780: ([r+0x128] & 0x1000) && [r+0x160] != -1.
        public const int RendererFlags = 0x128;
        public const int RendererSceneHandle = 0x160;
        public const uint RendererVisibleBit = 0x1000;

        // Material::GetFloat 0x84a4f0 / HasProperty 0x84ba10: holder=[m+0xb0];
        // built when ([holder+0xd8]&1) && [holder+0x10]!=0; sheet = holder+0x18.
        public const int MaterialPropertiesHolder = 0xB0;
        public const int HolderBuiltFlags = 0xD8;
        public const int HolderBuiltPtr = 0x10;
        public const int HolderSheet = 0x18;

        // ShaderPropertySheet: FindPropertyData 0x710d20 (type ranges at +0x10+4*type),
        // GetPropertyData 0x711690 (desc: bits 0-19 value offset, 20-29 array size),
        // GetFloatValueInsRGBSpace 0x83d3f0 (bit 30: gamma flag -> LinearToGammaSpace on read),
        // HasProperty 0x83db50 (names +0x28, count +0x38), values +0x68.
        public const int SheetRanges = 0x10;   // int[6]: float, vector, matrix, texture, buffer starts + end
        public const int SheetNames = 0x28;
        public const int SheetCount = 0x38;
        public const int SheetDescs = 0x48;
        public const int SheetValues = 0x68;
        public static readonly string[] SheetTypes = { "float", "vector", "matrix", "texture", "buffer" };

        // ShaderScripting::GetGlobalFloat 0x711050: mov r8,[rip+0xe42581] (next 0x711057) =
        // g_SharedPassContext; global sheet at ctx+0x28. IsKeywordEnabled 0x712040: keyword
        // bits at ctx+8 (4 x u64).
        public const ulong SharedPassContextPtr = 0x15535D8;
        public const int PassContextKeywords = 0x08;
        public const int PassContextSheet = 0x28;

        // keywords::GetKeywordName 0x82c990: mov rax,[rip+0xd26b55] (next 0x82c9ab) -> std::map*;
        // node: +0 left, +8 parent, +0x10 right, +0x19 isnil, +0x20 const char* name, +0x28 int index.
        public const ulong KeywordMapPtr = 0x1553500;
        // FastPropertyName::GetName 0x856950: mov rdx,[rip+0xcfce77] (next 0x8569a1) -> table*;
        // table+0 char** names, table+0x10 int count.
        public const ulong PropertyNameTablePtr = 0x1553818;

        // Engine built-in shader params (not in the global sheet): ShaderScripting::GetGlobalVector
        // 0x711230 / GetGlobalMatrix 0x711100 read ids with kind bits 0x40000000 / 0x80000000 from
        // GetGfxDevice()+0x11e8+16*i / +0x1978+64*i. GetGfxDevice 0x8cd260 = TlsGetValue([rip+0xc89bfa],
        // next 0x8cd266). Names: GetBuiltinVectorParamName 0x8a0e90 (lea [rip+0xc17b21], next 0x8a0e9f,
        // 0x79 entries), GetBuiltinMatrixParamName 0x8a0e30 (lea [rip+0xc17f51], next 0x8a0e3f, 0x15
        // entries), GetBuiltinTexEnvParamName 0x8a0e50 (lea [rip+0xc17fe1], next 0x8a0e5f, 0x19 entries)
        // (FastPropertyName::GetName 0x856950 dispatches 0x80000000 -> 0x8a0e30, 0xC0000000 -> 0x8a0e50);
        // vector array sizes: s_NameLookups ([rip+0xcb5d7e], next 0x8a0e7a)+0x60, u16.
        // Offline check of the file tables: vectors start unity_LightColor0.., _Time = 75,
        // matrices glstate_matrix_projection..unity_MatrixPreviousMI, texenvs white, black, ...
        public const ulong GfxDeviceTlsIndex = 0x1556E60;
        public const int GfxBuiltinVectors = 0x11E8, GfxBuiltinMatrices = 0x1978;
        public const ulong BuiltinVectorNames = 0x14B89C0, BuiltinMatrixNames = 0x14B8D90, BuiltinTexEnvNames = 0x14B8E40;
        public const int BuiltinVectorCount = 0x79, BuiltinMatrixCount = 0x15, BuiltinTexEnvCount = 0x19;
        public const ulong ParamNameLookupsPtr = 0x1556BF8;
        public const int ParamNameLookupsVectorArraySizes = 0x60;

        // GetManagerFromContext 0x62b880: lea rcx,[rip+0xee11c6] (next 0x62b88a) -> Object*[].
        public const ulong ManagerContext = 0x150CA50;
        public const int ManagerGraphicsSettings = 6;   // GetGraphicsSettings 0x67e1f0
        public const int ManagerBuildSettings = 10;     // GetBuildSettings 0x7fb290
        public const int ManagerQualitySettings = 11;   // GetQualitySettings 0x706280
        public const int BuildFlag101 = 0x101, BuildFlag102 = 0x102;   // PrecalcLightColor 0x69f4a0
        // GetGraphicsCaps 0x82c850: mov rax,[rip+0xc8a3a9] (next 0x82c857).
        public const ulong GraphicsCapsPtr = 0x14B6C00;
        public const int CapsActiveTier = 0xD4, CapsVideoMemoryMB = 0xD8 /* float */, CapsMaxRenderTextureSize = 0xF8;
        public const int GraphicsSettingsTierStride = 16, GraphicsSettingsTierCascaded = 0xB4;  // 0x69f48d

        // ---- v6 (requests of docs/hs2_offline_frame_orchestrator.md §8.6/§9.5) ----
        // GraphicsCaps fields used by the forward rules: +0xd0 bit 7 and +0x3fc (RenderShadowMaps
        // 0x6a29db..0x6a2a60 normal-bias/PCF_SOFT branch), +0x3a1 (Camera::ClearNoSkybox 0x65f8e0
        // clear flags 6 + byte), +0x404 (skybox projection eps = value * 1e-6, unity_forward.py),
        // +0xdd (unused-cascade m[14], PrepareShadowMaps 0x69fecf). Read raw; meaning per caller.
        public const int CapsD0 = 0xD0, CapsDD = 0xDD, Caps3A1 = 0x3A1, Caps3FC = 0x3FC, Caps404 = 0x404;
        // BuildSettings+0xfa: _MotionVectorDepthBias -0.001 branch (forward_values.py, 0x6cc4e7 region).
        public const int BuildFlagFA = 0xFA;
        // Camera+0x558: current render target RenderTexture* (DoMotionVectorRenderLoop 0x6c9b60:
        // pass r12+1, r12 = 1 when null or RenderTexture+0xd0 (antiAliasing) > 1).
        public const int CameraCurrentTarget = 0x558, RenderTextureAntiAliasing = 0xD0;
        // GfxDevice+0x21e8: single-pass stereo eye count (SetupMotionMatrices 0x6d011d: 0 = mono).
        public const int GfxSinglePassStereo = 0x21E8;
        // Renderer::AddAsRenderNode 0x739400: BaseRenderer subobject at Renderer+0x38
        // (rcx = r15, Renderer* = r15-0x38). BaseRenderer::FlattenBasicData 0x650b50 copies
        // BaseRenderer+0x08..+0x88 -> RenderNode+0x00..+0x80 (+0x08 localToWorld, +0x48 previous
        // localToWorld = _PreviousM) and BaseRenderer+0x88..+0xfc -> RenderNode+0x80..+0xf4
        // (+0xbc transform update frame -> node+0xb4, +0xf0 flags -> node+0xe8 = Renderer+0x128).
        public const int RendererBase = 0x38, BaseL2W = 0x08, BasePrevL2W = 0x48, BaseUpdateFrame = 0xBC, BaseFlags = 0xF0;
        public const int BaseBlockStart = 0x08, BaseBlockLength = 0xF4;
        // SkinnedMeshRenderer::AddAsRenderNode 0x77abe0 is entered with this = the BaseRenderer
        // subobject (rbx = Renderer+0x38): `lea rsi,[rbx-0x38]` (0x77ac43) is the SkinnedMeshRenderer*
        // passed to GetActiveBlendShapeCount/SkinOnGPU/PrepareSkin, and rcx = rbx goes to
        // Renderer::AddAsRenderNode (0x77ad8f). Its operands are therefore BASE-relative
        // (v6.0.0 wrongly added them to Renderer*; probe 6.1.0 reads Renderer+0x38+offset):
        //   [rbx+0x1a0] cached Mesh* (== [rsi+0x1d8] at 0x77ac8a; Mesh::CreateMesh test at 0x77ac0e),
        //   [rbx+0x28c] re-skin (dirty) byte, cleared at 0x77ad53 after PrepareSkin/SkinOnGPU,
        //   [rbx+0x256] forceMatrixRecalculationPerRender test,
        //   [rbx+0x258] / [rbx+0x260] the two GfxBuffer* passed to MeshRenderingData::Init(mesh, mesh,
        //   [+0x258], [+0x260]) (skinned VB and previous-position VB),
        //   [rbx+0x255] _HasLastPositionData (0x77ae22 -> RenderNode+0xe8 bit 9).
        // Renderer-relative: mesh +0x1d8, +0x28d, +0x28e, +0x290, +0x298, +0x2c4.
        public const int SmrBaseMesh = 0x1A0, SmrBaseHasLastPosition = 0x255, SmrBaseForceMatrixRecalc = 0x256,
                         SmrBaseSkinBuffer = 0x258, SmrBasePrevSkinBuffer = 0x260, SmrBaseDirty = 0x28C;
        // Renderer* -> address of a BaseRenderer-relative SkinnedMeshRenderer field.
        public static ulong SmrField(ulong renderer, int baseOffset) { return renderer + (ulong)RendererBase + (ulong)baseOffset; }
        // Material+0xb0 = SharedMaterialData* (GetWritableSharedMaterialData; Material::UpdateHashes
        // 0x84e3b0 writes +0xd0 = Pass::ComputePassValuesHash of the shadow-caster pass (caster sort
        // key bits 40..61), +0xd4 = ComputeStateBlockValuesHash; +0xd8 flags (bit 0 built, bit 1
        // instancing, bit 2 hashes dirty), +0xdc material instance id (material sort key).
        public const int SmdCasterPassHash = 0xD0, SmdStateBlockHash = 0xD4, SmdFlags = 0xD8, SmdInstanceId = 0xDC;
    }

    internal interface IMemory
    {
        // Must never fault: returns false when any byte is unreadable.
        bool Read(ulong address, byte[] buffer, int count);
    }

    internal sealed class NativeReadError : Exception
    {
        public NativeReadError(string message) : base(message) { }
    }

    internal static class Mem
    {
        public static byte[] Bytes(IMemory m, ulong a, int n)
        {
            if (n < 0 || n > (1 << 26)) throw new NativeReadError("Unreasonable read size " + n);
            var b = new byte[n];
            if (n > 0 && (a == 0 || !m.Read(a, b, n))) throw new NativeReadError("Unreadable 0x" + a.ToString("x") + "+" + n);
            return b;
        }
        public static ulong U64(IMemory m, ulong a) { return BitConverter.ToUInt64(Bytes(m, a, 8), 0); }
        public static uint U32(IMemory m, ulong a) { return BitConverter.ToUInt32(Bytes(m, a, 4), 0); }
        public static int I32(IMemory m, ulong a) { return BitConverter.ToInt32(Bytes(m, a, 4), 0); }
        public static byte U8(IMemory m, ulong a) { return Bytes(m, a, 1)[0]; }
        public static string CString(IMemory m, ulong a, int max)
        {
            if (a == 0) throw new NativeReadError("Null string pointer");
            var sb = new StringBuilder();
            var chunk = new byte[64];
            var one = new byte[1];
            ulong at = a;
            while (sb.Length < max)
            {
                // A 64-byte read stays inside one page unless it crosses a page boundary;
                // then the remainder of the page is read and the next page separately.
                int n = (int)Math.Min(64UL, 4096UL - (at & 4095UL));
                if (!m.Read(at, chunk, n))
                {
                    if (!m.Read(at, one, 1)) throw new NativeReadError("Unreadable string byte at 0x" + at.ToString("x"));
                    if (one[0] == 0) return sb.ToString();
                    sb.Append((char)one[0]); at++; continue;
                }
                for (int i = 0; i < n; i++)
                {
                    if (chunk[i] == 0) return sb.ToString();
                    sb.Append((char)chunk[i]);
                }
                at += (ulong)n;
            }
            throw new NativeReadError("Unterminated string at 0x" + a.ToString("x"));
        }
        public static string Hex(byte[] b) { var sb = new StringBuilder(b.Length * 2); foreach (var x in b) sb.Append(x.ToString("x2")); return sb.ToString(); }
    }

    // ---------------------------------------------------------------- render buffer pool
    internal sealed class PoolEntry
    {
        public ulong Address, Next, Prev;
        public int LastFrame, InstanceId;
    }

    internal sealed class PoolBucket
    {
        public int Index;
        public uint Hash;
        public uint[] Key;
        public List<PoolEntry> FreeInAcquisitionOrder = new List<PoolEntry>();
    }

    internal sealed class PoolSnapshot
    {
        public ulong TexturesAddress, Buckets;
        public uint Mask, Count;
        public int NameCounter, Frame;
        public List<PoolBucket> Keys = new List<PoolBucket>();
        public List<PoolEntry> InUse = new List<PoolEntry>();
    }

    internal static class PoolWalker
    {
        public const int MaxBuckets = 1 << 16, MaxEntries = 1 << 14;

        public static PoolSnapshot Read(IMemory m, ulong textures)
        {
            var s = new PoolSnapshot { TexturesAddress = textures };
            s.Buckets = Mem.U64(m, textures + UnityPlayerLayout.TexBuckets);
            s.Mask = Mem.U32(m, textures + UnityPlayerLayout.TexMask);
            s.Count = Mem.U32(m, textures + UnityPlayerLayout.TexCount);
            s.NameCounter = Mem.I32(m, textures + UnityPlayerLayout.TexNameCounter);
            s.Frame = Mem.I32(m, textures + UnityPlayerLayout.TexFrame);
            if (s.Mask % UnityPlayerLayout.BucketStride != 0) throw new NativeReadError("Pool mask is not a bucket multiple: " + s.Mask);
            long n = s.Mask / UnityPlayerLayout.BucketStride + 1;
            if (n > MaxBuckets) throw new NativeReadError("Pool bucket count implausible: " + n);
            var all = Mem.Bytes(m, s.Buckets, (int)n * UnityPlayerLayout.BucketStride);
            for (int i = 0; i < n; i++)
            {
                int o = i * UnityPlayerLayout.BucketStride;
                uint hash = BitConverter.ToUInt32(all, o + UnityPlayerLayout.BucketHash);
                if (hash >= 0xFFFFFFFEu) continue;
                var key = new uint[UnityPlayerLayout.DescDwords];
                for (int k = 0; k < key.Length; k++) key[k] = BitConverter.ToUInt32(all, o + UnityPlayerLayout.BucketKey + 4 * k);
                var bucket = new PoolBucket { Index = i, Hash = hash, Key = key };
                ulong head = BitConverter.ToUInt64(all, o + UnityPlayerLayout.BucketHead);
                ulong cur = head;
                int guard = 0;
                do
                {
                    if (cur == 0) throw new NativeReadError("Null entry in free list of bucket " + i);
                    if (++guard > MaxEntries) throw new NativeReadError("Free list of bucket " + i + " does not close");
                    bucket.FreeInAcquisitionOrder.Add(Entry(m, cur));
                    cur = Mem.U64(m, cur + UnityPlayerLayout.EntryNext);
                } while (cur != head);
                s.Keys.Add(bucket);
            }
            if (s.Keys.Count != s.Count) throw new NativeReadError("Occupied buckets " + s.Keys.Count + " differ from count " + s.Count);
            ulong root = textures + UnityPlayerLayout.TexInUseList;
            ulong node = Mem.U64(m, root + UnityPlayerLayout.EntryNext);
            int g = 0;
            while (node != root)
            {
                if (node == 0) throw new NativeReadError("Null node in in-use list");
                if (++g > MaxEntries) throw new NativeReadError("In-use list does not close");
                s.InUse.Add(Entry(m, node));
                node = Mem.U64(m, node + UnityPlayerLayout.EntryNext);
            }
            return s;
        }

        static PoolEntry Entry(IMemory m, ulong a)
        {
            var b = Mem.Bytes(m, a, 0x18);
            return new PoolEntry
            {
                Address = a, Next = BitConverter.ToUInt64(b, UnityPlayerLayout.EntryNext), Prev = BitConverter.ToUInt64(b, UnityPlayerLayout.EntryPrev),
                LastFrame = BitConverter.ToInt32(b, UnityPlayerLayout.EntryLastFrame), InstanceId = BitConverter.ToInt32(b, UnityPlayerLayout.EntryInstanceId)
            };
        }

        public static void Write(JsonOut w, PoolSnapshot s)
        {
            w.Begin('{');
            w.Key("textures_address"); w.Str("0x" + s.TexturesAddress.ToString("x"));
            w.Key("bucket_mask"); w.Raw(s.Mask.ToString(CultureInfo.InvariantCulture));
            w.Key("key_count"); w.Raw(s.Count.ToString(CultureInfo.InvariantCulture));
            w.Key("temp_buffer_name_counter"); w.Raw(s.NameCounter.ToString(CultureInfo.InvariantCulture));
            w.Key("gc_frame"); w.Raw(s.Frame.ToString(CultureInfo.InvariantCulture));
            w.Key("key_layout"); w.Str("RenderTextureDesc dwords: width,height,antiAliasing,volumeDepth,colorFormat,depthBufferFormat,dimension,shadowSamplingMode,vrUsage,flags,memoryless");
            w.Key("free_lists"); w.Begin('[');
            foreach (var b in s.Keys)
            {
                w.Begin('{');
                w.Key("bucket"); w.Raw(b.Index.ToString(CultureInfo.InvariantCulture));
                w.Key("hash"); w.Str("0x" + b.Hash.ToString("x8"));
                w.Key("key"); w.Begin('['); foreach (var k in b.Key) w.Raw(k.ToString(CultureInfo.InvariantCulture)); w.End(']');
                w.Key("free_in_acquisition_order"); Entries(w, b.FreeInAcquisitionOrder);
                w.End('}');
            }
            w.End(']');
            w.Key("in_use_front_to_back"); Entries(w, s.InUse);
            w.End('}');
        }

        static void Entries(JsonOut w, List<PoolEntry> list)
        {
            w.Begin('[');
            foreach (var e in list)
            {
                w.Begin('{'); w.Key("instance_id"); w.Raw(e.InstanceId.ToString(CultureInfo.InvariantCulture));
                w.Key("last_frame"); w.Raw(e.LastFrame.ToString(CultureInfo.InvariantCulture));
                w.Key("entry"); w.Str("0x" + e.Address.ToString("x")); w.End('}');
            }
            w.End(']');
        }
    }

    // ---------------------------------------------------------------- shader property sheet
    internal sealed class SheetEntry
    {
        public int Index, NameId, ArraySize;
        public string Type;
        public uint Desc;
        public int Offset;
        public bool GammaFlag;
        public byte[] Value;
    }

    internal sealed class SheetSnapshot
    {
        public ulong Address;
        public int[] Ranges = new int[6];
        public long Count;
        public List<SheetEntry> Entries = new List<SheetEntry>();
        public string Hash;
    }

    internal static class SheetReader
    {
        public const int MaxProperties = 1 << 15;

        public static SheetSnapshot Read(IMemory m, ulong sheet)
        {
            var s = new SheetSnapshot { Address = sheet };
            var rb = Mem.Bytes(m, sheet + UnityPlayerLayout.SheetRanges, 24);
            for (int i = 0; i < 6; i++) s.Ranges[i] = BitConverter.ToInt32(rb, 4 * i);
            s.Count = (long)Mem.U64(m, sheet + UnityPlayerLayout.SheetCount);
            if (s.Count < 0 || s.Count > MaxProperties) throw new NativeReadError("Sheet count implausible: " + s.Count);
            if (s.Ranges[0] != 0) throw new NativeReadError("Sheet float range does not start at 0: " + s.Ranges[0]);
            for (int i = 1; i < 6; i++)
                if (s.Ranges[i] < s.Ranges[i - 1]) throw new NativeReadError("Sheet type ranges not monotonic");
            if (s.Ranges[5] != s.Count) throw new NativeReadError("Sheet range end " + s.Ranges[5] + " differs from count " + s.Count);
            if (s.Count == 0) { s.Hash = HashOf(s); return s; }
            ulong names = Mem.U64(m, sheet + UnityPlayerLayout.SheetNames);
            ulong descs = Mem.U64(m, sheet + UnityPlayerLayout.SheetDescs);
            ulong values = Mem.U64(m, sheet + UnityPlayerLayout.SheetValues);
            int n = (int)s.Count;
            var nb = Mem.Bytes(m, names, 4 * n);
            var db = Mem.Bytes(m, descs, 4 * n);
            var offsets = new List<int>();
            for (int i = 0; i < n; i++)
            {
                uint d = BitConverter.ToUInt32(db, 4 * i);
                int type = 0;
                while (type < 5 && i >= s.Ranges[type + 1]) type++;
                var e = new SheetEntry
                {
                    Index = i, NameId = BitConverter.ToInt32(nb, 4 * i), Desc = d, Type = UnityPlayerLayout.SheetTypes[type],
                    Offset = (int)(d & 0xFFFFF), ArraySize = (int)((d >> 20) & 0x3FF), GammaFlag = ((d >> 30) & 1) != 0
                };
                s.Entries.Add(e);
                offsets.Add(e.Offset);
            }
            offsets.Sort();
            var sizes = new int[n];
            int extent = 0;
            foreach (var e in s.Entries)
            {
                int elements = Math.Max(1, e.ArraySize);
                int size;
                switch (e.Type)
                {
                    case "float": size = 4 * elements; break;
                    case "vector": size = 16 * elements; break;
                    case "matrix": size = 64 * elements; break;
                    default:
                        // texture / buffer: the value region runs to the next property's offset
                        // (layout beyond the first dword, the TextureID/ComputeBufferID, is not
                        // interpreted); the last region is read as one dword.
                        int next = -1;
                        foreach (var o in offsets) if (o > e.Offset) { next = o; break; }
                        size = next < 0 ? 4 : Math.Min(next - e.Offset, 1024);
                        break;
                }
                sizes[e.Index] = size;
                extent = Math.Max(extent, e.Offset + size);
            }
            var vb = Mem.Bytes(m, values, extent);
            foreach (var e in s.Entries)
            {
                e.Value = new byte[sizes[e.Index]];
                Buffer.BlockCopy(vb, e.Offset, e.Value, 0, e.Value.Length);
            }
            s.Hash = HashOf(s);
            return s;
        }

        static string HashOf(SheetSnapshot s)
        {
            using (var sha = System.Security.Cryptography.SHA256.Create())
            {
                var ms = new System.IO.MemoryStream();
                foreach (var r in s.Ranges) ms.Write(BitConverter.GetBytes(r), 0, 4);
                foreach (var e in s.Entries)
                {
                    ms.Write(BitConverter.GetBytes(e.NameId), 0, 4);
                    ms.Write(BitConverter.GetBytes(e.Desc), 0, 4);
                    ms.Write(e.Value, 0, e.Value.Length);
                }
                return Mem.Hex(sha.ComputeHash(ms.ToArray()));
            }
        }

        public static void Write(JsonOut w, SheetSnapshot s, Func<int, string> name)
        {
            w.Begin('{');
            w.Key("sha256"); w.Str(s.Hash);
            w.Key("type_ranges"); w.Begin('['); foreach (var r in s.Ranges) w.Raw(r.ToString(CultureInfo.InvariantCulture)); w.End(']');
            w.Key("value_semantics"); w.Str("raw stored bytes as bound to GPU cbuffers; gamma_flag=1 means managed getters return LinearToGammaSpace(stored) in Linear space");
            w.Key("entries"); w.Begin('[');
            foreach (var e in s.Entries)
            {
                w.Begin('{');
                w.Key("name"); w.Str(name(e.NameId));
                w.Key("name_id"); w.Raw(e.NameId.ToString(CultureInfo.InvariantCulture));
                w.Key("type"); w.Str(e.Type);
                w.Key("array_size"); w.Raw(e.ArraySize.ToString(CultureInfo.InvariantCulture));
                w.Key("gamma_flag"); w.Raw(e.GammaFlag ? "true" : "false");
                w.Key("value_hex"); w.Str(Mem.Hex(e.Value));
                if (e.Type == "float" || e.Type == "vector" || e.Type == "matrix")
                {
                    w.Key("value_f32"); w.Begin('[');
                    for (int i = 0; i + 4 <= e.Value.Length; i += 4) w.Num(BitConverter.ToSingle(e.Value, i));
                    w.End(']');
                }
                else
                {
                    w.Key("first_dword"); w.Raw(BitConverter.ToUInt32(e.Value, 0).ToString(CultureInfo.InvariantCulture));
                }
                w.End('}');
            }
            w.End(']');
            w.End('}');
        }
    }

    // ---------------------------------------------------------------- name tables
    internal static class NameTables
    {
        public const int MaxNames = 1 << 20;

        // FastPropertyName table: table+0 char** names, table+0x10 int count.
        public static string[] PropertyNames(IMemory m, ulong tablePtrAddress)
        {
            ulong table = Mem.U64(m, tablePtrAddress);
            ulong names = Mem.U64(m, table);
            int count = Mem.I32(m, table + 0x10);
            if (count < 0 || count > MaxNames) throw new NativeReadError("Property name count implausible: " + count);
            var ptrs = Mem.Bytes(m, names, 8 * count);
            var result = new string[count];
            for (int i = 0; i < count; i++)
            {
                ulong p = BitConverter.ToUInt64(ptrs, 8 * i);
                result[i] = p == 0 ? null : Mem.CString(m, p, 4096);
            }
            return result;
        }

        // std::map<const char*, int> (MSVC tree): returns index -> name.
        public static SortedDictionary<int, string> Keywords(IMemory m, ulong mapPtrAddress)
        {
            ulong map = Mem.U64(m, mapPtrAddress);
            ulong head = Mem.U64(m, map);
            ulong root = Mem.U64(m, head + 8);
            var result = new SortedDictionary<int, string>();
            var stack = new Stack<ulong>();
            if (Mem.U8(m, root + 0x19) == 0) stack.Push(root);
            int guard = 0;
            while (stack.Count > 0)
            {
                if (++guard > 4096) throw new NativeReadError("Keyword map traversal exceeded 4096 nodes");
                ulong node = stack.Pop();
                ulong name = Mem.U64(m, node + 0x20);
                int index = Mem.I32(m, node + 0x28);
                if (result.ContainsKey(index)) throw new NativeReadError("Duplicate keyword index " + index);
                result[index] = Mem.CString(m, name, 1024);
                foreach (var child in new[] { Mem.U64(m, node), Mem.U64(m, node + 0x10) })
                    if (child != 0 && Mem.U8(m, child + 0x19) == 0) stack.Push(child);
            }
            return result;
        }

        // FastPropertyName ids with the top two bits set are built-in params
        // (FastPropertyName::GetName 0x856950); they are named by tables not read here.
        public static string Resolve(string[] names, int id)
        {
            uint kind = (uint)id & 0xC0000000u;
            if (kind == 0x40000000u) return "<builtin vector " + (id & 0x3FFFFFFF) + ">";
            if (kind == 0x80000000u) return "<builtin matrix " + (id & 0x3FFFFFFF) + ">";
            if (kind == 0xC0000000u) return "<builtin texenv " + (id & 0x3FFFFFFF) + ">";
            if (names != null && id >= 0 && id < names.Length && names[id] != null) return names[id];
            return "<unresolved id " + id + ">";
        }
    }

    // ---------------------------------------------------------------- request and plan
    internal sealed class ProbeRequest
    {
        public string Mode;                 // frame | renderstyle | quit
        public bool DriverHooks;
        public byte[] Nonce;
        public string NonceHex, OutDir, RawOutDir, CapturePrefix, ExpectedRenderDocSha256, ExpectedUnityPlayerSha256;
        public string ReadbackDll, ExpectedReadbackSha256, ShaderManifest, Meshes;
        public int Frames, TriggerIndex;
        public float PollSeconds, RawTimeoutSeconds, BridgeTimeoutSeconds;
        public long RawMinFreeBytes;
        public string BridgeUrl, BridgeBodyFile, ExpectedBridgeSha256, BridgeDispatchRequestFile;
        public Dictionary<string, string> Raw;

        public const int CaptureOffset = 2;   // first Present-boundary attempt fails (Uncapped Map/Unmap), retry = next frame
        public int RawIndex { get { return TriggerIndex + 1; } }
        public int PostIndex { get { return Mode == "renderstyle" ? TriggerIndex + 1 : -1; } }
        public int ExpectedCaptureIndex { get { return TriggerIndex + CaptureOffset; } }

        static readonly string[] Common = { "Mode", "Nonce", "OutDir", "RawOutDir", "Frames", "TriggerIndex", "PollSeconds", "CapturePrefix",
            "ExpectedRenderDocSha256", "ExpectedUnityPlayerSha256", "ReadbackDll", "ExpectedReadbackSha256", "RawTimeoutSeconds",
            "RawMinFreeGB", "ShaderManifest", "Meshes", "DriverHooks" };
        static readonly string[] Quit = { "Mode", "Nonce", "OutDir" };
        static readonly string[] Bridge = { "BridgeUrl", "BridgeBodyFile", "ExpectedBridgeSha256", "BridgeDispatchRequestFile", "BridgeTimeoutSeconds" };

        public static ProbeRequest Parse(Dictionary<string, string> d, Func<string, bool> pathExists)
        {
            var r = new ProbeRequest { Raw = d };
            r.Mode = Get(d, "Mode");
            if (r.Mode != "frame" && r.Mode != "renderstyle" && r.Mode != "quit") throw new ArgumentException("Mode must be frame, renderstyle or quit");
            if (r.Mode == "quit")
            {
                // Clean exit through the game's own path (ExitDialog.GameEnd(isCheck: false), the
                // dialog's Yes button). Only a nonce and a fresh receipt directory.
                foreach (var k in d.Keys) if (Array.IndexOf(Quit, k) < 0) throw new ArgumentException("Unknown quit request key " + k);
                foreach (var k in Quit) Get(d, k);
                r.NonceHex = Get(d, "Nonce").ToLowerInvariant();
                if (r.NonceHex.Length != 32) throw new ArgumentException("Nonce must be 32 hex digits");
                r.OutDir = Abs(d, "OutDir");
                if (pathExists(r.OutDir)) throw new ArgumentException("Fresh OutDir required: " + r.OutDir);
                return r;
            }
            var allowed = new HashSet<string>(Common);
            if (r.Mode == "renderstyle") foreach (var k in Bridge) allowed.Add(k);
            foreach (var k in d.Keys) if (!allowed.Contains(k)) throw new ArgumentException("Unknown request key " + k);
            foreach (var k in Common) Get(d, k);
            r.NonceHex = Get(d, "Nonce").ToLowerInvariant();
            if (r.NonceHex.Length != 32) throw new ArgumentException("Nonce must be 32 hex digits");
            r.Nonce = new byte[16];
            for (int i = 0; i < 16; i++) r.Nonce[i] = Convert.ToByte(r.NonceHex.Substring(2 * i, 2), 16);
            r.OutDir = Abs(d, "OutDir"); r.RawOutDir = Abs(d, "RawOutDir"); r.CapturePrefix = Abs(d, "CapturePrefix");
            r.ReadbackDll = Abs(d, "ReadbackDll"); r.ShaderManifest = Abs(d, "ShaderManifest");
            r.ExpectedRenderDocSha256 = Sha(d, "ExpectedRenderDocSha256");
            r.ExpectedUnityPlayerSha256 = Sha(d, "ExpectedUnityPlayerSha256");
            r.ExpectedReadbackSha256 = Sha(d, "ExpectedReadbackSha256");
            if (r.ExpectedUnityPlayerSha256 != UnityPlayerLayout.Sha256) throw new ArgumentException("Native layouts were recovered only for UnityPlayer " + UnityPlayerLayout.Sha256);
            r.Frames = Int(d, "Frames"); r.TriggerIndex = Int(d, "TriggerIndex");
            r.PollSeconds = Flt(d, "PollSeconds"); r.RawTimeoutSeconds = Flt(d, "RawTimeoutSeconds");
            r.RawMinFreeBytes = (long)(Flt(d, "RawMinFreeGB") * 1073741824.0);
            r.Meshes = Get(d, "Meshes");
            var hooks = Get(d, "DriverHooks");
            if (hooks != "0" && hooks != "1") throw new ArgumentException("DriverHooks must be 0 or 1");
            r.DriverHooks = hooks == "1";
            if (r.TriggerIndex < 1) throw new ArgumentException("TriggerIndex must be >= 1 (the frame before the trigger is exported)");
            int minFrames = r.ExpectedCaptureIndex + (r.Mode == "renderstyle" ? 2 : 1);
            if (r.Frames < minFrames || r.Frames > 64) throw new ArgumentException("Frames must be in [" + minFrames + ", 64] for this TriggerIndex");
            if (r.PollSeconds <= 0 || r.PollSeconds > 600 || r.RawTimeoutSeconds <= 0 || r.RawTimeoutSeconds > 1800 || r.RawMinFreeBytes < 0)
                throw new ArgumentException("Bad timeout or free-space bound");
            if (pathExists(r.OutDir)) throw new ArgumentException("Fresh OutDir required: " + r.OutDir);
            if (pathExists(r.RawOutDir)) throw new ArgumentException("Fresh RawOutDir required: " + r.RawOutDir);
            if (r.Mode == "renderstyle")
            {
                foreach (var k in Bridge) Get(d, k);
                r.BridgeUrl = Get(d, "BridgeUrl");
                if (r.BridgeUrl != "http://127.0.0.1:43127/maker/renderstyle/capture") throw new ArgumentException("BridgeUrl must be the local /maker/renderstyle/capture route");
                r.BridgeBodyFile = Abs(d, "BridgeBodyFile");
                r.ExpectedBridgeSha256 = Sha(d, "ExpectedBridgeSha256");
                r.BridgeDispatchRequestFile = Abs(d, "BridgeDispatchRequestFile");
                r.BridgeTimeoutSeconds = Flt(d, "BridgeTimeoutSeconds");
                if (r.BridgeTimeoutSeconds <= 0 || r.BridgeTimeoutSeconds > 600) throw new ArgumentException("Bad BridgeTimeoutSeconds");
            }
            return r;
        }

        static string Get(Dictionary<string, string> d, string k)
        {
            string v;
            if (!d.TryGetValue(k, out v) || v.Length == 0) throw new ArgumentException("Missing request key " + k);
            return v;
        }
        static string Abs(Dictionary<string, string> d, string k)
        {
            var v = Get(d, k);
            if (!System.IO.Path.IsPathRooted(v)) throw new ArgumentException(k + " must be absolute");
            return v;
        }
        static string Sha(Dictionary<string, string> d, string k)
        {
            var v = Get(d, k).ToLowerInvariant();
            if (v.Length != 64) throw new ArgumentException(k + " must be a SHA-256");
            foreach (var c in v) if (!Uri.IsHexDigit(c)) throw new ArgumentException(k + " must be hex");
            return v;
        }
        static int Int(Dictionary<string, string> d, string k) { return int.Parse(Get(d, k), NumberStyles.Integer, CultureInfo.InvariantCulture); }
        static float Flt(Dictionary<string, string> d, string k) { return float.Parse(Get(d, k), NumberStyles.Float, CultureInfo.InvariantCulture); }

        public static Dictionary<string, string> ParseIni(string text)
        {
            var d = new Dictionary<string, string>();
            foreach (var raw in text.Split('\n'))
            {
                var line = raw.Trim();
                if (line.Length == 0 || line.StartsWith("#")) continue;
                int i = line.IndexOf('=');
                if (i <= 0) throw new FormatException("Bad request line: " + line);
                var key = line.Substring(0, i).Trim();
                if (d.ContainsKey(key)) throw new FormatException("Duplicate request key " + key);
                d[key] = line.Substring(i + 1).Trim();
            }
            return d;
        }

        // Per window index (each is one end-of-frame point) the actions in execution order.
        public string[] ActionsAt(int k)
        {
            var a = new List<string> { "export_frame", "export_scene", "write_marker" };
            if (k == RawIndex) a.Add(Mode == "renderstyle" ? "raw_export_blocking" : "raw_export");
            if (k == PostIndex) a.Add("bridge_post_until_queued");
            if (k == TriggerIndex) a.Add("renderdoc_trigger");
            return a.ToArray();
        }
    }

    // ---------------------------------------------------------------- minimal JSON reader
    // For the bridge request body and its HTTP responses only (no JSON library is
    // loaded into the game by the probe). Objects -> Dictionary<string, object>
    // (duplicate keys rejected), arrays -> List<object>, numbers -> double.
    internal static class MiniJson
    {
        public static object Parse(string text)
        {
            int i = 0;
            var v = Value(text, ref i);
            Ws(text, ref i);
            if (i != text.Length) throw new FormatException("Trailing JSON text at " + i);
            return v;
        }

        static void Ws(string s, ref int i) { while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++; }

        static object Value(string s, ref int i)
        {
            Ws(s, ref i);
            if (i >= s.Length) throw new FormatException("Unexpected end of JSON");
            char c = s[i];
            if (c == '{')
            {
                i++; var d = new Dictionary<string, object>(StringComparer.Ordinal);
                Ws(s, ref i);
                if (i < s.Length && s[i] == '}') { i++; return d; }
                while (true)
                {
                    Ws(s, ref i);
                    if (i >= s.Length || s[i] != '"') throw new FormatException("Object key expected at " + i);
                    var k = Str(s, ref i);
                    Ws(s, ref i);
                    if (i >= s.Length || s[i] != ':') throw new FormatException("':' expected at " + i);
                    i++;
                    if (d.ContainsKey(k)) throw new FormatException("Duplicate JSON key " + k);
                    d[k] = Value(s, ref i);
                    Ws(s, ref i);
                    if (i < s.Length && s[i] == ',') { i++; continue; }
                    if (i < s.Length && s[i] == '}') { i++; return d; }
                    throw new FormatException("',' or '}' expected at " + i);
                }
            }
            if (c == '[')
            {
                i++; var l = new List<object>();
                Ws(s, ref i);
                if (i < s.Length && s[i] == ']') { i++; return l; }
                while (true)
                {
                    l.Add(Value(s, ref i));
                    Ws(s, ref i);
                    if (i < s.Length && s[i] == ',') { i++; continue; }
                    if (i < s.Length && s[i] == ']') { i++; return l; }
                    throw new FormatException("',' or ']' expected at " + i);
                }
            }
            if (c == '"') return Str(s, ref i);
            if (string.CompareOrdinal(s, i, "true", 0, 4) == 0) { i += 4; return true; }
            if (string.CompareOrdinal(s, i, "false", 0, 5) == 0) { i += 5; return false; }
            if (string.CompareOrdinal(s, i, "null", 0, 4) == 0) { i += 4; return null; }
            int start = i;
            while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
            if (start == i) throw new FormatException("JSON value expected at " + i);
            return double.Parse(s.Substring(start, i - start), NumberStyles.Float, CultureInfo.InvariantCulture);
        }

        static string Str(string s, ref int i)
        {
            i++;
            var sb = new StringBuilder();
            while (true)
            {
                if (i >= s.Length) throw new FormatException("Unterminated JSON string");
                char c = s[i++];
                if (c == '"') return sb.ToString();
                if (c != '\\') { sb.Append(c); continue; }
                if (i >= s.Length) throw new FormatException("Bad escape");
                char e = s[i++];
                switch (e)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        if (i + 4 > s.Length) throw new FormatException("Bad \\u escape");
                        sb.Append((char)Convert.ToInt32(s.Substring(i, 4), 16)); i += 4; break;
                    default: throw new FormatException("Bad escape \\" + e);
                }
            }
        }
    }

    // Validated /maker/renderstyle/capture body (installed bridge efa943cb…: keys
    // path, yaw, target, ortho_size, width, height, freeze_pose; geometry_out_path is
    // only in the uncommitted bridge source and is rejected by the installed DLL).
    internal sealed class BridgeBody
    {
        public string Text, PngPath;
        public int Width, Height;

        static readonly HashSet<string> Keys = new HashSet<string> { "path", "yaw", "target", "ortho_size", "width", "height", "freeze_pose" };

        public static BridgeBody Parse(string text, Func<string, bool> exists)
        {
            var d = MiniJson.Parse(text) as Dictionary<string, object>;
            if (d == null) throw new ArgumentException("Bridge body must be a JSON object");
            foreach (var k in d.Keys) if (!Keys.Contains(k)) throw new ArgumentException("Installed bridge rejects body key " + k);
            var path = d.ContainsKey("path") ? d["path"] as string : null;
            if (path == null || !System.IO.Path.IsPathRooted(path) || !path.EndsWith(".png", StringComparison.OrdinalIgnoreCase) || exists(path))
                throw new ArgumentException("Bridge body path must be a fresh absolute .png");
            if (!d.ContainsKey("width") || !d.ContainsKey("height") || !(d["width"] is double) || !(d["height"] is double))
                throw new ArgumentException("Bridge body must give width and height explicitly (they identify the capture RT)");
            if (!d.ContainsKey("target") || !d.ContainsKey("ortho_size")) throw new ArgumentException("Bridge body requires target and ortho_size");
            double w = (double)d["width"], h = (double)d["height"];
            if (w != Math.Floor(w) || h != Math.Floor(h) || w < 64 || h < 64 || w > 4096 || h > 4096) throw new ArgumentException("width/height out of the bridge's bounds");
            if (d.ContainsKey("freeze_pose") && !(d["freeze_pose"] is bool)) throw new ArgumentException("freeze_pose must be boolean");
            return new BridgeBody { Text = text, PngPath = path, Width = (int)w, Height = (int)h };
        }
    }

    // ---------------------------------------------------------------- skin double buffer
    // Labels which frame's skin each of a SkinnedMeshRenderer's two GfxBuffers holds.
    // Observations: (frame, point, renderer, slot0=[+0x258], slot1=[+0x260]). Rule (session
    // v5.0.1 mode b evidence, pairing of 5af92a08): the frame's own skinning writes the buffer
    // that is slot 0 at the end of that frame, so a buffer holds the skin of the last frame F
    // whose end-of-frame observation had it in slot 0 (and, for F < first observed frame, the
    // label is unknown). The raw observations are exported with the labels so the rule can be
    // re-checked against the RDC.
    internal sealed class SkinSlotTracker
    {
        readonly Dictionary<ulong, int> _lastEndOfFrameSlot0 = new Dictionary<ulong, int>();

        public void EndOfFrame(int frame, ulong slot0)
        {
            if (slot0 != 0) _lastEndOfFrameSlot0[slot0] = frame;
        }

        // -1 = not observed as slot 0 at any earlier end of frame in this window.
        public int HoldsFrame(ulong buffer)
        {
            int f;
            return buffer != 0 && _lastEndOfFrameSlot0.TryGetValue(buffer, out f) ? f : -1;
        }
    }

    // SkinnedMeshRenderer native fields (probe 6.1.0). The BaseRenderer-relative layout of
    // SkinnedMeshRenderer::AddAsRenderNode is used only when its cached Mesh*
    // (Renderer+0x38+0x1a0) is exactly the native object of the renderer's sharedMesh
    // (0 when sharedMesh is null); anything else fails closed.
    internal static class SmrNative
    {
        public static ulong MeshAddress(ulong renderer) { return UnityPlayerLayout.SmrField(renderer, UnityPlayerLayout.SmrBaseMesh); }

        public static void RequireMesh(IMemory m, ulong renderer, ulong sharedMesh, string name)
        {
            if (renderer == 0) throw new NativeReadError("SkinnedMeshRenderer " + name + " has no native object");
            ulong got = Mem.U64(m, MeshAddress(renderer));
            if (got != sharedMesh)
                throw new NativeReadError("SkinnedMeshRenderer " + name + ": Renderer+0x38+0x1a0 = 0x" + got.ToString("x")
                                          + " is not the sharedMesh native object 0x" + sharedMesh.ToString("x"));
        }

        public static ulong SkinBuffer(IMemory m, ulong renderer, ulong sharedMesh, string name)
        {
            RequireMesh(m, renderer, sharedMesh, name);
            return Mem.U64(m, UnityPlayerLayout.SmrField(renderer, UnityPlayerLayout.SmrBaseSkinBuffer));
        }
    }

    // Global order certificate shared by camera callbacks and driver hooks.
    internal static class ProbeSequence
    {
        static long _next;
        public static long Next() { return ++_next; }
    }

    // ---------------------------------------------------------------- JSON
    // Streaming JSON writer. Floats are written in round-trip form, verified by
    // parsing back; values that do not round-trip use G9 (always exact for float32),
    // non-finite values their IEEE bits.
    internal sealed class JsonOut
    {
        readonly StringBuilder _sb;
        readonly Stack<bool> _first = new Stack<bool>();
        bool _afterKey;

        public JsonOut() : this(1 << 16) { }
        public JsonOut(int capacity) { _sb = new StringBuilder(capacity); }

        // Writes key:value where value is produced by body into a scratch writer; on any
        // exception writes key_error:"Type: message" instead, so a failing section never
        // leaves partial JSON behind.
        // Array-element form of Guard: the value, or {"error":"Type: message"}.
        public void GuardValue(Action<JsonOut> body)
        {
            var sub = new JsonOut(1024);
            try { body(sub); }
            catch (Exception e)
            {
                var inner = e is System.Reflection.TargetInvocationException && e.InnerException != null ? e.InnerException : e;
                Begin('{'); Key("error"); Str(inner.GetType().Name + ": " + inner.Message); End('}');
                return;
            }
            Raw(sub.ToString());
        }

        public void Guard(string key, Action<JsonOut> body)
        {
            var sub = new JsonOut(1024);
            try { body(sub); }
            catch (Exception e)
            {
                var inner = e is System.Reflection.TargetInvocationException && e.InnerException != null ? e.InnerException : e;
                Key(key + "_error"); Str(inner.GetType().Name + ": " + inner.Message);
                return;
            }
            Key(key); Raw(sub.ToString());
        }

        public void Begin(char c) { Sep(); _sb.Append(c); _first.Push(true); }
        public void End(char c) { _sb.Append(c); _first.Pop(); }
        public void Next() { }
        public void Key(string k) { Sep(); _sb.Append(Quote(k)).Append(':'); _afterKey = true; }
        public void Raw(string s) { Sep(); _sb.Append(s); }
        public void Str(string s) { Sep(); _sb.Append(Quote(s)); }
        public void Bool(bool b) { Raw(b ? "true" : "false"); }
        public void Int(long v) { Raw(v.ToString(CultureInfo.InvariantCulture)); }
        public void Num(float f) { Sep(); _sb.Append(Float(f)); }
        public int Length { get { return _sb.Length; } }

        public static string Float(float f)
        {
            if (float.IsNaN(f) || float.IsInfinity(f))
                return "{\"nonfinite_bits\":\"0x" + BitConverter.ToUInt32(BitConverter.GetBytes(f), 0).ToString("x8") + "\"}";
            if (f == 0 && BitConverter.ToInt32(BitConverter.GetBytes(f), 0) != 0) return "-0.0";
            var s = f.ToString("R", CultureInfo.InvariantCulture);
            float back;
            if (!float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out back) ||
                BitConverter.ToInt32(BitConverter.GetBytes(back), 0) != BitConverter.ToInt32(BitConverter.GetBytes(f), 0))
                s = f.ToString("G9", CultureInfo.InvariantCulture);
            return s;
        }

        public static string Quote(string s)
        {
            if (s == null) return "null";
            var sb = new StringBuilder(s.Length + 2).Append('"');
            foreach (var c in s)
            {
                if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                else if (c < 0x20 || (c >= 0xD800 && c <= 0xDFFF)) sb.Append("\\u").Append(((int)c).ToString("x4"));
                else sb.Append(c);
            }
            return sb.Append('"').ToString();
        }

        void Sep()
        {
            if (_afterKey) { _afterKey = false; return; }
            if (_first.Count == 0) return;
            if (_first.Peek()) { _first.Pop(); _first.Push(false); }
            else _sb.Append(',');
        }

        public override string ToString() { return _sb.ToString(); }
    }
}
