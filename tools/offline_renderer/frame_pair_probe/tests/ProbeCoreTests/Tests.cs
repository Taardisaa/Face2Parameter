// CPU tests of the Unity-free probe core (ProbeCore.cs). No game, no Unity.
// Native structures are built in a fake address space following the layouts
// recovered from UnityPlayer 2018.4.11f1; the pool test drives the list through
// re-implementations of the disassembled ReleaseTempBuffer / GetTempBuffer
// list operations and checks that the walker reports the acquisition order that
// GetTempBuffer would produce.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using HS2Codex.FramePairProbe;

static class Program
{
    static int failures, passes;

    static void Check(bool ok, string what)
    {
        if (ok) passes++; else { failures++; Console.WriteLine("FAIL: " + what); }
    }

    static void Throws<T>(Action a, string what) where T : Exception
    {
        try { a(); failures++; Console.WriteLine("FAIL (no exception): " + what); }
        catch (T) { passes++; }
        catch (Exception e) { failures++; Console.WriteLine("FAIL (wrong exception " + e.GetType().Name + "): " + what); }
    }

    sealed class FakeMemory : IMemory
    {
        readonly SortedDictionary<ulong, byte[]> _regions = new SortedDictionary<ulong, byte[]>();
        public ulong Next = 0x10000;
        public ulong Alloc(int size) { ulong a = Next; _regions[a] = new byte[size]; Next += (ulong)((size + 0xFFF) & ~0xFFF) + 0x1000; return a; }
        byte[] Region(ulong a, out ulong off)
        {
            foreach (var kv in _regions)
                if (a >= kv.Key && a < kv.Key + (ulong)kv.Value.Length) { off = a - kv.Key; return kv.Value; }
            off = 0; return null;
        }
        public bool Read(ulong address, byte[] buffer, int count)
        {
            ulong off; var r = Region(address, out off);
            if (r == null || off + (ulong)count > (ulong)r.Length) return false;
            Buffer.BlockCopy(r, (int)off, buffer, 0, count);
            return true;
        }
        public void Write(ulong address, byte[] data) { ulong off; var r = Region(address, out off); Buffer.BlockCopy(data, 0, r, (int)off, data.Length); }
        public void U64(ulong a, ulong v) { Write(a, BitConverter.GetBytes(v)); }
        public void U32(ulong a, uint v) { Write(a, BitConverter.GetBytes(v)); }
        public void I32(ulong a, int v) { Write(a, BitConverter.GetBytes(v)); }
        public ulong RU64(ulong a) { var b = new byte[8]; Read(a, b, 8); return BitConverter.ToUInt64(b, 0); }
        public ulong Str(string s) { var a = Alloc(s.Length + 1); Write(a, Encoding.ASCII.GetBytes(s + "\0")); return a; }
    }

    // ---- disassembly-derived list operations (UnityPlayer 2018.4.11f1) ----
    // ReleaseTempBuffer 0x75a6e6..0x75a72a: existing key -> insert entry before head
    // (entry.prev = head.prev, entry.next = head, head.prev.next = entry, ... ) and head = entry.
    static void SimRelease(FakeMemory m, ulong bucket, ulong entry, bool newKey)
    {
        if (newKey) { m.U64(entry + 0, entry); m.U64(entry + 8, entry); m.U64(bucket + 0x38, entry); return; }
        ulong head = m.RU64(bucket + 0x38);
        m.U64(entry + 0, m.RU64(head + 0));      // [rbx] = [rdx]
        m.U64(entry + 8, head);                  // [rbx+8] = rdx
        m.U64(m.RU64(head + 0) + 8, entry);      // [rax+8] = rbx (rax = old head.prev)
        m.U64(m.RU64(entry + 8) + 0, entry);     // [[rbx+8]] = rbx -> head.prev = entry
        m.U64(bucket + 0x38, entry);             // [r8+0x38] = rbx
    }

    // GetTempBuffer 0x7568db..0x7568f5: take head; if head.next == head the key empties, else head = head.next.
    static ulong SimAcquire(FakeMemory m, ulong bucket)
    {
        ulong head = m.RU64(bucket + 0x38);
        ulong next = m.RU64(head + 8);
        if (next == head) m.U32(bucket, 0xFFFFFFFE); else m.U64(bucket + 0x38, next);
        // unlink as List::push_front would before inserting into the in-use list
        m.U64(m.RU64(head + 8) + 0, m.RU64(head + 0));
        m.U64(m.RU64(head + 0) + 8, m.RU64(head + 8));
        return head;
    }

    static ulong Entry(FakeMemory m, int instanceId, int frame)
    {
        var e = m.Alloc(0x18);
        m.I32(e + 0x10, frame); m.I32(e + 0x14, instanceId);
        return e;
    }

    static void TestPool()
    {
        var m = new FakeMemory();
        ulong tex = m.Alloc(0x40);
        ulong buckets = m.Alloc(4 * 0x40);
        m.U64(tex + 0, buckets); m.U32(tex + 8, 3 * 0x40); m.U32(tex + 0xC, 2);
        m.I32(tex + 0x34, 147); m.I32(tex + 0x38, 26060);
        ulong root = tex + 0x20; m.U64(root, root); m.U64(root + 8, root);
        m.U32(buckets + 0x00, 0xFFFFFFFF);
        ulong b1 = buckets + 0x40, b3 = buckets + 0xC0;
        m.U32(b1, 0x1234); uint[] key = { 3200, 1800, 1, 1, 2, 0, 2, 0, 0, 0x20, 0 };
        for (int i = 0; i < key.Length; i++) m.U32(b1 + 8 + (ulong)(4 * i), key[i]);
        m.U32(buckets + 0x80, 0xFFFFFFFE);
        m.U32(b3, 0x5678);
        var e1 = Entry(m, -101, 10); var e2 = Entry(m, -102, 11); var e3 = Entry(m, -103, 12); var e4 = Entry(m, -104, 13);
        SimRelease(m, b1, e1, true); SimRelease(m, b1, e2, false); SimRelease(m, b1, e3, false);   // LIFO: e3, e2, e1
        var e5 = Entry(m, -105, 14); SimRelease(m, b3, e5, true);
        // in-use list: one entry at the front (push_front 0x6294f0 onto an empty root)
        m.U64(e4 + 0, root); m.U64(e4 + 8, root); m.U64(root + 8, e4); m.U64(root + 0, e4);
        var s = PoolWalker.Read(m, tex);
        Check(s.Keys.Count == 2 && s.Count == 2, "pool key count");
        var order = s.Keys[0].FreeInAcquisitionOrder.Select(e => e.InstanceId).ToArray();
        Check(order.SequenceEqual(new[] { -103, -102, -101 }), "pool LIFO acquisition order " + string.Join(",", order));
        Check(s.Keys[0].Key.SequenceEqual(key), "pool desc key");
        Check(s.InUse.Count == 1 && s.InUse[0].InstanceId == -104, "pool in-use list");
        Check(s.NameCounter == 147 && s.Frame == 26060, "pool counters");
        // the walker order equals what GetTempBuffer would hand out
        var acquired = new List<int>();
        for (int i = 0; i < 3; i++) acquired.Add(BitConverter.ToInt32(BitConverter.GetBytes(m.RU64(SimAcquire(m, b1) + 0x10) >> 32), 0));
        Check(acquired.SequenceEqual(order), "walker order matches simulated GetTempBuffer " + string.Join(",", acquired));
        m.U32(tex + 0xC, 2);   // b1 is now deleted, count should be 1: mismatch must be reported
        Throws<NativeReadError>(() => PoolWalker.Read(m, tex), "pool count mismatch");
        m.U32(tex + 0xC, 1);
        var s2 = PoolWalker.Read(m, tex);
        Check(s2.Keys.Count == 1 && s2.Keys[0].FreeInAcquisitionOrder[0].InstanceId == -105, "pool after acquisitions");
        m.U64(e5 + 8, e4);  // break the ring of b3: e5 -> e4 -> root ... never returns to e5
        m.U64(e4 + 8, e4);
        Throws<NativeReadError>(() => PoolWalker.Read(m, tex), "non-closing free list");
        var w = new JsonOut(); PoolWalker.Write(w, s);
        Check(MiniJson.Parse(w.ToString()) is Dictionary<string, object>, "pool JSON parses");
    }

    static void TestSheet()
    {
        var m = new FakeMemory();
        ulong sheet = m.Alloc(0x80);
        int[] ranges = { 0, 2, 3, 3, 4, 4 };
        for (int i = 0; i < 6; i++) m.I32(sheet + 0x10 + (ulong)(4 * i), ranges[i]);
        m.U64(sheet + 0x38, 4);
        ulong names = m.Alloc(16), descs = m.Alloc(16), values = m.Alloc(64);
        m.U64(sheet + 0x28, names); m.U64(sheet + 0x48, descs); m.U64(sheet + 0x68, values);
        int[] ids = { 10, 11, 12, 13 };
        uint[] d = { 0u, 4u | (1u << 30), 16u, 32u };
        for (int i = 0; i < 4; i++) { m.I32(names + (ulong)(4 * i), ids[i]); m.U32(descs + (ulong)(4 * i), d[i]); }
        m.Write(values, BitConverter.GetBytes(1.5f)); m.Write(values + 4, BitConverter.GetBytes(0.25f));
        for (int i = 0; i < 4; i++) m.Write(values + 16 + (ulong)(4 * i), BitConverter.GetBytes(i + 0.5f));
        m.I32(values + 32, 777);
        var s = SheetReader.Read(m, sheet);
        Check(s.Entries.Select(e => e.Type).SequenceEqual(new[] { "float", "float", "vector", "texture" }), "sheet types");
        Check(s.Entries[1].GammaFlag && !s.Entries[0].GammaFlag, "sheet gamma flag");
        Check(BitConverter.ToSingle(s.Entries[1].Value, 0) == 0.25f, "sheet raw float");
        Check(s.Entries[2].Value.Length == 16 && BitConverter.ToSingle(s.Entries[2].Value, 12) == 3.5f, "sheet vector");
        Check(BitConverter.ToInt32(s.Entries[3].Value, 0) == 777, "sheet texture id");
        var again = SheetReader.Read(m, sheet);
        Check(again.Hash == s.Hash, "sheet hash stable");
        m.Write(values, BitConverter.GetBytes(2.0f));
        Check(SheetReader.Read(m, sheet).Hash != s.Hash, "sheet hash changes with value");
        var names2 = new[] { null, null, null, null, null, null, null, null, null, null, "_A", "_B", "_C", "_MainTex" };
        var w = new JsonOut(); SheetReader.Write(w, s, id => NameTables.Resolve(names2, id));
        var doc = (Dictionary<string, object>)MiniJson.Parse(w.ToString());
        var entries = (List<object>)doc["entries"];
        Check((string)((Dictionary<string, object>)entries[3])["name"] == "_MainTex", "sheet JSON names");
        m.I32(sheet + 0x10 + 8, 1);   // non-monotonic ranges
        Throws<NativeReadError>(() => SheetReader.Read(m, sheet), "sheet ranges validation");
    }

    static void TestNameTables()
    {
        var m = new FakeMemory();
        ulong table = m.Alloc(0x18), arr = m.Alloc(8 * 3), ptr = m.Alloc(8);
        m.U64(ptr, table); m.U64(table, arr); m.I32(table + 0x10, 3);
        m.U64(arr, m.Str("_MainTex")); m.U64(arr + 8, 0); m.U64(arr + 16, m.Str(new string('x', 200)));
        var names = NameTables.PropertyNames(m, ptr);
        Check(names.Length == 3 && names[0] == "_MainTex" && names[1] == null && names[2].Length == 200, "property name table");
        Check(NameTables.Resolve(names, 0) == "_MainTex" && NameTables.Resolve(names, unchecked((int)0x40000005)).StartsWith("<builtin vector"), "name resolve");
        // MSVC std::map: head (isnil) with parent = root
        ulong mp = m.Alloc(8), map = m.Alloc(16), head = m.Alloc(0x30);
        m.U64(mp, map); m.U64(map, head); m.Write(head + 0x19, new byte[] { 1 });
        Func<string, int, ulong> node = (n, idx) => { var a = m.Alloc(0x30); m.U64(a + 0x20, m.Str(n)); m.I32(a + 0x28, idx); m.U64(a, head); m.U64(a + 0x10, head); return a; };
        var root = node("DIRECTIONAL", 5); var left = node("A_KW", 9); var right = node("SHADOWS_SCREEN", 2);
        m.U64(root, left); m.U64(root + 0x10, right); m.U64(head + 8, root);
        var kw = NameTables.Keywords(m, mp);
        Check(kw.Count == 3 && kw[5] == "DIRECTIONAL" && kw[9] == "A_KW" && kw[2] == "SHADOWS_SCREEN", "keyword map");
    }

    static Dictionary<string, string> BaseRequest(string mode)
    {
        var d = new Dictionary<string, string>
        {
            { "Mode", mode }, { "Nonce", "0123456789abcdef0123456789abcdef" }, { "OutDir", @"E:\x\window" }, { "RawOutDir", @"E:\x\raw" },
            { "Frames", "8" }, { "TriggerIndex", "3" }, { "PollSeconds", "90" }, { "CapturePrefix", @"E:\x\rdc" },
            { "ExpectedRenderDocSha256", new string('a', 64) }, { "ExpectedUnityPlayerSha256", UnityPlayerLayout.Sha256 },
            { "ReadbackDll", @"E:\g\HS2_RenderReadback.dll" }, { "ExpectedReadbackSha256", new string('b', 64) },
            { "RawTimeoutSeconds", "600" }, { "RawMinFreeGB", "50" }, { "ShaderManifest", @"E:\x\m.tsv" }, { "Meshes", "o_head" },
            { "DriverHooks", "1" },
        };
        if (mode == "renderstyle")
        {
            d["BridgeUrl"] = "http://127.0.0.1:43127/maker/renderstyle/capture"; d["BridgeBodyFile"] = @"E:\x\body.json";
            d["ExpectedBridgeSha256"] = new string('c', 64); d["BridgeDispatchRequestFile"] = @"E:\g\HS2_RenderDocScope.request.json";
            d["BridgeTimeoutSeconds"] = "120";
        }
        return d;
    }

    static void TestRequest()
    {
        Func<string, bool> none = p => false;
        var r = ProbeRequest.Parse(BaseRequest("frame"), none);
        Check(r.ExpectedCaptureIndex == 5 && r.RawIndex == 4 && r.PostIndex == -1, "frame plan indices");
        Check(r.ActionsAt(3).Last() == "renderdoc_trigger" && r.ActionsAt(4).Contains("raw_export") && !r.ActionsAt(5).Contains("renderdoc_trigger"), "frame actions");
        var b = ProbeRequest.Parse(BaseRequest("renderstyle"), none);
        var a4 = b.ActionsAt(4);
        Check(a4.SequenceEqual(new[] { "export_frame", "export_scene", "write_marker", "raw_export_blocking", "bridge_post_until_queued" }), "renderstyle post frame order " + string.Join(",", a4));
        Check(b.ActionsAt(3).SequenceEqual(new[] { "export_frame", "export_scene", "write_marker", "renderdoc_trigger" }), "trigger frame order");
        var d = BaseRequest("frame"); d["BridgeUrl"] = "x";
        Throws<ArgumentException>(() => ProbeRequest.Parse(d, none), "bridge keys rejected in frame mode");
        d = BaseRequest("renderstyle"); d["Frames"] = "6";
        Throws<ArgumentException>(() => ProbeRequest.Parse(d, none), "renderstyle needs a frame after the bridge frame");
        d = BaseRequest("frame"); d["TriggerIndex"] = "0";
        Throws<ArgumentException>(() => ProbeRequest.Parse(d, none), "trigger index 0 rejected");
        Throws<ArgumentException>(() => ProbeRequest.Parse(BaseRequest("frame"), p => p.EndsWith("raw")), "existing RawOutDir rejected");
        d = BaseRequest("frame"); d["ExpectedUnityPlayerSha256"] = new string('d', 64);
        Throws<ArgumentException>(() => ProbeRequest.Parse(d, none), "other UnityPlayer rejected");
        Throws<FormatException>(() => ProbeRequest.ParseIni("A=1\nA=2"), "duplicate ini key");
        var ini = ProbeRequest.ParseIni("# c\nMode=frame\r\nOutDir = E:\\a=b\n");
        Check(ini["OutDir"] == "E:\\a=b" && ini.Count == 2, "ini parse");
    }

    static void TestJson()
    {
        var rng = new Random(20261020);
        var bytes = new byte[4];
        int bad = 0;
        for (int i = 0; i < 200000; i++)
        {
            rng.NextBytes(bytes);
            float f = BitConverter.ToSingle(bytes, 0);
            if (float.IsNaN(f) || float.IsInfinity(f)) continue;
            var s = JsonOut.Float(f);
            float back = s == "-0.0" ? -0.0f : float.Parse(s, System.Globalization.CultureInfo.InvariantCulture);
            if (BitConverter.ToInt32(BitConverter.GetBytes(back), 0) != BitConverter.ToInt32(bytes, 0)) bad++;
        }
        foreach (var f in new[] { float.Epsilon, -float.Epsilon, float.MaxValue, 0.1f, 1f / 3f, 16777217f, 1.17549435e-38f })
        {
            var s = JsonOut.Float(f);
            if (BitConverter.ToInt32(BitConverter.GetBytes(float.Parse(s, System.Globalization.CultureInfo.InvariantCulture)), 0) != BitConverter.ToInt32(BitConverter.GetBytes(f), 0)) bad++;
        }
        Check(bad == 0, "float32 round trip (" + bad + " mismatches)");
        Check(JsonOut.Float(-0.0f) == "-0.0" && JsonOut.Float(float.NaN).Contains("nonfinite_bits"), "signed zero and NaN");
        var w = new JsonOut();
        w.Begin('{'); w.Key("ok"); w.Int(1);
        w.Guard("bad", x => { x.Begin('{'); x.Key("partial"); throw new InvalidOperationException("boom"); });
        w.Guard("good", x => { x.Begin('['); x.Num(1.5f); x.Str("q\"\\\n"); x.End(']'); });
        w.End('}');
        var doc = (Dictionary<string, object>)MiniJson.Parse(w.ToString());
        Check(doc.ContainsKey("bad_error") && !doc.ContainsKey("bad") && ((List<object>)doc["good"]).Count == 2, "Guard isolates a failing section");
        Check((string)((List<object>)doc["good"])[1] == "q\"\\\n", "string escapes round trip");
    }

    static void TestBridgeBody()
    {
        Func<string, bool> none = p => false;
        var ok = BridgeBody.Parse("{\"path\":\"E:\\\\x\\\\a.png\",\"yaw\":0,\"target\":[0,1.5,0],\"ortho_size\":0.16,\"width\":768,\"height\":1024,\"freeze_pose\":true}", none);
        Check(ok.Width == 768 && ok.Height == 1024 && ok.PngPath == "E:\\x\\a.png", "bridge body");
        Throws<ArgumentException>(() => BridgeBody.Parse("{\"path\":\"E:\\\\a.png\",\"target\":[0,0,0],\"ortho_size\":1,\"width\":64,\"height\":64,\"geometry_out_path\":\"E:\\\\g.json\"}", none), "geometry_out_path rejected (not in the installed bridge)");
        Throws<ArgumentException>(() => BridgeBody.Parse("{\"path\":\"E:\\\\a.png\",\"target\":[0,0,0],\"ortho_size\":1}", none), "width/height required");
        Throws<ArgumentException>(() => BridgeBody.Parse("{\"path\":\"E:\\\\a.png\",\"target\":[0,0,0],\"ortho_size\":1,\"width\":64,\"height\":64}", p => true), "existing png rejected");
        Throws<FormatException>(() => MiniJson.Parse("{\"a\":1,\"a\":2}"), "duplicate JSON key");
    }

    static void TestV6()
    {
        Func<string, bool> none = p => false;
        var q = ProbeRequest.Parse(new Dictionary<string, string> { { "Mode", "quit" }, { "Nonce", "0123456789abcdef0123456789abcdef" }, { "OutDir", @"C:\s\quit_x" } }, none);
        Check(q.Mode == "quit" && q.OutDir == @"C:\s\quit_x", "quit request");
        Throws<ArgumentException>(() => ProbeRequest.Parse(new Dictionary<string, string> { { "Mode", "quit" }, { "Nonce", "0123456789abcdef0123456789abcdef" }, { "OutDir", @"C:\s\q" }, { "Frames", "8" } }, none), "quit rejects window keys");
        Throws<ArgumentException>(() => ProbeRequest.Parse(new Dictionary<string, string> { { "Mode", "quit" }, { "Nonce", "0123456789abcdef0123456789abcdef" }, { "OutDir", @"C:\s\q" } }, p => true), "quit needs a fresh OutDir");
        var d = BaseRequest("frame"); d.Remove("DriverHooks");
        Throws<ArgumentException>(() => ProbeRequest.Parse(d, none), "DriverHooks required");
        d = BaseRequest("frame"); d["DriverHooks"] = "yes";
        Throws<ArgumentException>(() => ProbeRequest.Parse(d, none), "DriverHooks must be 0/1");
        d = BaseRequest("frame"); d["DriverHooks"] = "0";
        Check(!ProbeRequest.Parse(d, none).DriverHooks && ProbeRequest.Parse(BaseRequest("frame"), none).DriverHooks, "DriverHooks flag");

        // Skin double buffer: buffers A and B alternate as slot 0 at end of frame (A in 8748, B in 8749,
        // A in 8750). At bridge time of 8750 (before its skinning) slot0 = A holds 8748, slot1 = B holds 8749;
        // after 8750's skinning slot0 = A holds 8750 (the session v5.0.1 mode-b pairing).
        var tr = new SkinSlotTracker();
        ulong A = 0x1000, B = 0x2000;
        Check(tr.HoldsFrame(A) == -1, "unobserved buffer is unknown");
        tr.EndOfFrame(8748, A); tr.EndOfFrame(8749, B);
        Check(tr.HoldsFrame(A) == 8748 && tr.HoldsFrame(B) == 8749, "bridge-time labels N-2 / N-1");
        tr.EndOfFrame(8750, A);
        Check(tr.HoldsFrame(A) == 8750 && tr.HoldsFrame(B) == 8749 && tr.HoldsFrame(0) == -1, "end-of-frame labels");

        var w = new JsonOut();
        w.Begin('['); w.GuardValue(x => x.Int(1)); w.GuardValue(x => { x.Begin('{'); throw new InvalidOperationException("boom"); }); w.End(']');
        var arr = (List<object>)MiniJson.Parse(w.ToString());
        Check(arr.Count == 2 && (double)arr[0] == 1 && ((Dictionary<string, object>)arr[1]).ContainsKey("error"), "GuardValue array isolation");
        long s1 = ProbeSequence.Next(), s2 = ProbeSequence.Next();
        Check(s2 == s1 + 1, "sequence counter");
    }

    // Probe 6.1.0: SkinnedMeshRenderer::AddAsRenderNode (0x77abe0) operands are relative to the
    // BaseRenderer subobject at Renderer+0x38.
    static void TestSmrNative()
    {
        Check(UnityPlayerLayout.SmrField(0x1000, UnityPlayerLayout.SmrBaseMesh) == 0x1000 + 0x1d8, "mesh at Renderer+0x1d8");
        Check(UnityPlayerLayout.SmrField(0, UnityPlayerLayout.SmrBaseHasLastPosition) == 0x28d
              && UnityPlayerLayout.SmrField(0, UnityPlayerLayout.SmrBaseForceMatrixRecalc) == 0x28e
              && UnityPlayerLayout.SmrField(0, UnityPlayerLayout.SmrBaseSkinBuffer) == 0x290
              && UnityPlayerLayout.SmrField(0, UnityPlayerLayout.SmrBasePrevSkinBuffer) == 0x298
              && UnityPlayerLayout.SmrField(0, UnityPlayerLayout.SmrBaseDirty) == 0x2c4, "Renderer-relative skin field offsets");

        var m = new FakeMemory();
        ulong smr = m.Alloc(0x400), mesh = 0x7700ABCDUL;
        m.U64(smr + 0x1d8, mesh);
        m.U64(smr + 0x290, 0x5000); m.U64(smr + 0x298, 0x6000);
        // The v6.0.0 (wrong) slots hold unrelated data, as observed for o_head.
        m.U64(smr + 0x258, 0x16a00000034UL); m.U64(smr + 0x260, 0x3a);
        Check(SmrNative.SkinBuffer(m, smr, mesh, "o_head") == 0x5000, "slot 0 read at Renderer+0x290 after the mesh check");
        Throws<NativeReadError>(() => SmrNative.RequireMesh(m, smr, mesh + 8, "o_head"), "cached mesh differs from sharedMesh");
        Throws<NativeReadError>(() => SmrNative.SkinBuffer(m, smr, 0, "o_head"), "null sharedMesh with a cached mesh");
        Throws<NativeReadError>(() => SmrNative.RequireMesh(m, 0, mesh, "o_head"), "renderer without native object");
        Throws<NativeReadError>(() => SmrNative.RequireMesh(m, 0x9990000, mesh, "o_head"), "unreadable renderer");
        ulong empty = m.Alloc(0x400);
        SmrNative.RequireMesh(m, empty, 0, "no_mesh");
        Check(true, "null sharedMesh with null cached mesh accepted");
    }

    static int Main()
    {
        foreach (var t in new Action[] { TestPool, TestSheet, TestNameTables, TestRequest, TestJson, TestBridgeBody, TestV6, TestSmrNative })
        {
            try { t(); }
            catch (Exception e) { failures++; Console.WriteLine("FAIL (exception in " + t.Method.Name + "): " + e); }
        }
        Console.WriteLine("ProbeCoreTests: " + passes + " passed, " + failures + " failed");
        return failures == 0 ? 0 : 1;
    }
}
