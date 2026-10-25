// HS2 offline renderer frame-program executor (D3D11, MSVC).
//
//   frame_executor.exe PROGRAM_DIR OUT_DIR [--warp] [--adapter-index N] [--debug-layer] [--hash-only]
//   frame_executor.exe PROGRAM_DIR OUT_DIR --serve [--warp] [--adapter-index N]
// --serve creates the device and every program object once, then executes the command list once per
// request line read from stdin (JSON: {"out", "resource_data", "map_write_blobs", "readbacks", "hash_only"}).
// Before every request each resource the frame writes (or a request patches) is restored to its
// state after program creation: DEFAULT resources from a GPU copy taken at creation, DYNAMIC ones from
// their initial bytes (Map WRITE_DISCARD).  One result line is printed per request.
//
// --hash-only computes and records every readback's SHA-256 without writing the raw bytes
// (disk-constrained acceptance loops); the default writes <name>.raw.bin (+ .topdown.bin).
//
// Loads a frame program (src/render/frame_program.py contract), creates every
// resource, view, state and shader from the exact bytes, executes the ordered
// commands and writes every requested readback as raw bytes plus a JSON
// descriptor. Nothing is decoded, converted or defaulted implicitly:
//  * descriptors are verbatim D3D11 struct fields; unknown keys are rejected;
//  * every blob is SHA-256 verified against its content address before use;
//  * each draw/dispatch starts from ID3D11DeviceContext::ClearState and binds its
//    complete snapshot (outputs first, then inputs); the bound state is read back
//    through the Get* API and any slot the runtime refused (hazard) is an error;
//  * Map/Unmap writes reproduce RenderDoc v1.46 Serialise_Unmap replay exactly
//    (NO_OVERWRITE via an immutable temp buffer + CopySubresourceRegion, other
//    types via a real Map and MapIntercept::CopyToD3D(start, end));
//  * readbacks follow D3D11Replay::GetTextureData: CopyResource into a STAGING
//    copy of the descriptor, Map(READ), MapIntercept::CopyFromD3D tight packing.
// Unsupported features (OM UAVs, stream-out, predication, MSAA readback, class
// linkage, rasterizer desc1/2, ...) fail explicitly.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
static const char* EXECUTOR_VERSION = "hs2_frame_executor/1+serve";

// ------------------------------------------------------------------ errors
struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };
static std::string hex32(uint32_t v) { char b[16]; snprintf(b, sizeof b, "0x%08X", v); return b; }
static void check(HRESULT hr, const std::string& where) {
    if (FAILED(hr)) throw Fail(where + " failed HRESULT " + hex32((uint32_t)hr));
}
template <class T> struct Com {
    T* p = nullptr;
    Com() = default;
    Com(const Com&) = delete;
    Com& operator=(const Com&) = delete;
    Com(Com&& o) noexcept : p(o.p) { o.p = nullptr; }
    Com& operator=(Com&& o) noexcept { if (this != &o) { reset(); p = o.p; o.p = nullptr; } return *this; }
    ~Com() { reset(); }
    void reset() { if (p) p->Release(); p = nullptr; }
    T** put() { reset(); return &p; }
    T* operator->() const { return p; }
    T* get() const { return p; }
};

// -------------------------------------------------------------------- JSON
struct J {
    enum T { Null, Bool, Num, Str, Arr, Obj } t = Null;
    bool b = false;
    std::string s;  // string value or raw number text
    std::vector<J> a;
    std::vector<std::pair<std::string, J>> o;
    const J* find(const std::string& k) const {
        for (auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
    bool isNull() const { return t == Null; }
};
struct Parser {
    const char* p; const char* e;
    [[noreturn]] void bad(const char* what) { throw Fail(std::string("program.json parse error: ") + what); }
    void ws() { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p; }
    J value() {
        ws(); if (p >= e) bad("unexpected end");
        J v;
        if (*p == '{') {
            v.t = J::Obj; ++p; ws();
            if (p < e && *p == '}') { ++p; return v; }
            for (;;) {
                ws(); if (p >= e || *p != '"') bad("object key");
                std::string k = str(); ws();
                if (p >= e || *p != ':') bad("colon"); ++p;
                v.o.emplace_back(std::move(k), value()); ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == '}') { ++p; break; }
                bad("object separator");
            }
        } else if (*p == '[') {
            v.t = J::Arr; ++p; ws();
            if (p < e && *p == ']') { ++p; return v; }
            for (;;) {
                v.a.push_back(value()); ws();
                if (p < e && *p == ',') { ++p; continue; }
                if (p < e && *p == ']') { ++p; break; }
                bad("array separator");
            }
        } else if (*p == '"') { v.t = J::Str; v.s = str(); }
        else if (!strncmp(p, "null", 4)) { p += 4; }
        else if (!strncmp(p, "true", 4)) { v.t = J::Bool; v.b = true; p += 4; }
        else if (!strncmp(p, "false", 5)) { v.t = J::Bool; p += 5; }
        else {
            const char* s0 = p;
            if (*p == '-') ++p;
            while (p < e && (isdigit((unsigned char)*p) || *p == '.' || *p == 'e' || *p == 'E' || *p == '+' || *p == '-')) ++p;
            if (p == s0) bad("value");
            v.t = J::Num; v.s.assign(s0, p);
        }
        return v;
    }
    std::string str() {
        ++p; std::string out;
        while (p < e && *p != '"') {
            if (*p == '\\') {
                ++p; if (p >= e) bad("escape");
                char c = *p++;
                switch (c) {
                case '"': out += '"'; break; case '\\': out += '\\'; break; case '/': out += '/'; break;
                case 'b': out += '\b'; break; case 'f': out += '\f'; break; case 'n': out += '\n'; break;
                case 'r': out += '\r'; break; case 't': out += '\t'; break;
                case 'u': {
                    if (e - p < 4) bad("unicode escape");
                    unsigned cp = std::stoul(std::string(p, p + 4), nullptr, 16); p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && e - p >= 6 && p[0] == '\\' && p[1] == 'u') {
                        unsigned lo = std::stoul(std::string(p + 2, p + 6), nullptr, 16); p += 6;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    if (cp < 0x80) out += (char)cp;
                    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                    else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: bad("escape char");
                }
            } else out += *p++;
        }
        if (p >= e) bad("unterminated string");
        ++p; return out;
    }
};

static uint64_t ju(const J& v, const std::string& what) {
    if (v.t != J::Num || v.s.find_first_of(".eE") != std::string::npos || (!v.s.empty() && v.s[0] == '-'))
        throw Fail(what + ": unsigned integer required, got '" + v.s + "'");
    return std::stoull(v.s);
}
static int64_t ji(const J& v, const std::string& what) {
    if (v.t != J::Num || v.s.find_first_of(".eE") != std::string::npos) throw Fail(what + ": integer required");
    return std::stoll(v.s);
}
static double jd(const J& v, const std::string& what) {
    if (v.t != J::Num) throw Fail(what + ": number required");
    return strtod(v.s.c_str(), nullptr);
}
static float jf(const J& v, const std::string& what) {
    double d = jd(v, what); float f = (float)d;
    if ((double)f != d && !(d != d)) throw Fail(what + ": value " + v.s + " is not exactly representable as float32");
    return f;
}
static bool jb(const J& v, const std::string& what) {
    if (v.t != J::Bool) throw Fail(what + ": boolean required");
    return v.b;
}
static const std::string& js(const J& v, const std::string& what) {
    if (v.t != J::Str) throw Fail(what + ": string required");
    return v.s;
}

// Strict field access: every key of the object must be consumed (verbatim descriptors).
struct Fields {
    const J& obj; std::string where; std::set<std::string> used;
    Fields(const J& o, std::string w) : obj(o), where(std::move(w)) {
        if (o.t != J::Obj) throw Fail(where + ": object required");
    }
    const J& req(const std::string& k) {
        const J* v = obj.find(k); if (!v) throw Fail(where + ": missing field " + k);
        used.insert(k); return *v;
    }
    const J* opt(const std::string& k) { const J* v = obj.find(k); if (v) used.insert(k); return v; }
    uint32_t u32(const std::string& k) {
        uint64_t v = ju(req(k), where + "." + k);
        if (v > 0xFFFFFFFFull) throw Fail(where + "." + k + " exceeds 32 bits"); return (uint32_t)v;
    }
    int32_t i32(const std::string& k) {
        int64_t v = ji(req(k), where + "." + k);
        if (v < INT32_MIN || v > INT32_MAX) throw Fail(where + "." + k + " exceeds int32"); return (int32_t)v;
    }
    float f32(const std::string& k) { return jf(req(k), where + "." + k); }
    BOOL boolean(const std::string& k) { return jb(req(k), where + "." + k) ? TRUE : FALSE; }
    void done() {
        for (auto& kv : obj.o) if (!used.count(kv.first)) throw Fail(where + ": unsupported field '" + kv.first + "'");
    }
};

// ------------------------------------------------------------------ SHA-256
static std::string sha256(const void* data, size_t n) {
    BCRYPT_ALG_HANDLE alg = nullptr; BCRYPT_HASH_HANDLE h = nullptr; unsigned char out[32];
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) throw Fail("BCrypt SHA256 unavailable");
    if (BCryptCreateHash(alg, &h, nullptr, 0, nullptr, 0, 0) != 0) throw Fail("BCryptCreateHash");
    const unsigned char* p = (const unsigned char*)data;
    while (n) { ULONG k = (ULONG)std::min<size_t>(n, 1u << 30); BCryptHashData(h, (PUCHAR)p, k, 0); p += k; n -= k; }
    BCryptFinishHash(h, out, 32, 0); BCryptDestroyHash(h); BCryptCloseAlgorithmProvider(alg, 0);
    static const char* x = "0123456789abcdef"; std::string s;
    for (unsigned char c : out) { s += x[c >> 4]; s += x[c & 15]; }
    return s;
}
static std::vector<char> readFile(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw Fail("Missing file " + path.string());
    auto n = f.tellg(); std::vector<char> d((size_t)n); f.seekg(0);
    if (n > 0) f.read(d.data(), n);
    if (!f) throw Fail("Short read " + path.string());
    return d;
}
static void writeFile(const fs::path& path, const void* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw Fail("Cannot write " + path.string());
    f.write((const char*)data, (std::streamsize)n);
    if (!f) throw Fail("Short write " + path.string());
}

// ------------------------------------------------------------- DXGI layout
struct Block { uint32_t w, h, bytes; };
static Block blockLayout(uint32_t fmt) {
    if ((fmt >= 70 && fmt <= 84) || (fmt >= 94 && fmt <= 99)) {
        bool eight = fmt == 70 || fmt == 71 || fmt == 72 || fmt == 79 || fmt == 80 || fmt == 81;
        return {4, 4, eight ? 8u : 16u};
    }
    if (fmt == 66) return {8, 1, 1};
    if (fmt == 68 || fmt == 69) return {2, 1, 4};
    uint32_t n = 0;
    if (fmt >= 1 && fmt <= 4) n = 16;
    else if (fmt >= 5 && fmt <= 8) n = 12;
    else if (fmt >= 9 && fmt <= 22) n = 8;
    else if ((fmt >= 23 && fmt <= 47) || fmt == 67 || (fmt >= 87 && fmt <= 93)) n = 4;
    else if ((fmt >= 48 && fmt <= 59) || fmt == 85 || fmt == 86 || fmt == 115) n = 2;
    else if (fmt >= 60 && fmt <= 65) n = 1;
    else throw Fail("DXGI storage format not implemented: " + std::to_string(fmt));
    return {1, 1, n};
}
static uint32_t ceilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
// RenderDoc GetByteSize(w, h, d, fmt, mip) for non-YUV formats.
static uint64_t byteSize(uint32_t w, uint32_t h, uint32_t d, uint32_t fmt, uint32_t mip) {
    Block b = blockLayout(fmt);
    uint64_t W = std::max(1u, w >> mip), H = std::max(1u, h >> mip), D = std::max(1u, d >> mip);
    return (uint64_t)ceilDiv((uint32_t)W, b.w) * b.bytes * ceilDiv((uint32_t)H, b.h) * D;
}
static uint32_t rowPitch(uint32_t w, uint32_t fmt, uint32_t mip) { return (uint32_t)byteSize(w, 1, 1, fmt, mip); }
static bool isBlockFormat(uint32_t fmt) { return (fmt >= 70 && fmt <= 84) || (fmt >= 94 && fmt <= 99); }

// ------------------------------------------------------------- program model
enum class Kind { Buffer, Tex1D, Tex2D, Tex3D };
struct Resource {
    uint64_t id; Kind kind; std::string name;
    D3D11_BUFFER_DESC bd{}; D3D11_TEXTURE1D_DESC t1{}; D3D11_TEXTURE2D_DESC t2{}; D3D11_TEXTURE3D_DESC t3{};
    Com<ID3D11Resource> res;
    uint32_t mips = 1, arrays = 1;
    uint32_t format() const { return kind == Kind::Tex1D ? t1.Format : kind == Kind::Tex2D ? t2.Format : kind == Kind::Tex3D ? t3.Format : 0; }
    uint32_t width() const { return kind == Kind::Tex1D ? t1.Width : kind == Kind::Tex2D ? t2.Width : kind == Kind::Tex3D ? t3.Width : bd.ByteWidth; }
    uint32_t height() const { return kind == Kind::Tex2D ? t2.Height : kind == Kind::Tex3D ? t3.Height : 1; }
    uint32_t depth() const { return kind == Kind::Tex3D ? t3.Depth : 1; }
    uint32_t usage() const { return kind == Kind::Tex1D ? t1.Usage : kind == Kind::Tex2D ? t2.Usage : kind == Kind::Tex3D ? t3.Usage : bd.Usage; }
    uint32_t bind() const { return kind == Kind::Tex1D ? t1.BindFlags : kind == Kind::Tex2D ? t2.BindFlags : kind == Kind::Tex3D ? t3.BindFlags : bd.BindFlags; }
    uint32_t subresources() const { return kind == Kind::Buffer ? 1 : mips * arrays; }
};
struct Program;
struct Executor {
    fs::path root, out;
    J prog;
    Com<ID3D11Device> dev; Com<ID3D11DeviceContext> ctx; Com<ID3D11DeviceContext1> ctx1;
    std::unordered_map<uint64_t, std::unique_ptr<Resource>> res;
    std::unordered_map<uint64_t, Com<ID3D11ShaderResourceView>> srv;
    std::unordered_map<uint64_t, Com<ID3D11RenderTargetView>> rtv;
    std::unordered_map<uint64_t, Com<ID3D11DepthStencilView>> dsv;
    std::unordered_map<uint64_t, Com<ID3D11UnorderedAccessView>> uav;
    std::unordered_map<uint64_t, uint64_t> viewResource;
    std::unordered_map<uint64_t, Com<ID3D11DeviceChild>> shaders;
    std::unordered_map<uint64_t, std::string> shaderStage;
    std::unordered_map<uint64_t, Com<ID3D11InputLayout>> layouts;
    std::unordered_map<uint64_t, Com<ID3D11BlendState>> blends;
    std::unordered_map<uint64_t, Com<ID3D11DepthStencilState>> depths;
    std::unordered_map<uint64_t, Com<ID3D11RasterizerState>> rasters;
    std::unordered_map<uint64_t, Com<ID3D11SamplerState>> samplers;
    std::unordered_map<std::string, Com<ID3D11Buffer>> blobCBs;
    std::set<uint64_t> ids;
    std::unordered_map<std::string, std::vector<char>> blobCache;
    uint64_t blobBytesVerified = 0, blobsVerified = 0;
    std::string adapterJson;
    std::ostringstream readbackJson; bool firstReadback = true; bool hashOnly = false;
    std::map<std::string, double> timing;
    std::vector<std::pair<double, uint64_t>> slowCreates;

    // ---- serve mode state
    std::unordered_map<size_t, std::vector<char>> cmdBlobOverride;      // command index -> replacement map_write bytes
    std::unordered_map<uint64_t, Com<ID3D11Resource>> pristine;          // DEFAULT resources: copy after creation
    std::unordered_map<uint64_t, std::vector<std::vector<char>>> pristineCpu;  // DYNAMIC resources: initial bytes
    std::unordered_map<uint64_t, std::vector<uint32_t>> pristinePitch;   // row pitch per subresource (DYNAMIC textures)
    std::set<uint64_t> restoreSet;
    const J* readbackOverride = nullptr;
    long long stopAfter = -1;                                            // serve: last command executed (-1: all)
    void claim(uint64_t id, const std::string& what) {
        if (!id) throw Fail(what + ": id 0 is reserved");
        if (!ids.insert(id).second) throw Fail(what + ": duplicate object id " + std::to_string(id));
    }
    std::vector<char> blob(const std::string& h, bool cache = true) {
        if (h.size() != 64) throw Fail("Invalid blob address " + h);
        auto it = blobCache.find(h);
        if (it != blobCache.end()) return it->second;
        auto d = readFile(root / "blobs" / h);
        if (sha256(d.data(), d.size()) != h) throw Fail("Blob content does not match its SHA-256 address " + h);
        blobBytesVerified += d.size(); ++blobsVerified;
        if (cache && d.size() <= (1u << 20)) blobCache[h] = d;
        return d;
    }
    Resource& resource(uint64_t id, const std::string& what) {
        auto it = res.find(id); if (it == res.end()) throw Fail(what + ": unknown resource " + std::to_string(id));
        return *it->second;
    }
    template <class M> auto get(M& m, uint64_t id, const std::string& what) -> decltype(m.begin()->second.get()) {
        auto it = m.find(id); if (it == m.end()) throw Fail(what + ": unknown object " + std::to_string(id));
        return it->second.get();
    }
    template <class M> auto optGet(M& m, const J* v, const std::string& what) -> decltype(m.begin()->second.get()) {
        if (!v || v->isNull()) return nullptr;
        return get(m, ju(*v, what), what);
    }

    // ------------------------------------------------------------ device
    void createDevice(bool warp, int adapterIndex, bool debugLayer) {
        UINT flags = debugLayer ? D3D11_CREATE_DEVICE_DEBUG : 0;
        D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
        D3D_FEATURE_LEVEL got{};
        Com<IDXGIAdapter1> adapter;
        std::string selection;
        if (warp) {
            check(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags, levels, 2, D3D11_SDK_VERSION,
                                    dev.put(), &got, ctx.put()), "D3D11CreateDevice(WARP)");
            selection = "warp";
        } else {
            Com<IDXGIFactory1> factory;
            check(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)factory.put()), "CreateDXGIFactory1");
            const J* cap = nullptr;
            if (const J* p = prog.find("provenance")) cap = p->find("capture_adapter");
            uint64_t vendor = 0, device = 0;
            if (cap && !cap->isNull()) {
                vendor = ju(*cap->find("VendorId"), "capture_adapter.VendorId");
                device = ju(*cap->find("DeviceId"), "capture_adapter.DeviceId");
            }
            for (UINT i = 0;; ++i) {
                Com<IDXGIAdapter1> a;
                if (factory->EnumAdapters1(i, a.put()) == DXGI_ERROR_NOT_FOUND) break;
                DXGI_ADAPTER_DESC1 d{}; a->GetDesc1(&d);
                bool pick = adapterIndex >= 0 ? (int)i == adapterIndex
                                              : (vendor ? (d.VendorId == vendor && d.DeviceId == device)
                                                        : (i == 0 && !(d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)));
                if (pick) { adapter = std::move(a); selection = adapterIndex >= 0 ? "explicit index " + std::to_string(i)
                                                                         : vendor ? "matches capture VendorId/DeviceId" : "adapter 0"; break; }
            }
            if (!adapter.get()) throw Fail("No hardware adapter matches the selection (capture adapter or --adapter-index); "
                                           "nothing is substituted");
            DXGI_ADAPTER_DESC1 d{}; adapter->GetDesc1(&d);
            if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) throw Fail("Selected adapter is a software adapter; use --warp explicitly");
            check(D3D11CreateDevice(adapter.get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, flags, levels, 2, D3D11_SDK_VERSION,
                                    dev.put(), &got, ctx.put()), "D3D11CreateDevice(hardware)");
        }
        if (got < D3D_FEATURE_LEVEL_11_0) throw Fail("Feature level 11_0 required");
        check(ctx->QueryInterface(__uuidof(ID3D11DeviceContext1), (void**)ctx1.put()), "ID3D11DeviceContext1");
        Com<IDXGIDevice> dxgiDev; check(dev->QueryInterface(__uuidof(IDXGIDevice), (void**)dxgiDev.put()), "IDXGIDevice");
        Com<IDXGIAdapter> used; check(dxgiDev->GetAdapter(used.put()), "GetAdapter");
        DXGI_ADAPTER_DESC d{}; used->GetDesc(&d);
        LARGE_INTEGER umd{}; bool haveUmd = SUCCEEDED(used->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd));
        char desc[256]; WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, desc, sizeof desc, nullptr, nullptr);
        std::string sd; for (char* c = desc; *c; ++c) { if (*c == '"' || *c == '\\') sd += '\\'; sd += *c; }
        std::ostringstream o;
        o << "{\"driver_type\":\"" << (warp ? "warp" : "hardware") << "\",\"selection\":\"" << selection
          << "\",\"description\":\"" << sd << "\",\"vendor_id\":" << d.VendorId << ",\"device_id\":" << d.DeviceId
          << ",\"subsys_id\":" << d.SubSysId << ",\"revision\":" << d.Revision << ",\"luid\":\"" << d.AdapterLuid.HighPart
          << ":" << d.AdapterLuid.LowPart << "\",\"feature_level\":\"" << hex32((uint32_t)got) << "\",\"umd_version\":"
          << (haveUmd ? "\"" + std::to_string(HIWORD(umd.HighPart)) + "." + std::to_string(LOWORD(umd.HighPart)) + "." +
                            std::to_string(HIWORD(umd.LowPart)) + "." + std::to_string(LOWORD(umd.LowPart)) + "\"" : "null")
          << ",\"debug_layer\":" << (debugLayer ? "true" : "false") << "}";
        adapterJson = o.str();
    }

    // --------------------------------------------------------- resources
    void createResources() {
        for (const J& r : prog.find("resources")->a) {
            Fields f(r, "resource");
            uint64_t id = ju(f.req("id"), "resource.id"); claim(id, "resource");
            auto R = std::make_unique<Resource>(); R->id = id;
            const std::string& kind = js(f.req("kind"), "resource.kind");
            R->name = js(f.req("name"), "resource.name");
            f.opt("provenance");
            std::string where = "resource " + std::to_string(id);
            Fields d(f.req("desc"), where + ".desc");
            if (kind == "buffer") {
                R->kind = Kind::Buffer;
                R->bd = {d.u32("ByteWidth"), (D3D11_USAGE)d.u32("Usage"), d.u32("BindFlags"), d.u32("CPUAccessFlags"),
                         d.u32("MiscFlags"), d.u32("StructureByteStride")};
            } else if (kind == "tex1d") {
                R->kind = Kind::Tex1D;
                auto& t = R->t1; t.Width = d.u32("Width"); t.MipLevels = d.u32("MipLevels"); t.ArraySize = d.u32("ArraySize");
                t.Format = (DXGI_FORMAT)d.u32("Format"); t.Usage = (D3D11_USAGE)d.u32("Usage"); t.BindFlags = d.u32("BindFlags");
                t.CPUAccessFlags = d.u32("CPUAccessFlags"); t.MiscFlags = d.u32("MiscFlags");
            } else if (kind == "tex2d" || kind == "texcube") {
                R->kind = Kind::Tex2D;
                auto& t = R->t2; t.Width = d.u32("Width"); t.Height = d.u32("Height"); t.MipLevels = d.u32("MipLevels");
                t.ArraySize = d.u32("ArraySize"); t.Format = (DXGI_FORMAT)d.u32("Format");
                Fields sd(d.req("SampleDesc"), where + ".SampleDesc"); t.SampleDesc.Count = sd.u32("Count"); t.SampleDesc.Quality = sd.u32("Quality"); sd.done();
                t.Usage = (D3D11_USAGE)d.u32("Usage"); t.BindFlags = d.u32("BindFlags");
                t.CPUAccessFlags = d.u32("CPUAccessFlags"); t.MiscFlags = d.u32("MiscFlags");
                if ((kind == "texcube") != ((t.MiscFlags & D3D11_RESOURCE_MISC_TEXTURECUBE) != 0)) throw Fail(where + ": texcube kind/flag mismatch");
            } else if (kind == "tex3d") {
                R->kind = Kind::Tex3D;
                auto& t = R->t3; t.Width = d.u32("Width"); t.Height = d.u32("Height"); t.Depth = d.u32("Depth");
                t.MipLevels = d.u32("MipLevels"); t.Format = (DXGI_FORMAT)d.u32("Format"); t.Usage = (D3D11_USAGE)d.u32("Usage");
                t.BindFlags = d.u32("BindFlags"); t.CPUAccessFlags = d.u32("CPUAccessFlags"); t.MiscFlags = d.u32("MiscFlags");
            } else throw Fail(where + ": unknown kind " + kind);
            d.done();
            if (R->kind != Kind::Buffer) {
                uint32_t ml = R->kind == Kind::Tex1D ? R->t1.MipLevels : R->kind == Kind::Tex2D ? R->t2.MipLevels : R->t3.MipLevels;
                uint32_t big = std::max({R->width(), R->height(), R->depth()}), full = 0; while (big) { ++full; big >>= 1; }
                R->mips = ml ? ml : full;
                R->arrays = R->kind == Kind::Tex1D ? R->t1.ArraySize : R->kind == Kind::Tex2D ? R->t2.ArraySize : 1;
            }
            const J& init = f.req("initial"); f.done();
            std::vector<std::vector<char>> datas; std::vector<D3D11_SUBRESOURCE_DATA> sub;
            if (!init.isNull()) {
                uint32_t n = R->subresources();
                if (init.a.size() != n) throw Fail(where + ": initial data must cover every subresource");
                datas.resize(n); sub.resize(n); std::vector<bool> seen(n);
                for (const J& e : init.a) {
                    Fields fi(e, where + ".initial");
                    uint32_t s = fi.u32("subresource"); if (s >= n || seen[s]) throw Fail(where + ": bad/duplicate initial subresource");
                    seen[s] = true; datas[s] = blob(js(fi.req("blob"), "initial.blob"), false);
                    sub[s].SysMemPitch = fi.u32("row_pitch"); sub[s].SysMemSlicePitch = fi.u32("slice_pitch"); fi.done();
                    if (R->kind != Kind::Buffer) {
                        uint32_t mip = s % R->mips;
                        Block b = blockLayout(R->format());
                        uint64_t rows = ceilDiv(std::max(1u, R->height() >> mip), b.h), depth = std::max(1u, R->depth() >> mip);
                        uint64_t need = (depth - 1) * sub[s].SysMemSlicePitch + (rows - 1) * sub[s].SysMemPitch + rowPitch(R->width(), R->format(), mip);
                        if (datas[s].size() < need || sub[s].SysMemPitch < rowPitch(R->width(), R->format(), mip))
                            throw Fail(where + ": initial blob shorter than its pitched extent");
                    } else if (datas[s].size() != R->bd.ByteWidth) throw Fail(where + ": buffer initial blob must be ByteWidth");
                    sub[s].pSysMem = datas[s].data();
                }
            }
            auto c0 = std::chrono::steady_clock::now();
            create(*R, sub, where);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
            slowCreates.emplace_back(ms, id);
            res[id] = std::move(R);
        }
    }
    void create(Resource& R, std::vector<D3D11_SUBRESOURCE_DATA>& sub, const std::string& where) {
        const D3D11_SUBRESOURCE_DATA* data = sub.empty() ? nullptr : sub.data();
        bool viaCopy = data && (R.bind() & D3D11_BIND_DEPTH_STENCIL);
        if (data && R.kind == Kind::Tex2D && R.t2.SampleDesc.Count > 1) throw Fail(where + ": MSAA initial data is not implemented");
        if (R.kind == Kind::Buffer) {
            Com<ID3D11Buffer> b; check(dev->CreateBuffer(&R.bd, data, b.put()), where + " CreateBuffer");
            R.res.p = b.p; b.p = nullptr; return;
        }
        auto make = [&](const void* descPtr, const D3D11_SUBRESOURCE_DATA* d, ID3D11Resource** outRes, const std::string& w) {
            if (R.kind == Kind::Tex1D) { Com<ID3D11Texture1D> t; check(dev->CreateTexture1D((const D3D11_TEXTURE1D_DESC*)descPtr, d, t.put()), w); *outRes = t.p; t.p = nullptr; }
            else if (R.kind == Kind::Tex2D) { Com<ID3D11Texture2D> t; check(dev->CreateTexture2D((const D3D11_TEXTURE2D_DESC*)descPtr, d, t.put()), w); *outRes = t.p; t.p = nullptr; }
            else { Com<ID3D11Texture3D> t; check(dev->CreateTexture3D((const D3D11_TEXTURE3D_DESC*)descPtr, d, t.put()), w); *outRes = t.p; t.p = nullptr; }
        };
        const void* desc = R.kind == Kind::Tex1D ? (const void*)&R.t1 : R.kind == Kind::Tex2D ? (const void*)&R.t2 : (const void*)&R.t3;
        if (!viaCopy) { make(desc, data, &R.res.p, where + " CreateTexture"); return; }
        // Depth-stencil resources take no initial data; upload through a bind-less DEFAULT copy with the
        // depth->colour format mapping RenderDoc uses for its own initial-state textures, then CopyResource.
        make(desc, nullptr, &R.res.p, where + " CreateTexture(depth)");
        auto mapFmt = [](DXGI_FORMAT f) {
            switch (f) { case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
            case DXGI_FORMAT_D32_FLOAT: return DXGI_FORMAT_R32_FLOAT; case DXGI_FORMAT_D24_UNORM_S8_UINT: return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
            case DXGI_FORMAT_D16_UNORM: return DXGI_FORMAT_R16_FLOAT; default: return f; } };
        Com<ID3D11Resource> tmp;
        if (R.kind == Kind::Tex1D) { auto t = R.t1; t.BindFlags = 0; t.CPUAccessFlags = 0; t.MiscFlags = 0; t.Usage = D3D11_USAGE_DEFAULT; t.Format = mapFmt(t.Format); make(&t, data, &tmp.p, where + " depth upload copy"); }
        else { auto t = R.t2; t.BindFlags = 0; t.CPUAccessFlags = 0; t.MiscFlags = 0; t.Usage = D3D11_USAGE_DEFAULT; t.Format = mapFmt(t.Format); make(&t, data, &tmp.p, where + " depth upload copy"); }
        ctx->CopyResource(R.res.get(), tmp.get());
    }

    // ------------------------------------------------------------- views
    void createViews() {
        for (const J& v : prog.find("views")->a) {
            Fields f(v, "view");
            uint64_t id = ju(f.req("id"), "view.id"); claim(id, "view");
            std::string where = "view " + std::to_string(id);
            const std::string& kind = js(f.req("kind"), where + ".kind");
            Resource& R = resource(ju(f.req("resource"), where + ".resource"), where);
            f.opt("name"); f.opt("provenance");
            const J& dj = f.req("desc"); f.done();
            viewResource[id] = R.id;
            if (kind == "srv") {
                D3D11_SHADER_RESOURCE_VIEW_DESC d{}; bool has = !dj.isNull();
                if (has) {
                    Fields df(dj, where + ".desc"); d.Format = (DXGI_FORMAT)df.u32("Format"); d.ViewDimension = (D3D11_SRV_DIMENSION)df.u32("ViewDimension");
                    auto mem = [&](const char* n) { return Fields(df.req(n), where + "." + n); };
                    switch (d.ViewDimension) {
                    case D3D11_SRV_DIMENSION_BUFFER: { auto m = mem("Buffer"); d.Buffer.FirstElement = m.u32("FirstElement"); d.Buffer.NumElements = m.u32("NumElements"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_BUFFEREX: { auto m = mem("BufferEx"); d.BufferEx.FirstElement = m.u32("FirstElement"); d.BufferEx.NumElements = m.u32("NumElements"); d.BufferEx.Flags = m.u32("Flags"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURE1D: { auto m = mem("Texture1D"); d.Texture1D.MostDetailedMip = m.u32("MostDetailedMip"); d.Texture1D.MipLevels = m.u32("MipLevels"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURE1DARRAY: { auto m = mem("Texture1DArray"); d.Texture1DArray.MostDetailedMip = m.u32("MostDetailedMip"); d.Texture1DArray.MipLevels = m.u32("MipLevels"); d.Texture1DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture1DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURE2D: { auto m = mem("Texture2D"); d.Texture2D.MostDetailedMip = m.u32("MostDetailedMip"); d.Texture2D.MipLevels = m.u32("MipLevels"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURE2DARRAY: { auto m = mem("Texture2DArray"); d.Texture2DArray.MostDetailedMip = m.u32("MostDetailedMip"); d.Texture2DArray.MipLevels = m.u32("MipLevels"); d.Texture2DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture2DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURE2DMS: { auto m = mem("Texture2DMS"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURE2DMSARRAY: { auto m = mem("Texture2DMSArray"); d.Texture2DMSArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture2DMSArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURE3D: { auto m = mem("Texture3D"); d.Texture3D.MostDetailedMip = m.u32("MostDetailedMip"); d.Texture3D.MipLevels = m.u32("MipLevels"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURECUBE: { auto m = mem("TextureCube"); d.TextureCube.MostDetailedMip = m.u32("MostDetailedMip"); d.TextureCube.MipLevels = m.u32("MipLevels"); m.done(); break; }
                    case D3D11_SRV_DIMENSION_TEXTURECUBEARRAY: { auto m = mem("TextureCubeArray"); d.TextureCubeArray.MostDetailedMip = m.u32("MostDetailedMip"); d.TextureCubeArray.MipLevels = m.u32("MipLevels"); d.TextureCubeArray.First2DArrayFace = m.u32("First2DArrayFace"); d.TextureCubeArray.NumCubes = m.u32("NumCubes"); m.done(); break; }
                    default: throw Fail(where + ": unsupported SRV dimension");
                    }
                    df.done();
                }
                check(dev->CreateShaderResourceView(R.res.get(), has ? &d : nullptr, srv[id].put()), where + " CreateShaderResourceView");
            } else if (kind == "rtv") {
                D3D11_RENDER_TARGET_VIEW_DESC d{}; bool has = !dj.isNull();
                if (has) {
                    Fields df(dj, where + ".desc"); d.Format = (DXGI_FORMAT)df.u32("Format"); d.ViewDimension = (D3D11_RTV_DIMENSION)df.u32("ViewDimension");
                    auto mem = [&](const char* n) { return Fields(df.req(n), where + "." + n); };
                    switch (d.ViewDimension) {
                    case D3D11_RTV_DIMENSION_BUFFER: { auto m = mem("Buffer"); d.Buffer.FirstElement = m.u32("FirstElement"); d.Buffer.NumElements = m.u32("NumElements"); m.done(); break; }
                    case D3D11_RTV_DIMENSION_TEXTURE1D: { auto m = mem("Texture1D"); d.Texture1D.MipSlice = m.u32("MipSlice"); m.done(); break; }
                    case D3D11_RTV_DIMENSION_TEXTURE1DARRAY: { auto m = mem("Texture1DArray"); d.Texture1DArray.MipSlice = m.u32("MipSlice"); d.Texture1DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture1DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_RTV_DIMENSION_TEXTURE2D: { auto m = mem("Texture2D"); d.Texture2D.MipSlice = m.u32("MipSlice"); m.done(); break; }
                    case D3D11_RTV_DIMENSION_TEXTURE2DARRAY: { auto m = mem("Texture2DArray"); d.Texture2DArray.MipSlice = m.u32("MipSlice"); d.Texture2DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture2DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_RTV_DIMENSION_TEXTURE2DMS: { auto m = mem("Texture2DMS"); m.done(); break; }
                    case D3D11_RTV_DIMENSION_TEXTURE2DMSARRAY: { auto m = mem("Texture2DMSArray"); d.Texture2DMSArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture2DMSArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_RTV_DIMENSION_TEXTURE3D: { auto m = mem("Texture3D"); d.Texture3D.MipSlice = m.u32("MipSlice"); d.Texture3D.FirstWSlice = m.u32("FirstWSlice"); d.Texture3D.WSize = m.u32("WSize"); m.done(); break; }
                    default: throw Fail(where + ": unsupported RTV dimension");
                    }
                    df.done();
                }
                check(dev->CreateRenderTargetView(R.res.get(), has ? &d : nullptr, rtv[id].put()), where + " CreateRenderTargetView");
            } else if (kind == "dsv") {
                D3D11_DEPTH_STENCIL_VIEW_DESC d{}; bool has = !dj.isNull();
                if (has) {
                    Fields df(dj, where + ".desc"); d.Format = (DXGI_FORMAT)df.u32("Format"); d.ViewDimension = (D3D11_DSV_DIMENSION)df.u32("ViewDimension"); d.Flags = df.u32("Flags");
                    auto mem = [&](const char* n) { return Fields(df.req(n), where + "." + n); };
                    switch (d.ViewDimension) {
                    case D3D11_DSV_DIMENSION_TEXTURE1D: { auto m = mem("Texture1D"); d.Texture1D.MipSlice = m.u32("MipSlice"); m.done(); break; }
                    case D3D11_DSV_DIMENSION_TEXTURE1DARRAY: { auto m = mem("Texture1DArray"); d.Texture1DArray.MipSlice = m.u32("MipSlice"); d.Texture1DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture1DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_DSV_DIMENSION_TEXTURE2D: { auto m = mem("Texture2D"); d.Texture2D.MipSlice = m.u32("MipSlice"); m.done(); break; }
                    case D3D11_DSV_DIMENSION_TEXTURE2DARRAY: { auto m = mem("Texture2DArray"); d.Texture2DArray.MipSlice = m.u32("MipSlice"); d.Texture2DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture2DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_DSV_DIMENSION_TEXTURE2DMS: { auto m = mem("Texture2DMS"); m.done(); break; }
                    case D3D11_DSV_DIMENSION_TEXTURE2DMSARRAY: { auto m = mem("Texture2DMSArray"); d.Texture2DMSArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture2DMSArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    default: throw Fail(where + ": unsupported DSV dimension");
                    }
                    df.done();
                }
                check(dev->CreateDepthStencilView(R.res.get(), has ? &d : nullptr, dsv[id].put()), where + " CreateDepthStencilView");
            } else if (kind == "uav") {
                D3D11_UNORDERED_ACCESS_VIEW_DESC d{}; bool has = !dj.isNull();
                if (has) {
                    Fields df(dj, where + ".desc"); d.Format = (DXGI_FORMAT)df.u32("Format"); d.ViewDimension = (D3D11_UAV_DIMENSION)df.u32("ViewDimension");
                    auto mem = [&](const char* n) { return Fields(df.req(n), where + "." + n); };
                    switch (d.ViewDimension) {
                    case D3D11_UAV_DIMENSION_BUFFER: { auto m = mem("Buffer"); d.Buffer.FirstElement = m.u32("FirstElement"); d.Buffer.NumElements = m.u32("NumElements"); d.Buffer.Flags = m.u32("Flags"); m.done(); break; }
                    case D3D11_UAV_DIMENSION_TEXTURE1D: { auto m = mem("Texture1D"); d.Texture1D.MipSlice = m.u32("MipSlice"); m.done(); break; }
                    case D3D11_UAV_DIMENSION_TEXTURE1DARRAY: { auto m = mem("Texture1DArray"); d.Texture1DArray.MipSlice = m.u32("MipSlice"); d.Texture1DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture1DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_UAV_DIMENSION_TEXTURE2D: { auto m = mem("Texture2D"); d.Texture2D.MipSlice = m.u32("MipSlice"); m.done(); break; }
                    case D3D11_UAV_DIMENSION_TEXTURE2DARRAY: { auto m = mem("Texture2DArray"); d.Texture2DArray.MipSlice = m.u32("MipSlice"); d.Texture2DArray.FirstArraySlice = m.u32("FirstArraySlice"); d.Texture2DArray.ArraySize = m.u32("ArraySize"); m.done(); break; }
                    case D3D11_UAV_DIMENSION_TEXTURE3D: { auto m = mem("Texture3D"); d.Texture3D.MipSlice = m.u32("MipSlice"); d.Texture3D.FirstWSlice = m.u32("FirstWSlice"); d.Texture3D.WSize = m.u32("WSize"); m.done(); break; }
                    default: throw Fail(where + ": unsupported UAV dimension");
                    }
                    df.done();
                }
                check(dev->CreateUnorderedAccessView(R.res.get(), has ? &d : nullptr, uav[id].put()), where + " CreateUnorderedAccessView");
            } else throw Fail(where + ": unknown view kind " + kind);
        }
    }

    // ----------------------------------------------- shaders, layouts, states
    void createShadersAndStates() {
        for (const J& s : prog.find("shaders")->a) {
            Fields f(s, "shader"); uint64_t id = ju(f.req("id"), "shader.id"); claim(id, "shader");
            std::string where = "shader " + std::to_string(id);
            const std::string& stage = js(f.req("stage"), where); auto b = blob(js(f.req("blob"), where), false);
            f.opt("source"); f.opt("provenance"); f.done();
            Com<ID3D11DeviceChild> out;
            if (stage == "vs") check(dev->CreateVertexShader(b.data(), b.size(), nullptr, (ID3D11VertexShader**)out.put()), where);
            else if (stage == "ps") check(dev->CreatePixelShader(b.data(), b.size(), nullptr, (ID3D11PixelShader**)out.put()), where);
            else if (stage == "gs") check(dev->CreateGeometryShader(b.data(), b.size(), nullptr, (ID3D11GeometryShader**)out.put()), where);
            else if (stage == "hs") check(dev->CreateHullShader(b.data(), b.size(), nullptr, (ID3D11HullShader**)out.put()), where);
            else if (stage == "ds") check(dev->CreateDomainShader(b.data(), b.size(), nullptr, (ID3D11DomainShader**)out.put()), where);
            else if (stage == "cs") check(dev->CreateComputeShader(b.data(), b.size(), nullptr, (ID3D11ComputeShader**)out.put()), where);
            else throw Fail(where + ": unknown stage " + stage);
            shaders[id] = std::move(out); shaderStage[id] = stage;
        }
        for (const J& l : prog.find("input_layouts")->a) {
            Fields f(l, "input_layout"); uint64_t id = ju(f.req("id"), "input_layout.id"); claim(id, "input_layout");
            std::string where = "input layout " + std::to_string(id);
            auto sig = blob(js(f.req("signature_blob"), where), false);
            const J& el = f.req("elements"); f.opt("provenance"); f.done();
            std::vector<std::string> names; names.reserve(el.a.size()); std::vector<D3D11_INPUT_ELEMENT_DESC> e(el.a.size());
            for (size_t i = 0; i < el.a.size(); ++i) {
                Fields fe(el.a[i], where + ".element");
                names.push_back(js(fe.req("SemanticName"), where));
                e[i].SemanticIndex = fe.u32("SemanticIndex"); e[i].Format = (DXGI_FORMAT)fe.u32("Format");
                e[i].InputSlot = fe.u32("InputSlot"); e[i].AlignedByteOffset = fe.u32("AlignedByteOffset");
                e[i].InputSlotClass = (D3D11_INPUT_CLASSIFICATION)fe.u32("InputSlotClass"); e[i].InstanceDataStepRate = fe.u32("InstanceDataStepRate");
                fe.done();
            }
            for (size_t i = 0; i < e.size(); ++i) e[i].SemanticName = names[i].c_str();
            check(dev->CreateInputLayout(e.data(), (UINT)e.size(), sig.data(), sig.size(), layouts[id].put()), where);
        }
        for (const J& s : prog.find("blend_states")->a) {
            Fields f(s, "blend_state"); uint64_t id = ju(f.req("id"), "blend.id"); claim(id, "blend_state");
            std::string where = "blend state " + std::to_string(id);
            Fields d(f.req("desc"), where); f.opt("provenance"); f.done();
            D3D11_BLEND_DESC b{}; b.AlphaToCoverageEnable = d.boolean("AlphaToCoverageEnable"); b.IndependentBlendEnable = d.boolean("IndependentBlendEnable");
            const J& rts = d.req("RenderTarget"); if (rts.a.size() != 8) throw Fail(where + ": 8 RenderTarget entries required");
            for (int i = 0; i < 8; ++i) {
                Fields r(rts.a[i], where + ".RenderTarget");
                auto& t = b.RenderTarget[i]; t.BlendEnable = r.boolean("BlendEnable"); t.SrcBlend = (D3D11_BLEND)r.u32("SrcBlend");
                t.DestBlend = (D3D11_BLEND)r.u32("DestBlend"); t.BlendOp = (D3D11_BLEND_OP)r.u32("BlendOp");
                t.SrcBlendAlpha = (D3D11_BLEND)r.u32("SrcBlendAlpha"); t.DestBlendAlpha = (D3D11_BLEND)r.u32("DestBlendAlpha");
                t.BlendOpAlpha = (D3D11_BLEND_OP)r.u32("BlendOpAlpha"); t.RenderTargetWriteMask = (UINT8)r.u32("RenderTargetWriteMask"); r.done();
            }
            d.done(); check(dev->CreateBlendState(&b, blends[id].put()), where);
        }
        for (const J& s : prog.find("depth_states")->a) {
            Fields f(s, "depth_state"); uint64_t id = ju(f.req("id"), "depth.id"); claim(id, "depth_state");
            std::string where = "depth state " + std::to_string(id);
            Fields d(f.req("desc"), where); f.opt("provenance"); f.done();
            D3D11_DEPTH_STENCIL_DESC x{}; x.DepthEnable = d.boolean("DepthEnable"); x.DepthWriteMask = (D3D11_DEPTH_WRITE_MASK)d.u32("DepthWriteMask");
            x.DepthFunc = (D3D11_COMPARISON_FUNC)d.u32("DepthFunc"); x.StencilEnable = d.boolean("StencilEnable");
            x.StencilReadMask = (UINT8)d.u32("StencilReadMask"); x.StencilWriteMask = (UINT8)d.u32("StencilWriteMask");
            for (auto [name, face] : {std::pair<const char*, D3D11_DEPTH_STENCILOP_DESC*>{"FrontFace", &x.FrontFace}, {"BackFace", &x.BackFace}}) {
                Fields o(d.req(name), where + "." + name);
                face->StencilFailOp = (D3D11_STENCIL_OP)o.u32("StencilFailOp"); face->StencilDepthFailOp = (D3D11_STENCIL_OP)o.u32("StencilDepthFailOp");
                face->StencilPassOp = (D3D11_STENCIL_OP)o.u32("StencilPassOp"); face->StencilFunc = (D3D11_COMPARISON_FUNC)o.u32("StencilFunc"); o.done();
            }
            d.done(); check(dev->CreateDepthStencilState(&x, depths[id].put()), where);
        }
        for (const J& s : prog.find("raster_states")->a) {
            Fields f(s, "raster_state"); uint64_t id = ju(f.req("id"), "raster.id"); claim(id, "raster_state");
            std::string where = "raster state " + std::to_string(id);
            Fields d(f.req("desc"), where); f.opt("provenance"); f.done();
            D3D11_RASTERIZER_DESC x{}; x.FillMode = (D3D11_FILL_MODE)d.u32("FillMode"); x.CullMode = (D3D11_CULL_MODE)d.u32("CullMode");
            x.FrontCounterClockwise = d.boolean("FrontCounterClockwise"); x.DepthBias = d.i32("DepthBias");
            x.DepthBiasClamp = d.f32("DepthBiasClamp"); x.SlopeScaledDepthBias = d.f32("SlopeScaledDepthBias");
            x.DepthClipEnable = d.boolean("DepthClipEnable"); x.ScissorEnable = d.boolean("ScissorEnable");
            x.MultisampleEnable = d.boolean("MultisampleEnable"); x.AntialiasedLineEnable = d.boolean("AntialiasedLineEnable");
            d.done(); check(dev->CreateRasterizerState(&x, rasters[id].put()), where);
        }
        for (const J& s : prog.find("samplers")->a) {
            Fields f(s, "sampler"); uint64_t id = ju(f.req("id"), "sampler.id"); claim(id, "sampler");
            std::string where = "sampler " + std::to_string(id);
            Fields d(f.req("desc"), where); f.opt("provenance"); f.done();
            D3D11_SAMPLER_DESC x{}; x.Filter = (D3D11_FILTER)d.u32("Filter");
            x.AddressU = (D3D11_TEXTURE_ADDRESS_MODE)d.u32("AddressU"); x.AddressV = (D3D11_TEXTURE_ADDRESS_MODE)d.u32("AddressV");
            x.AddressW = (D3D11_TEXTURE_ADDRESS_MODE)d.u32("AddressW"); x.MipLODBias = d.f32("MipLODBias");
            x.MaxAnisotropy = d.u32("MaxAnisotropy"); x.ComparisonFunc = (D3D11_COMPARISON_FUNC)d.u32("ComparisonFunc");
            const J& bc = d.req("BorderColor"); if (bc.a.size() != 4) throw Fail(where + ": BorderColor needs 4 floats");
            for (int i = 0; i < 4; ++i) x.BorderColor[i] = jf(bc.a[i], where + ".BorderColor");
            x.MinLOD = d.f32("MinLOD"); x.MaxLOD = d.f32("MaxLOD"); d.done();
            check(dev->CreateSamplerState(&x, samplers[id].put()), where);
        }
    }

    // ---------------------------------------------------------- bindings
    struct StageSet {
        ID3D11DeviceChild* shader = nullptr;
        ID3D11Buffer* cbs[14] = {}; UINT first[14] = {}; UINT num[14] = {}; bool ranged = false; int cbMax = 0;
        ID3D11ShaderResourceView* srvs[128] = {}; int srvMax = 0;
        ID3D11SamplerState* smp[16] = {}; int smpMax = 0;
        ID3D11UnorderedAccessView* uavs[64] = {}; UINT counts[64]; int uavMax = 0;
        StageSet() { for (auto& c : counts) c = 0xFFFFFFFFu; }
    };
    ID3D11Buffer* blobCB(const std::string& h, const std::string& where) {
        auto it = blobCBs.find(h);
        if (it != blobCBs.end()) return it->second.get();
        auto d = blob(h);
        if (d.empty() || d.size() % 16 || d.size() > 65536) throw Fail(where + ": CB blob size must be a positive multiple of 16 <= 65536");
        D3D11_BUFFER_DESC bd{(UINT)d.size(), D3D11_USAGE_IMMUTABLE, D3D11_BIND_CONSTANT_BUFFER, 0, 0, 0};
        D3D11_SUBRESOURCE_DATA sd{d.data(), (UINT)d.size(), (UINT)d.size()};
        check(dev->CreateBuffer(&bd, &sd, blobCBs[h].put()), where + " CB blob buffer");
        return blobCBs[h].get();
    }
    StageSet stage(const J& j, const std::string& stageName, const std::string& where, bool compute) {
        StageSet s; Fields f(j, where);
        if (const J* sh = f.opt("shader"); sh && !sh->isNull()) {
            uint64_t id = ju(*sh, where + ".shader");
            s.shader = get(shaders, id, where + ".shader");
            if (shaderStage[id] != stageName) throw Fail(where + ": shader " + std::to_string(id) + " is a " + shaderStage[id] + " shader");
        }
        for (const J& c : f.req("constant_buffers").a) {
            Fields fc(c, where + ".cb"); UINT slot = fc.u32("slot"); if (slot >= 14) throw Fail(where + ": CB slot");
            if (const J* b = fc.opt("blob")) { s.cbs[slot] = blobCB(js(*b, where), where); s.first[slot] = 0; s.num[slot] = 4096; }
            else {
                Resource& R = resource(fc.u32("buffer"), where + ".cb");
                if (R.kind != Kind::Buffer) throw Fail(where + ": CB is not a buffer");
                s.cbs[slot] = (ID3D11Buffer*)R.res.get(); s.first[slot] = fc.u32("first_constant"); s.num[slot] = fc.u32("num_constants");
                if (s.first[slot] != 0 || s.num[slot] != 4096) s.ranged = true;
            }
            fc.done(); s.cbMax = std::max(s.cbMax, (int)slot + 1);
        }
        for (const J& c : f.req("srvs").a) {
            Fields fc(c, where + ".srv"); UINT slot = fc.u32("slot"); if (slot >= 128) throw Fail(where + ": SRV slot");
            s.srvs[slot] = get(srv, fc.u32("view"), where + ".srv"); fc.done(); s.srvMax = std::max(s.srvMax, (int)slot + 1);
        }
        for (const J& c : f.req("samplers").a) {
            Fields fc(c, where + ".sampler"); UINT slot = fc.u32("slot"); if (slot >= 16) throw Fail(where + ": sampler slot");
            s.smp[slot] = get(samplers, fc.u32("sampler"), where + ".sampler"); fc.done(); s.smpMax = std::max(s.smpMax, (int)slot + 1);
        }
        if (compute) {
            for (const J& c : f.req("uavs").a) {
                Fields fc(c, where + ".uav"); UINT slot = fc.u32("slot"); if (slot >= 64) throw Fail(where + ": UAV slot");
                s.uavs[slot] = get(uav, fc.u32("view"), where + ".uav"); s.counts[slot] = fc.u32("initial_count"); fc.done();
                s.uavMax = std::max(s.uavMax, (int)slot + 1);
            }
        }
        f.done();
        return s;
    }
    template <class SetCB, class SetCB1, class SetSRV, class SetSmp>
    void applyStage(const StageSet& s, SetCB setCB, SetCB1 setCB1, SetSRV setSRV, SetSmp setSmp) {
        if (s.cbMax) { if (s.ranged) setCB1(s.cbMax, s.cbs, s.first, s.num); else setCB(s.cbMax, s.cbs); }
        if (s.srvMax) setSRV(s.srvMax, s.srvs);
        if (s.smpMax) setSmp(s.smpMax, s.smp);
    }
    template <class GetSRV>
    void verifySRVs(const StageSet& s, GetSRV getSRV, const std::string& where) {
        if (!s.srvMax) return;
        ID3D11ShaderResourceView* got[128] = {};
        getSRV(s.srvMax, got);
        for (int i = 0; i < s.srvMax; ++i) {
            bool ok = got[i] == s.srvs[i];
            if (got[i]) got[i]->Release();
            if (!ok) throw Fail(where + ": runtime refused SRV slot " + std::to_string(i) + " (input/output hazard in snapshot)");
        }
    }

    void bindDraw(const J& st, const std::string& where) {
        Fields f(st, where + ".state");
        if (!f.req("predication").isNull()) throw Fail(where + ": predication is not implemented");
        if (!f.req("stream_out").a.empty()) throw Fail(where + ": stream-out is not implemented");
        ctx->ClearState();
        // OM (outputs first)
        Fields om(f.req("om"), where + ".om");
        ID3D11RenderTargetView* rt[8] = {}; UINT nrt = 0;
        for (const J& r : om.req("render_targets").a) {
            Fields fr(r, where + ".rtv"); UINT slot = fr.u32("slot"); if (slot >= 8) throw Fail(where + ": RTV slot");
            rt[slot] = get(rtv, fr.u32("view"), where + ".rtv"); fr.done(); nrt = std::max(nrt, slot + 1);
        }
        ID3D11DepthStencilView* ds = optGet(dsv, om.req("depth_stencil").isNull() ? nullptr : &om.req("depth_stencil"), where + ".dsv");
        if (!om.req("uavs").a.empty()) throw Fail(where + ": OM UAVs are not implemented");
        ID3D11BlendState* bs = optGet(blends, &om.req("blend_state"), where + ".blend_state");
        const J& bfj = om.req("blend_factor"); if (bfj.a.size() != 4) throw Fail(where + ": blend factor");
        float bf[4]; for (int i = 0; i < 4; ++i) bf[i] = jf(bfj.a[i], where + ".blend_factor");
        UINT mask = om.u32("sample_mask");
        ID3D11DepthStencilState* dss = optGet(depths, &om.req("depth_stencil_state"), where + ".depth_stencil_state");
        UINT sref = om.u32("stencil_ref"); om.done();
        ctx->OMSetRenderTargets(nrt, rt, ds);
        ctx->OMSetBlendState(bs, bf, mask);
        ctx->OMSetDepthStencilState(dss, sref);
        {
            ID3D11RenderTargetView* g[8] = {}; ID3D11DepthStencilView* gd = nullptr;
            ctx->OMGetRenderTargets(8, g, &gd);
            bool ok = gd == ds; if (gd) gd->Release();
            for (int i = 0; i < 8; ++i) { ok = ok && g[i] == rt[i]; if (g[i]) g[i]->Release(); }
            if (!ok) throw Fail(where + ": runtime refused the output-merger binding (invalid OM combination)");
        }
        // RS
        Fields rs(f.req("rs"), where + ".rs");
        ctx->RSSetState(optGet(rasters, &rs.req("state"), where + ".rs.state"));
        D3D11_VIEWPORT vp[16]; UINT nvp = 0;
        for (const J& v : rs.req("viewports").a) {
            if (nvp >= 16) throw Fail(where + ": too many viewports");
            Fields fv(v, where + ".viewport");
            vp[nvp++] = {fv.f32("TopLeftX"), fv.f32("TopLeftY"), fv.f32("Width"), fv.f32("Height"), fv.f32("MinDepth"), fv.f32("MaxDepth")};
            fv.done();
        }
        D3D11_RECT sc[16]; UINT nsc = 0;
        for (const J& v : rs.req("scissors").a) {
            if (nsc >= 16) throw Fail(where + ": too many scissors");
            Fields fv(v, where + ".scissor");
            sc[nsc++] = {fv.i32("left"), fv.i32("top"), fv.i32("right"), fv.i32("bottom")}; fv.done();
        }
        rs.done();
        if (nvp) ctx->RSSetViewports(nvp, vp);
        if (nsc) ctx->RSSetScissorRects(nsc, sc);
        // IA
        Fields ia(f.req("ia"), where + ".ia");
        ctx->IASetInputLayout(optGet(layouts, &ia.req("input_layout"), where + ".input_layout"));
        ctx->IASetPrimitiveTopology((D3D11_PRIMITIVE_TOPOLOGY)ia.u32("topology"));
        ID3D11Buffer* vbs[32] = {}; UINT strides[32] = {}, offsets[32] = {}; UINT nvb = 0;
        for (const J& v : ia.req("vertex_buffers").a) {
            Fields fv(v, where + ".vb"); UINT slot = fv.u32("slot"); if (slot >= 32) throw Fail(where + ": VB slot");
            Resource& R = resource(fv.u32("buffer"), where + ".vb"); if (R.kind != Kind::Buffer) throw Fail(where + ": VB is not a buffer");
            vbs[slot] = (ID3D11Buffer*)R.res.get(); strides[slot] = fv.u32("stride"); offsets[slot] = fv.u32("offset"); fv.done();
            nvb = std::max(nvb, slot + 1);
        }
        if (nvb) ctx->IASetVertexBuffers(0, nvb, vbs, strides, offsets);
        const J& ib = ia.req("index_buffer");
        if (!ib.isNull()) {
            Fields fi(ib, where + ".ib"); Resource& R = resource(fi.u32("buffer"), where + ".ib");
            if (R.kind != Kind::Buffer) throw Fail(where + ": IB is not a buffer");
            ctx->IASetIndexBuffer((ID3D11Buffer*)R.res.get(), (DXGI_FORMAT)fi.u32("format"), fi.u32("offset")); fi.done();
        }
        ia.done();
        // shader stages (inputs)
        StageSet vs = stage(f.req("vs"), "vs", where + ".vs", false), hs = stage(f.req("hs"), "hs", where + ".hs", false),
                 dss2 = stage(f.req("ds"), "ds", where + ".ds", false), gs = stage(f.req("gs"), "gs", where + ".gs", false),
                 ps = stage(f.req("ps"), "ps", where + ".ps", false);
        f.done();
        if (!vs.shader) throw Fail(where + ": draw without vertex shader");
        ctx->VSSetShader((ID3D11VertexShader*)vs.shader, nullptr, 0);
        ctx->HSSetShader((ID3D11HullShader*)hs.shader, nullptr, 0);
        ctx->DSSetShader((ID3D11DomainShader*)dss2.shader, nullptr, 0);
        ctx->GSSetShader((ID3D11GeometryShader*)gs.shader, nullptr, 0);
        ctx->PSSetShader((ID3D11PixelShader*)ps.shader, nullptr, 0);
        auto* c = ctx.get(); auto* c1 = ctx1.get();
        applyStage(vs, [&](UINT n, ID3D11Buffer* const* b) { c->VSSetConstantBuffers(0, n, b); }, [&](UINT n, ID3D11Buffer* const* b, const UINT* a, const UINT* k) { c1->VSSetConstantBuffers1(0, n, b, a, k); },
                   [&](UINT n, ID3D11ShaderResourceView* const* v) { c->VSSetShaderResources(0, n, v); }, [&](UINT n, ID3D11SamplerState* const* v) { c->VSSetSamplers(0, n, v); });
        applyStage(hs, [&](UINT n, ID3D11Buffer* const* b) { c->HSSetConstantBuffers(0, n, b); }, [&](UINT n, ID3D11Buffer* const* b, const UINT* a, const UINT* k) { c1->HSSetConstantBuffers1(0, n, b, a, k); },
                   [&](UINT n, ID3D11ShaderResourceView* const* v) { c->HSSetShaderResources(0, n, v); }, [&](UINT n, ID3D11SamplerState* const* v) { c->HSSetSamplers(0, n, v); });
        applyStage(dss2, [&](UINT n, ID3D11Buffer* const* b) { c->DSSetConstantBuffers(0, n, b); }, [&](UINT n, ID3D11Buffer* const* b, const UINT* a, const UINT* k) { c1->DSSetConstantBuffers1(0, n, b, a, k); },
                   [&](UINT n, ID3D11ShaderResourceView* const* v) { c->DSSetShaderResources(0, n, v); }, [&](UINT n, ID3D11SamplerState* const* v) { c->DSSetSamplers(0, n, v); });
        applyStage(gs, [&](UINT n, ID3D11Buffer* const* b) { c->GSSetConstantBuffers(0, n, b); }, [&](UINT n, ID3D11Buffer* const* b, const UINT* a, const UINT* k) { c1->GSSetConstantBuffers1(0, n, b, a, k); },
                   [&](UINT n, ID3D11ShaderResourceView* const* v) { c->GSSetShaderResources(0, n, v); }, [&](UINT n, ID3D11SamplerState* const* v) { c->GSSetSamplers(0, n, v); });
        applyStage(ps, [&](UINT n, ID3D11Buffer* const* b) { c->PSSetConstantBuffers(0, n, b); }, [&](UINT n, ID3D11Buffer* const* b, const UINT* a, const UINT* k) { c1->PSSetConstantBuffers1(0, n, b, a, k); },
                   [&](UINT n, ID3D11ShaderResourceView* const* v) { c->PSSetShaderResources(0, n, v); }, [&](UINT n, ID3D11SamplerState* const* v) { c->PSSetSamplers(0, n, v); });
        verifySRVs(vs, [&](UINT n, ID3D11ShaderResourceView** v) { c->VSGetShaderResources(0, n, v); }, where + ".vs");
        verifySRVs(hs, [&](UINT n, ID3D11ShaderResourceView** v) { c->HSGetShaderResources(0, n, v); }, where + ".hs");
        verifySRVs(dss2, [&](UINT n, ID3D11ShaderResourceView** v) { c->DSGetShaderResources(0, n, v); }, where + ".ds");
        verifySRVs(gs, [&](UINT n, ID3D11ShaderResourceView** v) { c->GSGetShaderResources(0, n, v); }, where + ".gs");
        verifySRVs(ps, [&](UINT n, ID3D11ShaderResourceView** v) { c->PSGetShaderResources(0, n, v); }, where + ".ps");
    }
    void bindDispatch(const J& st, const std::string& where) {
        Fields f(st, where + ".state");
        StageSet cs = stage(f.req("cs"), "cs", where + ".cs", true); f.done();
        if (!cs.shader) throw Fail(where + ": dispatch without compute shader");
        ctx->ClearState();
        auto* c = ctx.get(); auto* c1 = ctx1.get();
        if (cs.uavMax) ctx->CSSetUnorderedAccessViews(0, cs.uavMax, cs.uavs, cs.counts);
        ctx->CSSetShader((ID3D11ComputeShader*)cs.shader, nullptr, 0);
        applyStage(cs, [&](UINT n, ID3D11Buffer* const* b) { c->CSSetConstantBuffers(0, n, b); }, [&](UINT n, ID3D11Buffer* const* b, const UINT* a, const UINT* k) { c1->CSSetConstantBuffers1(0, n, b, a, k); },
                   [&](UINT n, ID3D11ShaderResourceView* const* v) { c->CSSetShaderResources(0, n, v); }, [&](UINT n, ID3D11SamplerState* const* v) { c->CSSetSamplers(0, n, v); });
        if (cs.uavMax) {
            ID3D11UnorderedAccessView* g[64] = {}; ctx->CSGetUnorderedAccessViews(0, cs.uavMax, g);
            for (int i = 0; i < cs.uavMax; ++i) { bool ok = g[i] == cs.uavs[i]; if (g[i]) g[i]->Release(); if (!ok) throw Fail(where + ": runtime refused CS UAV slot " + std::to_string(i)); }
        }
        verifySRVs(cs, [&](UINT n, ID3D11ShaderResourceView** v) { c->CSGetShaderResources(0, n, v); }, where + ".cs");
    }

    // ------------------------------------------------------- data commands
    static bool boxOf(const J* b, D3D11_BOX& out, const std::string& where) {
        if (!b || b->isNull()) return false;
        Fields f(*b, where + ".box");
        out.left = f.u32("left"); out.top = f.u32("top"); out.front = f.u32("front");
        out.right = f.u32("right"); out.bottom = f.u32("bottom"); out.back = f.u32("back"); f.done();
        return true;
    }
    // RenderDoc v1.46 Serialise_Unmap replay + MapIntercept::Init/CopyToD3D (d3d11_context_wrap.cpp).
    void mapWrite(Resource& R, UINT sub, UINT mapType, UINT mapFlags, uint64_t start, uint64_t end, const std::vector<char>& data,
                  const std::string& where) {
        if (start >= end) throw Fail(where + ": empty written range (replay performs no Map)");
        if (data.size() != end - start) throw Fail(where + ": written data length differs from range");
        if (mapType == D3D11_MAP_WRITE_NO_OVERWRITE) {
            if (R.kind != Kind::Buffer) throw Fail(where + ": NO_OVERWRITE on a texture is not implemented");
            D3D11_BUFFER_DESC bd{(UINT)(end - start), D3D11_USAGE_IMMUTABLE, D3D11_BIND_VERTEX_BUFFER, 0, 0, 0};
            D3D11_SUBRESOURCE_DATA sd{data.data(), bd.ByteWidth, bd.ByteWidth};
            Com<ID3D11Buffer> tmp; check(dev->CreateBuffer(&bd, &sd, tmp.put()), where + " NO_OVERWRITE temp buffer");
            ctx->CopySubresourceRegion(R.res.get(), sub, (UINT)start, 0, 0, tmp.get(), 0, nullptr);
            return;
        }
        if (mapType != D3D11_MAP_WRITE && mapType != D3D11_MAP_WRITE_DISCARD) throw Fail(where + ": map type " + std::to_string(mapType) + " not implemented");
        D3D11_MAPPED_SUBRESOURCE m{};
        check(ctx->Map(R.res.get(), sub, (D3D11_MAP)mapType, mapFlags & ~(UINT)D3D11_MAP_FLAG_DO_NOT_WAIT, &m), where + " Map");
        // MapIntercept: app (tightly packed) pitches, rows and slices of this subresource
        uint64_t appRow, appDepth; int numRows = 1, numSlices = 1;
        uint64_t d3dRow = m.RowPitch, d3dDepth = m.DepthPitch;
        if (R.kind == Kind::Buffer) {
            appRow = appDepth = R.bd.ByteWidth;
            if (d3dRow == 0) d3dRow = R.bd.ByteWidth;
            if (d3dDepth == 0) d3dDepth = R.bd.ByteWidth;
        } else {
            UINT mip = sub % R.mips; uint32_t fmt = R.format();
            if (R.kind == Kind::Tex2D || R.kind == Kind::Tex3D) numRows = (int)R.height();
            if (R.kind == Kind::Tex3D) numSlices = (int)R.depth();
            numRows = std::max(1, numRows >> mip); numSlices = std::max(1, numSlices >> mip);
            if (isBlockFormat(fmt)) numRows = (int)((numRows + 3) / 4 * 4) / 4;
            appRow = rowPitch(R.width(), fmt, mip);
            appDepth = byteSize(R.width(), R.height(), 1, fmt, mip);
            if (R.kind == Kind::Tex2D && d3dDepth == 0) d3dDepth = appDepth;
            if (R.kind == Kind::Tex1D) { if (d3dDepth == 0) d3dDepth = appRow; if (d3dDepth == 0) d3dDepth = appDepth; }
        }
        const char* app = data.data();
        const char* appEnd = app + (end - start);
        char* sliceDst = (char*)m.pData + start; const char* sliceSrc = app;
        for (int s = 0; s < numSlices; ++s) {
            const char* rowSrc = sliceSrc; char* rowDst = sliceDst;
            for (int r = 0; r < numRows; ++r) {
                size_t len = (size_t)appRow;
                if (rowSrc + len > appEnd) len = (size_t)(appEnd - rowSrc);
                memcpy(rowDst, rowSrc, len);
                rowSrc += appRow; rowDst += d3dRow;
                if (rowSrc > appEnd) { ctx->Unmap(R.res.get(), sub); return; }
            }
            sliceSrc += appDepth; sliceDst += d3dDepth;
        }
        ctx->Unmap(R.res.get(), sub);
    }

    // ---------------------------------------------------------- readbacks
    void readback(const J& rb, size_t cmdIndex) {
        Fields f(rb, "readback");
        std::string name = js(f.req("name"), "readback.name");
        for (char ch : name) if (!(isalnum((unsigned char)ch) || ch == '_' || ch == '-' || ch == '.')) throw Fail("readback name " + name);
        f.req("after_command"); Resource& R = resource(f.u32("resource"), "readback " + name);
        UINT sub = f.u32("subresource"); std::string orientation = js(f.req("orientation"), "readback.orientation");
        if (orientation != "unity_texture_rows_bottom_up" && orientation != "d3d_rows_top_down") throw Fail("readback orientation " + orientation);
        const J* bo = f.opt("byte_offset"); const J* bl = f.opt("byte_length"); f.opt("provenance"); f.done();
        if (sub >= R.subresources()) throw Fail("readback " + name + ": subresource out of range");
        std::vector<char> out; std::ostringstream desc;
        auto t0 = std::chrono::steady_clock::now();
        if (R.kind == Kind::Buffer) {
            uint64_t off = bo ? ju(*bo, "byte_offset") : 0, len = bl ? ju(*bl, "byte_length") : 0;
            if (!len) len = R.bd.ByteWidth - off;
            if (off + len > R.bd.ByteWidth) throw Fail("readback " + name + ": buffer range outside buffer");
            D3D11_BUFFER_DESC bd{R.bd.ByteWidth, D3D11_USAGE_STAGING, 0, D3D11_CPU_ACCESS_READ, 0, 0};
            Com<ID3D11Buffer> st; check(dev->CreateBuffer(&bd, nullptr, st.put()), "readback staging buffer");
            ctx->CopyResource(st.get(), R.res.get());
            D3D11_MAPPED_SUBRESOURCE m{}; check(ctx->Map(st.get(), 0, D3D11_MAP_READ, 0, &m), "readback Map");
            out.assign((char*)m.pData + off, (char*)m.pData + off + len); ctx->Unmap(st.get(), 0);
            desc << "\"kind\":\"buffer\",\"byte_offset\":" << off << ",\"byte_length\":" << len << ",\"buffer_bytes\":" << R.bd.ByteWidth;
        } else {
            if (R.kind == Kind::Tex2D && R.t2.SampleDesc.Count > 1) throw Fail("readback " + name + ": MSAA readback is not implemented");
            Com<ID3D11Resource> st;
            if (R.kind == Kind::Tex1D) { auto d = R.t1; d.BindFlags = 0; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0; d.Usage = D3D11_USAGE_STAGING; Com<ID3D11Texture1D> t; check(dev->CreateTexture1D(&d, nullptr, t.put()), "readback staging"); st.p = t.p; t.p = nullptr; }
            else if (R.kind == Kind::Tex2D) { auto d = R.t2; d.BindFlags = 0; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0; d.Usage = D3D11_USAGE_STAGING; Com<ID3D11Texture2D> t; check(dev->CreateTexture2D(&d, nullptr, t.put()), "readback staging"); st.p = t.p; t.p = nullptr; }
            else { auto d = R.t3; d.BindFlags = 0; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0; d.Usage = D3D11_USAGE_STAGING; Com<ID3D11Texture3D> t; check(dev->CreateTexture3D(&d, nullptr, t.put()), "readback staging"); st.p = t.p; t.p = nullptr; }
            ctx->CopyResource(st.get(), R.res.get());
            UINT mip = sub % R.mips; uint32_t fmt = R.format();
            int numRows = 1, numSlices = 1;
            if (R.kind != Kind::Tex1D) numRows = (int)R.height();
            if (R.kind == Kind::Tex3D) numSlices = (int)R.depth();
            numRows = std::max(1, numRows >> mip); numSlices = std::max(1, numSlices >> mip);
            if (isBlockFormat(fmt)) numRows = (numRows + 3) / 4;
            uint64_t appRow = rowPitch(R.width(), fmt, mip), appDepth = byteSize(R.width(), R.height(), 1, fmt, mip);
            uint64_t total = byteSize(R.width(), R.height(), R.depth(), fmt, mip);
            out.resize((size_t)total);
            D3D11_MAPPED_SUBRESOURCE m{}; check(ctx->Map(st.get(), sub, D3D11_MAP_READ, 0, &m), "readback Map");
            const char* sliceSrc = (const char*)m.pData; char* sliceDst = out.data();
            for (int s = 0; s < numSlices; ++s) {
                const char* rs = sliceSrc; char* rd = sliceDst;
                for (int r = 0; r < numRows; ++r) { memcpy(rd, rs, (size_t)appRow); rs += m.RowPitch; rd += appRow; }
                sliceSrc += m.DepthPitch; sliceDst += appDepth;
            }
            ctx->Unmap(st.get(), sub);
            uint32_t w = std::max(1u, R.width() >> mip), h = std::max(1u, R.height() >> mip), dd = std::max(1u, R.depth() >> mip);
            desc << "\"kind\":\"" << (R.kind == Kind::Tex1D ? "tex1d" : R.kind == Kind::Tex2D ? "tex2d" : "tex3d") << "\",\"format\":" << fmt
                 << ",\"width\":" << w << ",\"height\":" << h << ",\"depth\":" << dd << ",\"mip\":" << mip << ",\"array_slice\":" << sub / R.mips
                 << ",\"row_bytes\":" << appRow << ",\"rows\":" << numRows << ",\"block_format\":" << (isBlockFormat(fmt) ? "true" : "false");
            if (orientation == "unity_texture_rows_bottom_up" && R.kind == Kind::Tex2D && !isBlockFormat(fmt) && blockLayout(fmt).w == 1) {
                std::vector<char> td(out.size());
                for (int r = 0; r < numRows; ++r) memcpy(td.data() + (size_t)r * appRow, out.data() + (size_t)(numRows - 1 - r) * appRow, (size_t)appRow);
                if (!hashOnly) writeFile(out_dir() / (name + ".topdown.bin"), td.data(), td.size());
                desc << ",\"topdown\":" << (hashOnly ? std::string("null") : "\"" + name + ".topdown.bin\"")
                     << ",\"topdown_sha256\":\"" << sha256(td.data(), td.size()) << "\"";
            } else if (orientation == "unity_texture_rows_bottom_up") desc << ",\"topdown\":null,\"topdown_note\":\"row reversal applies to non-block 2D textures only\"";
        }
        if (!hashOnly) writeFile(out_dir() / (name + ".raw.bin"), out.data(), out.size());
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        readbackJson << (firstReadback ? "" : ",") << "\n{\"name\":\"" << name << "\",\"after_command\":" << cmdIndex << ",\"resource\":" << R.id
                     << ",\"subresource\":" << sub << ",\"orientation\":\"" << orientation << "\",\"raw\":" << (hashOnly ? std::string("null") : "\"" + name + ".raw.bin\"") << ",\"bytes\":" << out.size()
                     << ",\"sha256\":\"" << sha256(out.data(), out.size()) << "\"," << desc.str() << ",\"readback_ms\":" << ms << "}";
        firstReadback = false;
    }
    fs::path out_dir() const { return out; }

    // ------------------------------------------------------------- run
    void setup() {
        auto T = [&](const char* k, std::chrono::steady_clock::time_point t0) {
            timing[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
        auto t0 = std::chrono::steady_clock::now(); createResources(); T("create_resources_ms", t0);
        t0 = std::chrono::steady_clock::now(); createViews(); createShadersAndStates(); T("create_views_shaders_states_ms", t0);
    }
    void run() { setup(); execute(); report(); }
    void execute() {
        auto T = [&](const char* k, std::chrono::steady_clock::time_point t0) {
            timing[k] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); };
        auto t0 = std::chrono::steady_clock::now();
        const J& cmds = *prog.find("commands");
        std::multimap<uint64_t, const J*> rbs;
        for (const J& rb : (readbackOverride ? *readbackOverride : *prog.find("readbacks")).a)
            rbs.emplace(ju(*rb.find("after_command"), "readback.after_command"), &rb);
        for (auto& kv : rbs) if (kv.first >= cmds.a.size()) throw Fail("readback after_command beyond the command list");
        std::map<std::string, uint64_t> opCount;
        t0 = std::chrono::steady_clock::now();
        double readMs = 0;
        for (size_t i = 0; i < cmds.a.size(); ++i) {
            if (stopAfter >= 0 && i > (size_t)stopAfter) break;
            const J& c = cmds.a[i];
            Fields f(c, "command");
            std::string op = js(f.req("op"), "command.op");
            std::string where = "command " + std::to_string(i) + " (" + op;
            if (const J* p = f.opt("provenance")) if (const J* e = p->find("event")) where += ", event " + e->s;
            where += ")";
            ++opCount[op];
            if (op == "clear_rtv") {
                auto* v = get(rtv, f.u32("view"), where); const J& col = f.req("color"); if (col.a.size() != 4) throw Fail(where + ": color");
                float cc[4]; for (int k = 0; k < 4; ++k) cc[k] = jf(col.a[k], where); ctx->ClearRenderTargetView(v, cc);
            } else if (op == "clear_dsv") {
                auto* v = get(dsv, f.u32("view"), where); ctx->ClearDepthStencilView(v, f.u32("flags"), f.f32("depth"), (UINT8)f.u32("stencil"));
            } else if (op == "clear_uav_float") {
                auto* v = get(uav, f.u32("view"), where); const J& vals = f.req("values"); if (vals.a.size() != 4) throw Fail(where);
                float x[4]; for (int k = 0; k < 4; ++k) x[k] = jf(vals.a[k], where); ctx->ClearUnorderedAccessViewFloat(v, x);
            } else if (op == "clear_uav_uint") {
                auto* v = get(uav, f.u32("view"), where); const J& vals = f.req("values"); if (vals.a.size() != 4) throw Fail(where);
                UINT x[4]; for (int k = 0; k < 4; ++k) x[k] = (UINT)ju(vals.a[k], where); ctx->ClearUnorderedAccessViewUint(v, x);
            } else if (op == "update_subresource") {
                Resource& R = resource(f.u32("resource"), where); UINT sub = f.u32("subresource");
                D3D11_BOX box; bool hasBox = boxOf(f.opt("box"), box, where);
                auto d = blob(js(f.req("blob"), where)); UINT rp = f.u32("row_pitch"), dp = f.u32("depth_pitch");
                const J& cf = f.req("copy_flags");
                if (sub >= R.subresources()) throw Fail(where + ": subresource");
                if (cf.isNull()) ctx->UpdateSubresource(R.res.get(), sub, hasBox ? &box : nullptr, d.data(), rp, dp);
                else ctx1->UpdateSubresource1(R.res.get(), sub, hasBox ? &box : nullptr, d.data(), rp, dp, (UINT)ju(cf, where));
            } else if (op == "map_write") {
                Resource& R = resource(f.u32("resource"), where); UINT sub = f.u32("subresource");
                if (sub >= R.subresources()) throw Fail(where + ": subresource");
                auto ov = cmdBlobOverride.find(i);
                if (ov != cmdBlobOverride.end()) {
                    js(f.req("blob"), where);
                    mapWrite(R, sub, f.u32("map_type"), f.u32("map_flags"), ju(f.req("start"), where), ju(f.req("end"), where),
                             ov->second, where);
                } else
                    mapWrite(R, sub, f.u32("map_type"), f.u32("map_flags"), ju(f.req("start"), where), ju(f.req("end"), where),
                             blob(js(f.req("blob"), where)), where);
            } else if (op == "copy_resource") {
                ctx->CopyResource(resource(f.u32("dst"), where).res.get(), resource(f.u32("src"), where).res.get());
            } else if (op == "copy_subresource_region") {
                Resource& D = resource(f.u32("dst"), where); Resource& S = resource(f.u32("src"), where);
                D3D11_BOX box; bool hasBox = boxOf(f.opt("src_box"), box, where);
                UINT ds = f.u32("dst_subresource"), ss = f.u32("src_subresource");
                if (ds >= D.subresources() || ss >= S.subresources()) throw Fail(where + ": subresource");
                const J& cf = f.req("copy_flags");
                if (cf.isNull()) ctx->CopySubresourceRegion(D.res.get(), ds, f.u32("dst_x"), f.u32("dst_y"), f.u32("dst_z"), S.res.get(), ss, hasBox ? &box : nullptr);
                else ctx1->CopySubresourceRegion1(D.res.get(), ds, f.u32("dst_x"), f.u32("dst_y"), f.u32("dst_z"), S.res.get(), ss, hasBox ? &box : nullptr, (UINT)ju(cf, where));
            } else if (op == "resolve") {
                ctx->ResolveSubresource(resource(f.u32("dst"), where).res.get(), f.u32("dst_subresource"),
                                        resource(f.u32("src"), where).res.get(), f.u32("src_subresource"), (DXGI_FORMAT)f.u32("format"));
            } else if (op == "generate_mips") {
                ctx->GenerateMips(get(srv, f.u32("view"), where));
            } else if (op == "draw" || op == "draw_indexed" || op == "draw_instanced" || op == "draw_indexed_instanced") {
                bindDraw(f.req("state"), where);
                Fields a(f.req("args"), where + ".args");
                if (op == "draw") ctx->Draw(a.u32("VertexCount"), a.u32("StartVertexLocation"));
                else if (op == "draw_indexed") ctx->DrawIndexed(a.u32("IndexCount"), a.u32("StartIndexLocation"), a.i32("BaseVertexLocation"));
                else if (op == "draw_instanced") ctx->DrawInstanced(a.u32("VertexCountPerInstance"), a.u32("InstanceCount"), a.u32("StartVertexLocation"), a.u32("StartInstanceLocation"));
                else ctx->DrawIndexedInstanced(a.u32("IndexCountPerInstance"), a.u32("InstanceCount"), a.u32("StartIndexLocation"), a.i32("BaseVertexLocation"), a.u32("StartInstanceLocation"));
                a.done();
            } else if (op == "dispatch") {
                bindDispatch(f.req("state"), where);
                Fields a(f.req("args"), where + ".args");
                ctx->Dispatch(a.u32("ThreadGroupCountX"), a.u32("ThreadGroupCountY"), a.u32("ThreadGroupCountZ")); a.done();
            } else throw Fail(where + ": unsupported op");
            f.done();
            auto range = rbs.equal_range(i);
            if (range.first != range.second) {
                auto r0 = std::chrono::steady_clock::now();
                for (auto it = range.first; it != range.second; ++it) readback(*it->second, i);
                readMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();
            }
        }
        ctx->ClearState(); ctx->Flush();
        T("execute_commands_ms", t0); timing["readbacks_ms_included"] = readMs;
        HRESULT removed = dev->GetDeviceRemovedReason();
        if (FAILED(removed)) throw Fail("Device removed during execution HRESULT " + hex32((uint32_t)removed));
        opCounts = opCount;
    }
    std::map<std::string, uint64_t> opCounts;
    std::string report() {
        const J& cmds = *prog.find("commands");
        auto& opCount = opCounts;
        std::ostringstream o;
        o << "{\"schema\":\"hs2_frame_execution/1\",\"executor\":\"" << EXECUTOR_VERSION << "\",\"status\":\"pass\",\"adapter\":" << adapterJson
          << ",\"program_json_sha256\":\"" << programSha << "\",\"hash_only\":" << (hashOnly ? "true" : "false")
          << ",\"commands\":" << cmds.a.size() << ",\"ops\":{";
        bool first = true; for (auto& kv : opCount) { o << (first ? "" : ",") << "\"" << kv.first << "\":" << kv.second; first = false; }
        o << "},\"blobs_verified\":" << blobsVerified << ",\"blob_bytes_verified\":" << blobBytesVerified << ",\"timing\":{";
        first = true; for (auto& kv : timing) { o << (first ? "" : ",") << "\"" << kv.first << "\":" << kv.second; first = false; }
        o << "},\"readbacks\":[" << readbackJson.str() << "\n]}";
        std::string s = o.str(); writeFile(out / "execution.json", s.data(), s.size());
        if (!serving) std::cout << s << std::endl;
        return s;
    }
    std::string programSha;
    bool serving = false;

    // ------------------------------------------------------------- serve
    void addWritten(std::set<uint64_t>& w, uint64_t viewId) {
        auto it = viewResource.find(viewId); if (it != viewResource.end()) w.insert(it->second);
    }
    std::set<uint64_t> writtenResources() {
        std::set<uint64_t> w;
        for (const J& c : prog.find("commands")->a) {
            const std::string& op = c.find("op")->s;
            if (op == "clear_rtv" || op == "clear_dsv" || op == "clear_uav_float" || op == "clear_uav_uint" || op == "generate_mips")
                addWritten(w, ju(*c.find("view"), "view"));
            else if (op == "update_subresource" || op == "map_write") w.insert(ju(*c.find("resource"), "resource"));
            else if (op == "copy_resource" || op == "copy_subresource_region" || op == "resolve") w.insert(ju(*c.find("dst"), "dst"));
            else if (op == "dispatch") {
                for (const J& u : c.find("state")->find("cs")->find("uavs")->a) addWritten(w, ju(*u.find("view"), "uav"));
            } else if (op.rfind("draw", 0) == 0) {
                const J* om = c.find("state")->find("om");
                for (const J& r : om->find("render_targets")->a) addWritten(w, ju(*r.find("view"), "rtv"));
                const J* ds = om->find("depth_stencil"); if (ds && !ds->isNull()) addWritten(w, ju(*ds, "dsv"));
            }
        }
        return w;
    }
    const J* resourceJson(uint64_t id) {
        for (const J& r : prog.find("resources")->a) if (ju(*r.find("id"), "id") == id) return &r;
        return nullptr;
    }
    void keepPristine(uint64_t id) {
        if (restoreSet.count(id)) return;
        Resource& R = resource(id, "pristine");
        if (R.usage() == D3D11_USAGE_IMMUTABLE) throw Fail("resource " + std::to_string(id) + " is IMMUTABLE and cannot be restored or patched");
        if (R.usage() == D3D11_USAGE_DYNAMIC) {
            const J* rj = resourceJson(id);
            std::vector<std::vector<char>> subs(R.subresources()); std::vector<uint32_t> pitch(R.subresources(), 0);
            const J* init = rj ? rj->find("initial") : nullptr;
            if (init && !init->isNull()) {
                for (const J& e : init->a) { uint32_t sidx = (uint32_t)ju(*e.find("subresource"), "sub");
                    subs[sidx] = blob(e.find("blob")->s, false); pitch[sidx] = (uint32_t)ju(*e.find("row_pitch"), "rp"); }
            } else {
                for (uint32_t k = 0; k < R.subresources(); ++k) {
                    uint64_t n = R.kind == Kind::Buffer ? R.bd.ByteWidth : byteSize(R.width(), R.height(), R.depth(), R.format(), k % R.mips);
                    subs[k].assign((size_t)n, 0); pitch[k] = R.kind == Kind::Buffer ? R.bd.ByteWidth : (uint32_t)rowPitch(R.width(), R.format(), k % R.mips);
                }
            }
            pristineCpu[id] = std::move(subs); pristinePitch[id] = std::move(pitch);
        } else {
            Com<ID3D11Resource> dup;
            if (R.kind == Kind::Buffer) { Com<ID3D11Buffer> b; check(dev->CreateBuffer(&R.bd, nullptr, b.put()), "pristine buffer"); dup.p = b.p; b.p = nullptr; }
            else if (R.kind == Kind::Tex1D) { Com<ID3D11Texture1D> t; check(dev->CreateTexture1D(&R.t1, nullptr, t.put()), "pristine tex1d"); dup.p = t.p; t.p = nullptr; }
            else if (R.kind == Kind::Tex2D) { Com<ID3D11Texture2D> t; check(dev->CreateTexture2D(&R.t2, nullptr, t.put()), "pristine tex2d"); dup.p = t.p; t.p = nullptr; }
            else { Com<ID3D11Texture3D> t; check(dev->CreateTexture3D(&R.t3, nullptr, t.put()), "pristine tex3d"); dup.p = t.p; t.p = nullptr; }
            ctx->CopyResource(dup.get(), R.res.get());
            pristine[id] = std::move(dup);
        }
        restoreSet.insert(id);
    }
    void writeDynamic(Resource& R, uint32_t sub, const std::vector<char>& data, uint32_t pitch) {
        D3D11_MAPPED_SUBRESOURCE m{}; check(ctx->Map(R.res.get(), sub, D3D11_MAP_WRITE_DISCARD, 0, &m), "restore Map");
        if (R.kind == Kind::Buffer) memcpy(m.pData, data.data(), std::min<size_t>(data.size(), R.bd.ByteWidth));
        else {
            uint32_t mip = sub % R.mips; int rows = (int)std::max(1u, R.height() >> mip); if (isBlockFormat(R.format())) rows = (rows + 3) / 4;
            uint64_t row = rowPitch(R.width(), R.format(), mip);
            for (int r = 0; r < rows; ++r) memcpy((char*)m.pData + (size_t)r * m.RowPitch, data.data() + (size_t)r * pitch, (size_t)row);
        }
        ctx->Unmap(R.res.get(), sub);
    }
    void restoreAll() {
        for (uint64_t id : restoreSet) {
            Resource& R = resource(id, "restore");
            auto it = pristine.find(id);
            if (it != pristine.end()) { ctx->CopyResource(R.res.get(), it->second.get()); continue; }
            auto& subs = pristineCpu[id]; auto& pitch = pristinePitch[id];
            for (uint32_t k = 0; k < subs.size(); ++k) writeDynamic(R, k, subs[k], pitch[k]);
        }
    }
    void serve() {
        setup();
        for (uint64_t id : writtenResources()) keepPristine(id);
        ctx->Flush();
        std::cout << "{\"status\":\"ready\",\"restore_set\":" << restoreSet.size() << ",\"timing\":{\"create_resources_ms\":"
                  << timing["create_resources_ms"] << ",\"create_views_shaders_states_ms\":" << timing["create_views_shaders_states_ms"]
                  << "}}" << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line.empty()) continue;
            try {
                auto tr = std::chrono::steady_clock::now();
                Parser rp{line.data(), line.data() + line.size()}; J req = rp.value();
                out = fs::absolute(fs::path(req.find("out")->s));
                if (fs::exists(out)) throw Fail("Fresh output directory required: " + out.string());
                fs::create_directories(out);
                const J* hj = req.find("hash_only"); hashOnly = hj && hj->t == J::Bool && hj->b;
                restoreAll();
                cmdBlobOverride.clear();
                if (const J* mw = req.find("map_write_blobs"))
                    for (auto& kv : mw->o) cmdBlobOverride[(size_t)std::stoull(kv.first)] = readFile(fs::path(kv.second.s));
                if (const J* rd = req.find("resource_data"))
                    for (auto& kv : rd->o) {
                        uint64_t id = std::stoull(kv.first); keepPristine(id); Resource& R = resource(id, "resource_data");
                        for (const J& e : kv.second.a) {
                            uint32_t sidx = (uint32_t)ju(*e.find("subresource"), "sub"); auto data = readFile(fs::path(e.find("file")->s));
                            uint32_t rpitch = (uint32_t)ju(*e.find("row_pitch"), "row_pitch"); uint32_t spitch = (uint32_t)ju(*e.find("slice_pitch"), "slice_pitch");
                            if (R.kind == Kind::Buffer && data.size() != R.bd.ByteWidth) throw Fail("resource_data size differs from ByteWidth");
                            if (R.usage() == D3D11_USAGE_DYNAMIC) writeDynamic(R, sidx, data, rpitch);
                            else ctx->UpdateSubresource(R.res.get(), sidx, nullptr, data.data(), rpitch, spitch);
                        }
                    }
                readbackOverride = req.find("readbacks");
                const J* sa = req.find("stop_after"); stopAfter = sa ? (long long)ju(*sa, "stop_after") : -1;
                if (readbackOverride) for (const J& rb : readbackOverride->a)
                    if (stopAfter >= 0 && (long long)ju(*rb.find("after_command"), "after_command") > stopAfter)
                        throw Fail("readback after stop_after");
                readbackJson.str(""); readbackJson.clear(); firstReadback = true;
                timing["restore_and_patch_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
                execute();
                std::string rep = report();
                timing["request_total_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tr).count();
                std::cout << "{\"status\":\"pass\",\"out\":\"" << out.generic_string() << "\",\"request_total_ms\":" << timing["request_total_ms"]
                          << ",\"restore_and_patch_ms\":" << timing["restore_and_patch_ms"] << ",\"execute_commands_ms\":" << timing["execute_commands_ms"] << "}" << std::endl;
            } catch (const std::exception& e) {
                std::string msg = e.what(), esc; for (char c : msg) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
                std::cout << "{\"status\":\"fail\",\"error\":\"" << esc << "\"}" << std::endl;
            }
        }
    }
};

int main(int argc, char** argv) {
    fs::path outDir;
    try {
        if (argc < 3) throw Fail("Usage: frame_executor PROGRAM_DIR OUT_DIR [--warp] [--adapter-index N] [--debug-layer] [--hash-only]");
        Executor ex; ex.root = fs::absolute(argv[1]); outDir = ex.out = fs::absolute(argv[2]);
        bool warp = false, debugLayer = false; int adapterIndex = -1;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--warp") warp = true;
            else if (a == "--debug-layer") debugLayer = true;
            else if (a == "--hash-only") ex.hashOnly = true;
            else if (a == "--serve") ex.serving = true;
            else if (a == "--adapter-index" && i + 1 < argc) adapterIndex = std::stoi(argv[++i]);
            else throw Fail("Unknown argument " + a);
        }
        if (warp && adapterIndex >= 0) throw Fail("--warp and --adapter-index are exclusive");
        if (fs::exists(ex.out)) throw Fail("Fresh output directory required: " + ex.out.string());
        auto t0 = std::chrono::steady_clock::now();
        auto text = readFile(ex.root / "program.json");
        ex.programSha = sha256(text.data(), text.size());
        Parser p{text.data(), text.data() + text.size()}; ex.prog = p.value(); p.ws();
        if (p.p != p.e) throw Fail("Trailing data after program.json");
        const J* schema = ex.prog.find("schema");
        if (!schema || schema->t != J::Str || schema->s != "hs2_frame_program/1") throw Fail("program.json schema must be hs2_frame_program/1");
        for (const char* k : {"provenance", "resources", "views", "shaders", "input_layouts", "blend_states", "depth_states",
                              "raster_states", "samplers", "commands", "readbacks"})
            if (!ex.prog.find(k)) throw Fail(std::string("program.json misses ") + k);
        for (auto& kv : ex.prog.o) {
            static const std::set<std::string> known = {"schema", "provenance", "resources", "views", "shaders", "input_layouts", "blend_states",
                                                        "depth_states", "raster_states", "samplers", "commands", "readbacks"};
            if (!known.count(kv.first)) throw Fail("program.json has unsupported top-level key " + kv.first);
        }
        ex.timing["parse_program_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        t0 = std::chrono::steady_clock::now();
        ex.createDevice(warp, adapterIndex, debugLayer);
        ex.timing["create_device_ms"] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fs::create_directories(ex.out);
        if (ex.serving) { ex.serve(); return 0; }
        ex.run();
        return 0;
    } catch (const std::exception& e) {
        std::string msg = e.what(); std::string esc;
        for (char c : msg) { if (c == '"' || c == '\\') esc += '\\'; esc += c; }
        std::cerr << "{\"status\":\"fail\",\"error\":\"" << esc << "\"}" << std::endl;
        if (!outDir.empty() && fs::exists(outDir)) {
            std::string s = "{\"schema\":\"hs2_frame_execution/1\",\"status\":\"fail\",\"error\":\"" + esc + "\"}";
            try { writeFile(outDir / "execution_failure.json", s.data(), s.size()); } catch (...) {}
        }
        return 2;
    }
}
