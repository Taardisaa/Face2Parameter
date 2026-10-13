// CPU protocol fixture ONLY. This assembly must never be deployed to HS2.
using System;
using System.Collections.Generic;
using System.IO;
using System.Security.Cryptography;
using Newtonsoft.Json;
namespace HS2Mod.McpBridge {
    internal sealed class MakerGeometryService {
#if WRONG_CTOR
        public MakerGeometryService() { throw new InvalidOperationException("CPU constructor fixture must not run"); }
#endif
#if WRONG_SIGNATURE
        public object ExportGeometry(string absoluteJsonPath,string[] meshes=null,bool includeBlendShapeFrames=false,string materialTextureDirectory=null,bool includeActorTransforms=false) { return null; }
#else
        public Dictionary<string,object> ExportGeometry(string absoluteJsonPath,string[] meshes=null,bool includeBlendShapeFrames=false,string materialTextureDirectory=null,bool includeActorTransforms=false) {
            if (meshes != null || !includeBlendShapeFrames || materialTextureDirectory != null || !includeActorTransforms) throw new InvalidOperationException("Exact installed argument vector not preserved");
            if (Environment.GetEnvironmentVariable("HS2_SCOPE_CPU_FIXTURE_FAILURE") == "throw") throw new InvalidOperationException("CPU fixture native export failure");
            string stamp=DateTime.UtcNow.ToString("O");
            File.WriteAllText(absoluteJsonPath,JsonConvert.SerializeObject(new Dictionary<string,object> {
                {"snapshot_kind","maker_live_skinned_geometry"},{"frame_count",42},{"frame_count_end",42},
                {"captured_utc",stamp},{"include_actor_transforms",true},{"include_blendshape_frames",true},
                {"fixture_only",true},{"native_face_drivers",new Dictionary<string,object>{{"fixture_only",true}}}
            }));
            string sha;using(var hash=SHA256.Create())using(var file=File.OpenRead(absoluteJsonPath)) sha=BitConverter.ToString(hash.ComputeHash(file)).Replace("-","").ToLowerInvariant();
            if(Environment.GetEnvironmentVariable("HS2_SCOPE_CPU_FIXTURE_FAILURE")=="hash")sha="wrong";
            return new Dictionary<string,object>{{"path",absoluteJsonPath},{"sha256",sha},{"bytes",new FileInfo(absoluteJsonPath).Length},{"captured_utc",stamp},{"frame_count",42},{"fixture_only",true}};
        }
#endif
    }
}
