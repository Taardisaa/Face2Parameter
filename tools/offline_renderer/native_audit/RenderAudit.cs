using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace HS2_RenderAudit
{
    // Invoked only on the game's main thread by the separately installed bridge.
    // No materials getter (clones), Set methods, rendering, texture readback,
    // reflection-probe render, ABMX Apply or property getters with side effects.
    public static class RenderAudit
    {
        static readonly BindingFlags AllFields=BindingFlags.Instance|BindingFlags.Public|BindingFlags.NonPublic;
        static string rawOutput;
        static bool queuedThisCapture;
        static readonly Dictionary<int,Dictionary<string,object>> rawJobs=new Dictionary<int,Dictionary<string,object>>();
        [DllImport("kernel32",CharSet=CharSet.Unicode,SetLastError=true)] static extern IntPtr LoadLibrary(string path);
        [DllImport("HS2_RenderReadback.dll",CallingConvention=CallingConvention.Cdecl)] static extern IntPtr GetRenderEventFunc();
        [DllImport("HS2_RenderReadback.dll",CallingConvention=CallingConvention.Cdecl,CharSet=CharSet.Unicode)] static extern int QueueTexture(IntPtr resource,string path);
        [DllImport("HS2_RenderReadback.dll",CallingConvention=CallingConvention.Cdecl)] static extern int GetTextureStatus(int job,StringBuilder error,int capacity);
        static void QueueRaw(Texture texture,Dictionary<string,object> info)
        {
            if(rawOutput==null)return;
            int id=texture.GetInstanceID();
            if(!rawJobs.TryGetValue(id,out var job))
            {
                string directory=System.IO.Path.Combine(rawOutput,"texture_"+id+"_frame_"+Time.frameCount);
                job=new Dictionary<string,object>{{"texture_id",id},{"path",directory},{"submitted_frame",Time.frameCount},
                    {"submitted_utc",DateTime.UtcNow.ToString("o")},{"scope","Render-thread native resource snapshot, not later synchronous Material/Shader descriptor frame equality."}};
                try
                {
                    if(SystemInfo.graphicsDeviceType!=GraphicsDeviceType.Direct3D11)throw new NotSupportedException("Raw export requires D3D11");
                    string native=System.IO.Path.Combine(System.IO.Path.GetDirectoryName(typeof(RenderAudit).Assembly.Location),"HS2_RenderReadback.dll");
                    if(LoadLibrary(native)==IntPtr.Zero)throw new InvalidOperationException("LoadLibrary failed: "+Marshal.GetLastWin32Error());
                    var pointer=texture.GetNativeTexturePtr();
                    if(pointer==IntPtr.Zero)throw new InvalidOperationException("Texture has no native resource");
                    int handle=QueueTexture(pointer,directory);
                    if(handle<0)throw new InvalidOperationException("Native QueueTexture rejected resource");
                    job["native_job"]=handle;job["status"]="pending";queuedThisCapture=true;
                }
                catch(Exception e){job["status"]="failed";job["error"]=e.GetType().Name+": "+e.Message;}
                rawJobs[id]=job;
            }
            info["raw_export"]=job;
        }
        public static Dictionary<string,object> PollRawTextures()
        {
            foreach(var job in rawJobs.Values)
                if(job.TryGetValue("native_job",out var handle))
                {
                    var error=new StringBuilder(4096);
                    int status=GetTextureStatus((int)handle,error,error.Capacity);
                    job["status"]=status==0?"pending":status==1?"complete":"failed";
                    if(status<0||status==2)job["error"]=error.ToString();
                    job["observed_status_frame"]=Time.frameCount;
                }
            return new Dictionary<string,object>{{"jobs",rawJobs.Values.ToArray()},
                {"pending",rawJobs.Values.Count(j=>(string)j["status"]=="pending")},
                {"complete",rawJobs.Values.Count(j=>(string)j["status"]=="complete")},
                {"failed",rawJobs.Values.Count(j=>(string)j["status"]=="failed")}};
        }
        static object Vec(Vector4 v)=>new[]{v.x,v.y,v.z,v.w};
        static object Vec(Vector3 v)=>new[]{v.x,v.y,v.z};
        static object ColorValue(Color c)=>new[]{c.r,c.g,c.b,c.a};
        static object MatrixValue(Matrix4x4 m)=>Enumerable.Range(0,16).Select(i=>m[i/4,i%4]).ToArray();
        static string PathOf(Transform t)=>t==null?null:(t.parent==null?"":PathOf(t.parent))+"/"+t.name+"["+t.GetSiblingIndex()+"]";
        static object SH(SphericalHarmonicsL2 sh)=>Enumerable.Range(0,3).Select(c=>Enumerable.Range(0,9).Select(i=>sh[c,i]).ToArray()).ToArray();
        static object TextureInfo(Texture t)
        {
            if(t==null)return new Dictionary<string,object>{{"is_null",true}};
            var row=new Dictionary<string,object>{{"is_null",false},{"instance_id",t.GetInstanceID()},
                {"name",t.name},{"type",t.GetType().FullName},{"dimension",t.dimension.ToString()},
                {"width",t.width},{"height",t.height},{"wrap_u",t.wrapModeU.ToString()},
                {"wrap_v",t.wrapModeV.ToString()},{"wrap_w",t.wrapModeW.ToString()},
                {"filter",t.filterMode.ToString()},{"anisotropy",t.anisoLevel},{"pixels_exported",false}};
            if(t is Texture2D tex){row["format"]=tex.format.ToString();row["mipmap_count"]=tex.mipmapCount;row["is_readable"]=tex.isReadable;}
            if(t is Cubemap cube){row["format"]=cube.format.ToString();row["mipmap_count"]=cube.mipmapCount;}
            if(t is RenderTexture rt){row["format"]=rt.format.ToString();row["samples"]=rt.antiAliasing;row["depth"]=rt.depth;row["volume_depth"]=rt.volumeDepth;row["srgb"]=rt.sRGB;row["use_mipmap"]=rt.useMipMap;row["auto_generate_mips"]=rt.autoGenerateMips;}
            foreach(string property in new[]{"graphicsFormat","isDataSRGB","activeTextureColorSpace"})
            {
                var p=t.GetType().GetProperty(property,BindingFlags.Public|BindingFlags.Instance);
                if(p!=null)try{row[property]=p.GetValue(t,null)?.ToString();}catch(Exception e){row[property+"_error"]=e.Message;}
            }
            QueueRaw(t,row);
            return row;
        }
        static List<JObject> MergeSpecs(JArray manifests)
        {
            var result=new Dictionary<string,JObject>(StringComparer.Ordinal);
            foreach(string path in manifests)
            {
                var document=JObject.Parse(File.ReadAllText(path));
                foreach(JObject spec in (JArray)document["shaders"])
                {
                    var name=(string)spec["name"];
                    if(result.ContainsKey(name))throw new InvalidOperationException("Duplicate shader request "+name);
                    result[name]=spec;
                }
            }
            return result.Values.ToList();
        }
        static object ReadUniform(Material material,string name,string kind,int size)
        {
            if(size>0)
            {
                if(kind=="float")return material.GetFloatArray(name);
                if(kind=="vector")return material.GetVectorArray(name).Select(Vec).ToArray();
                if(kind=="matrix")return material.GetMatrixArray(name).Select(MatrixValue).ToArray();
                throw new NotSupportedException("Native uniform array type "+kind);
            }
            switch(kind)
            {
                case "float":case "range":return material.GetFloat(name);
                case "int":case "bool":return material.GetInt(name);
                case "vector":return Vec(material.GetVector(name));
                case "color":return ColorValue(material.GetColor(name));
                case "texture":return TextureInfo(material.GetTexture(name));
                case "matrix":return MatrixValue(material.GetMatrix(name));
                default:throw new NotSupportedException(kind);
            }
        }
        static object ReadMaterial(Material material,List<JObject> specs)
        {
            var row=new Dictionary<string,object>{{"name",material.name},{"instance_id",material.GetInstanceID()},
                {"shader",material.shader.name},{"keywords",material.shaderKeywords},{"render_queue",material.renderQueue},
                {"passes",Enumerable.Range(0,material.passCount).Select(material.GetPassName).ToArray()}};
            var spec=specs.SingleOrDefault(s=>(string)s["name"]==material.shader.name);
            if(spec==null){row["typed_properties_complete"]=false;row["missing_shader_manifest"]=true;return row;}
            var requested=new Dictionary<string,JObject>(StringComparer.Ordinal);
            foreach(JObject item in (JArray)spec["compiled_material_globals"]??new JArray())requested[(string)item["name"]]=item;
            foreach(JObject item in (JArray)spec["properties"]??new JArray())requested[(string)item["name"]]=item;
            var values=new List<object>();
            foreach(var item in requested)
            {
                string name=item.Key,kind=(string)item.Value["type"];
                var record=new Dictionary<string,object>{{"name",name},{"type",kind},{"has_property",material.HasProperty(name)}};
                try
                {
                    record["value"]=ReadUniform(material,name,kind,(int?)item.Value["array_size"]??0);
                    if(kind=="color")record["linear_candidate"]=ColorValue(material.GetColor(name).linear);
                    record["read_succeeded"]=true;
                }
                catch(Exception e){record["read_succeeded"]=false;record["error"]=e.GetType().Name+": "+e.Message;}
                values.Add(record);
            }
            row["properties"]=values;
            row["compiled_textures"]=((JArray)spec["compiled_texture_bindings"]??new JArray()).Select(t=>new Dictionary<string,object>{{"name",(string)t},{"global_descriptor",TextureInfo(Shader.GetGlobalTexture((string)t))}}).ToArray();
            row["color_gpu_conversion_certified"]=false;
            row["scope"]="GetColor/GetVector/GetFloat/GetInt/arrays reads; linear color candidate retained separately, not asserted GPU cbuffer. Engine per-draw globals may not be material-owned even within compiled $Globals.";
            return row;
        }
        static object FieldValue(object value,List<JObject> specs)
        {
            if(value==null)return null;
            var type=value.GetType();
            if(type.IsPrimitive||value is string||value is decimal)return value;
            if(type.IsEnum)return value.ToString();
            if(value is Color c)return ColorValue(c);
            if(value is Vector4 v4)return Vec(v4);
            if(value is Vector3 v3)return Vec(v3);
            if(value is Vector2 v2)return new[]{v2.x,v2.y};
            if(value is LayerMask layer)return layer.value;
            if(value is Matrix4x4 matrix)return MatrixValue(matrix);
            if(value is Material material)return ReadMaterial(material,specs);
            if(value is Texture texture)return TextureInfo(texture);
            if(value is Shader shader)return new Dictionary<string,object>{{"shader",shader.name},{"instance_id",shader.GetInstanceID()}};
            if(value is UnityEngine.Object obj)return new Dictionary<string,object>{{"type",type.FullName},{"name",obj.name},{"instance_id",obj.GetInstanceID()}};
            return new Dictionary<string,object>{{"type",type.FullName},{"covered",false}};
        }
        static object Effect(Behaviour effect,int order,List<JObject> specs)
        {
            var fields=new List<object>();
            for(Type type=effect.GetType();type!=null&&type!=typeof(MonoBehaviour);type=type.BaseType)
                foreach(var field in type.GetFields(AllFields|BindingFlags.DeclaredOnly))
                {
                    if(field.IsStatic)continue;
                    var record=new Dictionary<string,object>{{"declaring_type",type.FullName},{"field",field.Name},
                         {"public",field.IsPublic},{"serialize_field",field.IsDefined(typeof(SerializeField),false)}};
                    try{record["value"]=FieldValue(field.GetValue(effect),specs);record["read_succeeded"]=true;}
                    catch(Exception e){record["read_succeeded"]=false;record["error"]=e.GetType().Name+": "+e.Message;}
                    fields.Add(record);
                }
            return new Dictionary<string,object>{{"component_index",order},{"type",effect.GetType().FullName},
                {"enabled",effect.enabled},{"active",effect.gameObject.activeInHierarchy},{"fields",fields},
                {"scope","GetComponents order, not a certified GPU image effect execution trace. No property getters invoked."}};
        }
        public static Dictionary<string,object> Capture()
        {
            // A sidecar config beside this DLL keeps data paths out of the bridge.
            string configPath=typeof(RenderAudit).Assembly.Location+".json";
            var config=JObject.Parse(File.ReadAllText(configPath));
            rawOutput=(string)config["raw_texture_output_directory"];
            if(rawOutput!=null)rawOutput=System.IO.Path.GetFullPath(rawOutput);
            queuedThisCapture=false;
            var specs=MergeSpecs((JArray)config["shader_manifests"]);
            var camera=Camera.main;
            var rows=new List<object>();
            var chars=UnityEngine.Object.FindObjectsOfType<AIChara.ChaControl>().Where(c=>c.objHead!=null&&c.gameObject.activeInHierarchy).ToArray();
            foreach(var cha in chars)
                foreach(var renderer in cha.objHead.GetComponentsInChildren<Renderer>(true))
                {
                    var block=new MaterialPropertyBlock();renderer.GetPropertyBlock(block);
                    var row=new Dictionary<string,object>{{"renderer",renderer.name},{"renderer_path",PathOf(renderer.transform)},
                        {"enabled",renderer.enabled},{"active",renderer.gameObject.activeInHierarchy},
                        {"materials",renderer.sharedMaterials.Select(m=>m==null?null:ReadMaterial(m,specs)).ToArray()},
                        {"light_probe_usage",renderer.lightProbeUsage.ToString()},
                        {"reflection_probe_usage",renderer.reflectionProbeUsage.ToString()},
                        {"probe_anchor",renderer.probeAnchor==null?null:Vec(renderer.probeAnchor.position)},
                        {"bounds_center",Vec(renderer.bounds.center)},
                        {"property_block_present",!block.isEmpty},{"property_block_values_resolved",false}};
                    var perSlot=new List<bool>();
                    for(int slot=0;slot<renderer.sharedMaterials.Length;slot++){block.Clear();renderer.GetPropertyBlock(block,slot);perSlot.Add(!block.isEmpty);}
                    row["per_slot_property_block_present"]=perSlot;
                    SphericalHarmonicsL2 probe;
                    var point=renderer.probeAnchor==null?renderer.bounds.center:renderer.probeAnchor.position;
                    LightProbes.GetInterpolatedProbe(point,renderer,out probe);
                    row["interpolated_probe_sh_3x9"]=SH(probe);
                    var nearby=new List<ReflectionProbeBlendInfo>();renderer.GetClosestReflectionProbes(nearby);
                    row["reflection_probe_blends"]=nearby.Select(p=>new Dictionary<string,object>{{"weight",p.weight},{"probe_id",p.probe==null?(object)null:p.probe.GetInstanceID()},{"texture",p.probe==null?null:TextureInfo(p.probe.texture)}}).ToArray();
                    rows.Add(row);
                }
            var globals=new Dictionary<string,object>();
            var names=new[]{"unity_SHAr","unity_SHAg","unity_SHAb","unity_SHBr","unity_SHBg","unity_SHBb","unity_SHC",
                "unity_ProbesOcclusion","unity_SpecCube0_HDR","unity_SpecCube1_HDR","unity_SpecCube0_BoxMax","unity_SpecCube0_BoxMin",
                "unity_SpecCube0_ProbePosition","unity_SpecCube1_BoxMax","unity_SpecCube1_BoxMin","unity_SpecCube1_ProbePosition"};
            foreach(string name in names)globals[name]=Vec(Shader.GetGlobalVector(name));
            var textureNames=specs.SelectMany(s=>(JArray)s["compiled_texture_bindings"]??new JArray()).Select(t=>(string)t).Distinct();
            var textures=textureNames.ToDictionary(n=>n,n=>TextureInfo(Shader.GetGlobalTexture(n)));
            var reflection=UnityEngine.Object.FindObjectsOfType<ReflectionProbe>().Select(p=>new Dictionary<string,object>{
                {"path",PathOf(p.transform)},{"enabled",p.enabled},{"active",p.gameObject.activeInHierarchy},{"mode",p.mode.ToString()},
                {"intensity",p.intensity},{"blend_distance",p.blendDistance},{"box_projection",p.boxProjection},{"size",Vec(p.size)},
                {"center",Vec(p.center)},{"bounds_center",Vec(p.bounds.center)},{"bounds_size",Vec(p.bounds.size)},
                {"texture",TextureInfo(p.texture)},{"hdr_decode",Vec(p.textureHDRDecodeValues)}}).ToArray();
            var result=new Dictionary<string,object>{{"schema_version",2},{"captured_utc",DateTime.UtcNow.ToString("o")},{"frame_count",Time.frameCount},
                {"quality_color_space",QualitySettings.activeColorSpace.ToString()},{"pixel_light_count",QualitySettings.pixelLightCount},
                {"ambient_mode",RenderSettings.ambientMode.ToString()},{"ambient_intensity",RenderSettings.ambientIntensity},
                {"ambient_light",ColorValue(RenderSettings.ambientLight)},{"ambient_sky",ColorValue(RenderSettings.ambientSkyColor)},
                {"ambient_equator",ColorValue(RenderSettings.ambientEquatorColor)},{"ambient_ground",ColorValue(RenderSettings.ambientGroundColor)},
                {"ambient_probe_sh_3x9",SH(RenderSettings.ambientProbe)},{"reflection_intensity",RenderSettings.reflectionIntensity},
                {"reflection_bounces",RenderSettings.reflectionBounces},{"default_reflection_mode",RenderSettings.defaultReflectionMode.ToString()},
                {"custom_reflection",TextureInfo(RenderSettings.customReflection)},{"shader_global_vector_reads",globals},
                {"default_reflection_texture",TextureInfo(ReflectionProbe.defaultTexture)},
                {"default_reflection_hdr_decode",Vec(ReflectionProbe.defaultTextureHDRDecodeValues)},
                {"shader_global_textures",textures},{"head_renderers",rows},{"active_character_count",chars.Length},
                {"camera_effects",camera==null?null:camera.GetComponents<Behaviour>().Select((e,i)=>Effect(e,i,specs)).ToArray()},
                {"auxiliary_effects",UnityEngine.Object.FindObjectsOfType<Behaviour>().Where(b=>b.GetType().FullName=="DHH.SSS_convolution").Select((e,i)=>Effect(e,i,specs)).ToArray()},
                {"reflection_probes",reflection},{"config",configPath},{"config_sha256",Hash(configPath)},
                {"frame_count_end",Time.frameCount},{"runtime_mutation",false},
                {"scope","Main-thread read-only material/SH/effect field/probe descriptors. Optional native raw textures are queued on render thread; built-in engine per-draw cbuffer values/property blocks, shader-view gamma, shadow maps and actual GPU active variants remain unverified."}};
            if(queuedThisCapture)GL.IssuePluginEvent(GetRenderEventFunc(),1);
            result["raw_texture_export"]=PollRawTextures();
            return result;
        }
        static string Hash(string path)
        {
            using(var hash=System.Security.Cryptography.SHA256.Create())
                return BitConverter.ToString(hash.ComputeHash(File.ReadAllBytes(path))).Replace("-","").ToLowerInvariant();
        }
    }
}
