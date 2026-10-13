using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// Same deliberate dictionary/sequence/primitive boundary as installed JsonUtil.
// No reflection contract resolver, DTO serializer or ADO.NET converter path.
internal static class ScopeJson
{
    internal static JObject ParseObject(string text)
    {
        using (var reader = new JsonTextReader(new StringReader(text)) { DateParseHandling = DateParseHandling.None }) {
            var value = JObject.Load(reader, new JsonLoadSettings { DuplicatePropertyNameHandling = DuplicatePropertyNameHandling.Error });
            while (reader.Read()) if (reader.TokenType != JsonToken.Comment) throw new JsonReaderException("Unexpected content after JSON object");
            ValidateToken(value); return value;
        }
    }
    static void ValidateToken(JToken value)
    {
        switch (value.Type) {
            case JTokenType.Object: foreach (var property in ((JObject)value).Properties()) ValidateToken(property.Value); return;
            case JTokenType.Array: foreach (var item in (JArray)value) ValidateToken(item); return;
            case JTokenType.Integer: case JTokenType.String: case JTokenType.Boolean: case JTokenType.Null: return;
            case JTokenType.Float:
                double number = value.Value<double>(); if (Double.IsNaN(number) || Double.IsInfinity(number)) throw new InvalidOperationException("JSON requires finite numeric values"); return;
            default: throw new InvalidOperationException("Unsupported JSON token type: " + value.Type);
        }
    }
    internal static HS2_RenderDocScope.Request ReadRequest(string text)
    {
        var fields = ParseObject(text); var request = new HS2_RenderDocScope.Request();
        foreach (var field in fields.Properties()) {
            switch (field.Name) {
                case "Enabled": if (field.Value.Type != JTokenType.Boolean) throw new InvalidOperationException("Enabled must be boolean"); request.Enabled = field.Value.Value<bool>(); break;
                case "ExpectedPid":
                    if (field.Value.Type != JTokenType.Integer) throw new InvalidOperationException("ExpectedPid must be uint32 integer");
                    try { request.ExpectedPid = field.Value.Value<uint>(); } catch (Exception error) { throw new InvalidOperationException("ExpectedPid must be uint32 integer", error); } break;
                case "ExactPngPath": request.ExactPngPath = StringField(field); break;
                case "CapturePrefix": request.CapturePrefix = StringField(field); break;
                case "ReceiptPath": request.ReceiptPath = StringField(field); break;
                case "ExpectedNativeSha256": request.ExpectedNativeSha256 = StringField(field); break;
                case "ExpectedRenderDocSha256": request.ExpectedRenderDocSha256 = StringField(field); break;
                case "GeometryBeforePath": request.GeometryBeforePath = StringField(field); break;
                case "GeometryAfterPath": request.GeometryAfterPath = StringField(field); break;
                case "ExpectedBridgeSha256": request.ExpectedBridgeSha256 = StringField(field); break;
                default: throw new InvalidOperationException("Unsupported scope request field: " + field.Name);
            }
        }
        return request;
    }
    static string StringField(JProperty field)
    {
        if (field.Value.Type == JTokenType.Null) return null;
        if (field.Value.Type != JTokenType.String) throw new InvalidOperationException(field.Name + " must be string or null");
        return field.Value.Value<string>();
    }
    internal static Dictionary<string, object> RequestFields(HS2_RenderDocScope.Request request)
    {
        if (request == null) return null;
        var fields = new Dictionary<string, object> {
            { "Enabled", request.Enabled }, { "ExpectedPid", request.ExpectedPid }, { "ExactPngPath", request.ExactPngPath },
            { "CapturePrefix", request.CapturePrefix }, { "ReceiptPath", request.ReceiptPath },
            { "ExpectedNativeSha256", request.ExpectedNativeSha256 }, { "ExpectedRenderDocSha256", request.ExpectedRenderDocSha256 }
        };
        if (request.GeometryBeforePath != null) fields.Add("GeometryBeforePath", request.GeometryBeforePath);
        if (request.GeometryAfterPath != null) fields.Add("GeometryAfterPath", request.GeometryAfterPath);
        if (request.ExpectedBridgeSha256 != null) fields.Add("ExpectedBridgeSha256", request.ExpectedBridgeSha256);
        return fields;
    }
    internal static string Object(Dictionary<string, object> fields)
    {
        var buffer = new StringBuilder();
        using (var text = new StringWriter(buffer, CultureInfo.InvariantCulture))
        using (var writer = new JsonTextWriter(text) { Formatting = Formatting.None }) WriteValue(writer, fields);
        return buffer.ToString();
    }
    // Last-resort error envelope if even a streaming writer/previous JSON fails.
    // All diagnostic strings, including the original raw receipt, are escaped;
    // no Newtonsoft serializer/writer or reflection is used on this path.
    internal static string Emergency(string stage,string error,string saveError,string fallbackError,string previous)
    {
        return "{\"schema_version\":1,\"code\":-100,\"stage\":"+Escape(stage)+",\"error\":"+Escape(error)
            +",\"receipt_write_error\":"+Escape(saveError)+",\"streaming_fallback_error\":"+Escape(fallbackError)
            +",\"previous_receipt_raw\":"+Escape(previous)+",\"capture_accepted\":false}";
    }
    static string Escape(string value)
    {
        if(value==null)return "null";var text=new StringBuilder("\"");
        foreach(char c in value){if(c=='"' || c=='\\'){text.Append('\\');text.Append(c);}
            else if(c<32 || c>126){text.Append("\\u");text.Append(((int)c).ToString("x4",CultureInfo.InvariantCulture));}else text.Append(c);}
        return text.Append('"').ToString();
    }
    internal static void WriteValue(JsonWriter writer, object value)
    {
        if (value == null) { writer.WriteNull(); return; }
        if (value is JToken token) { ValidateToken(token); token.WriteTo(writer); return; }
        if (value is IDictionary dictionary) {
            writer.WriteStartObject(); foreach (DictionaryEntry entry in dictionary) {
                if (!(entry.Key is string key)) throw new InvalidOperationException("JSON object keys must be strings");
                writer.WritePropertyName(key); WriteValue(writer, entry.Value);
            } writer.WriteEndObject(); return;
        }
        if (value is IEnumerable sequence && !(value is string)) {
            writer.WriteStartArray(); foreach (var item in sequence) WriteValue(writer, item); writer.WriteEndArray(); return;
        }
        switch (value) {
            case string item: writer.WriteValue(item); return;
            case char item: writer.WriteValue(item.ToString()); return;
            case bool item: writer.WriteValue(item); return;
            case byte item: writer.WriteValue(item); return;
            case sbyte item: writer.WriteValue(item); return;
            case short item: writer.WriteValue(item); return;
            case ushort item: writer.WriteValue(item); return;
            case int item: writer.WriteValue(item); return;
            case uint item: writer.WriteValue(item); return;
            case long item: writer.WriteValue(item); return;
            case ulong item: writer.WriteValue(item); return;
            case float item: if (Single.IsNaN(item) || Single.IsInfinity(item)) throw new InvalidOperationException("JSON requires finite float"); writer.WriteValue(item); return;
            case double item: if (Double.IsNaN(item) || Double.IsInfinity(item)) throw new InvalidOperationException("JSON requires finite double"); writer.WriteValue(item); return;
            case decimal item: writer.WriteValue(item); return;
            case Enum item: writer.WriteValue(Convert.ToInt64(item, CultureInfo.InvariantCulture)); return;
            default: throw new InvalidOperationException("Unsupported JSON response value: " + value.GetType().FullName);
        }
    }
}
