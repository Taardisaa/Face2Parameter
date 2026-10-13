"""Verify streamed original-call evidence in official D3D11 XML; CPU only.

Rejects an automatic-retry/ordinary frame lacking the original annotations.
This is command provenance, not pixel equivalence or full RGB acceptance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def field(node, name):
    found = [child for child in node if child.attrib.get("name") == name]
    if len(found) != 1:
        raise ValueError(f"Exactly one {name} field required")
    return found[0]


def number(node, name):
    child = field(node, name)
    if child.tag not in ("uint", "int", "enum", "ResourceId"):
        raise ValueError(f"Unknown numeric schema: {name}/{child.tag}")
    return int(child.text)


def verify(root, nonce: str, width: int, height: int) -> dict:
    if root.tag != "rdc" or root.findtext("header/driver") != "D3D11":
        raise ValueError("Official actual D3D11 structured export required")
    chunks = list(root.find("chunks"))
    annotations = {}
    for position, chunk in enumerate(chunks):
        if chunk.attrib.get("name") != "ID3D11DeviceContext::SetCommandAnnotation":
            continue
        key = field(chunk, "key").text
        if key not in ("hs2_scope.begin", "hs2_scope.target", "hs2_scope.end"):
            continue
        if key in annotations:
            raise ValueError("Duplicate original-call annotation; one invocation required")
        expected_type = 9 if key == "hs2_scope.target" else 8
        if number(chunk, "valueType") != expected_type or number(chunk, "valueVectorWidth") != 1:
            raise ValueError("Wrong native annotation type/vector width")
        value = field(chunk, "value")
        if key == "hs2_scope.target":
            array = field(value, "vector.uint64")
            words = [int(child.text) for child in array]
            if array.tag != "array" or len(words) != 4 or not words[0] or any(words[1:]):
                raise ValueError("Actual APIObject ResourceId scalar union required")
            decoded = words[0]
        else:
            decoded = field(value, "string").text
            if decoded != nonce:
                raise ValueError("Original-call nonce mismatch")
        annotations[key] = {"position": position, "chunk_index": int(chunk.attrib["chunkIndex"]),
                            "thread_id": int(chunk.attrib["threadID"]), "context": number(chunk, "Context"),
                            "value": decoded}
    if set(annotations) != {"hs2_scope.begin", "hs2_scope.target", "hs2_scope.end"}:
        raise ValueError("Actual file lacks all three original-call annotations; reject retry frame")
    begin, target, end = [annotations[key] for key in ("hs2_scope.begin", "hs2_scope.target", "hs2_scope.end")]
    if not begin["position"] < target["position"] < end["position"]:
        raise ValueError("Original-call command annotation order differs")
    if not begin["context"] or len({row["context"] for row in annotations.values()}) != 1:
        raise ValueError("Markers must use the same actual immediate context")
    context = begin["context"]
    textures = {}
    for chunk in chunks:
        if chunk.attrib.get("name") in ("ID3D11Device::CreateTexture2D", "ID3D11Device3::CreateTexture2D1"):
            resource = number(chunk, "pTexture")
            if resource in textures:
                raise ValueError("Reused texture resource identity")
            desc = field(chunk, "Descriptor")
            textures[resource] = {key: number(desc, key) for key in ("Width", "Height", "Usage", "CPUAccessFlags")}
    ident = target["value"]
    if ident not in textures or (textures[ident]["Width"], textures[ident]["Height"]) != (width, height):
        raise ValueError("Annotated actual camera target size/creation unavailable")
    # Preserve exact ordered whole-subresource transfer paths. Unsupported shader
    # conversions/cropped readbacks stay unresolved rather than guessing their source.
    paths = {ident: []}
    candidates = []
    draws = 0
    for position in range(target["position"] + 1, end["position"]):
        chunk = chunks[position]
        name = chunk.attrib.get("name", "")
        if name.startswith("ID3D11DeviceContext::Draw"):
            draws += 1
        if name not in ("ID3D11DeviceContext::CopyResource", "ID3D11DeviceContext::CopySubresourceRegion",
                        "ID3D11DeviceContext1::CopySubresourceRegion1", "ID3D11DeviceContext::ResolveSubresource"):
            continue
        if number(chunk, "Context") != context:
            raise ValueError("Readback transfer uses a different context; explicit integration needed")
        src, dst = number(chunk, "pSrcResource"), number(chunk, "pDstResource")
        # A later write from an unrelated source invalidates an earlier alias path.
        if src not in paths:
            paths.pop(dst, None)
            continue
        if name.endswith("CopySubresourceRegion") or name.endswith("CopySubresourceRegion1"):
            if any(number(chunk, key) for key in ("DstSubresource", "DstX", "DstY", "DstZ", "SrcSubresource")):
                raise ValueError("Original target readback uses offsets/subresources; unresolved branch")
            box = field(chunk, "pSrcBox")
            if box.tag != "null":
                bounds = [number(box, key) for key in ("left", "top", "front", "right", "bottom", "back")]
                if bounds != [0, 0, 0, width, height, 1]:
                    raise ValueError("Original target cropped readback requires exact region integration")
        if name.endswith("ResolveSubresource") and (number(chunk, "DstSubresource") or number(chunk, "SrcSubresource")):
            raise ValueError("Resolved nonzero subresource requires explicit integration")
        if dst not in textures or (textures[dst]["Width"], textures[dst]["Height"]) != (width, height):
            raise ValueError("Readback transfer dimensions unknown/different")
        paths[dst] = paths[src] + [{"chunk_index": int(chunk.attrib["chunkIndex"]), "position": position,
                                    "name": name, "source": src, "destination": dst}]
        desc = textures[dst]
        if desc["Usage"] == 3 and desc["CPUAccessFlags"] & 0x20000:
            candidates.append({"staging_resource": dst, "copy_path": paths[dst]})
    if not draws or not candidates:
        raise ValueError("No original draws and real CPU-readable staging transfer from annotated target")
    return {"annotations": annotations, "actual_target_resource": ident, "target_descriptor": textures[ident],
            "recorded_draws_between_target_and_end": draws, "readback_candidates": candidates,
            "marker_and_copy_provenance_passed": True, "native_pixel_equivalence_accepted": False,
            "full_rgb_renderer_complete": False,
            "scope": "Actual command annotations/target staging-copy provenance only; replayed output, 16-render/pass and source-state acceptance remain required"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path)
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    data = args.xml.read_bytes()
    result = verify(ET.fromstring(data), args.nonce, args.width, args.height)
    archive = args.xml.with_suffix("")
    result["input"] = {"xml": str(args.xml.resolve()), "xml_sha256": hashlib.sha256(data).hexdigest(),
                       "zip": str(archive.resolve()), "zip_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    with args.out.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
    print(json.dumps({"target": result["actual_target_resource"], "receipt": str(args.out.resolve()), "rgb_accepted": False}))


if __name__ == "__main__":
    main()
