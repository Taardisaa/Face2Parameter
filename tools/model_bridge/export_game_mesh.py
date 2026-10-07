"""Emit model-generated vertices for direct HS2 import; never fit HS2 sliders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import ModelArtifact, sha


def export(artifact, output, *, head_local, with_rig=False, source_obj=None, texture=None,
           source_mask=None, left_eye_texture=None, right_eye_texture=None):
    vertices, faces = artifact.mesh(head_local=head_local)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Triangle mesh required")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError("Source triangle index out of bounds")
    payload = {
        "format": "hs2_source_head_mesh_v1", "geometry_mode": "head_local" if head_local else "original_pose",
        "vertices": vertices.tolist(), "triangles": faces.reshape(-1).tolist(),
        "source_parameters": {k: v.tolist() for k, v in artifact.parameters.items()},
        "source": {"manifest": str(artifact.path), "manifest_sha256": sha(artifact.path),
                   "image_index": artifact.image_index,
                   "model_revision": artifact.manifest["git_revision"],
                   "model_format": artifact.manifest["format"],
                   "checkpoint_sha256": artifact.manifest["checkpoint"]["sha256"],
                   "decoder_sources": artifact.manifest["sources"],
                   "image_sha256": artifact.image["input_sha256"],
                   "artifact_sha256": artifact.image["sha256"]},
        "shape_policy": "Direct source-model vertices and topology; no vertex fitting, remeshing or slider conversion",
        "game_support": "Source mesh interchange only; native head/rig/material/expression integration is separate",
    }
    if with_rig:
        if not head_local:
            raise ValueError("Audited source rig requires explicit head-local geometry")
        from .rig import export_rig
        payload["format"] = "hs2_source_head_mesh_v2"
        payload["rig"] = export_rig(artifact)
        payload["game_support"] = "Original source LBS rig; fixed shape/expression, source pose controls; native facial retargeting and neck join separate"
    if source_obj is not None:
        if not head_local:
            raise ValueError('Source surface requires explicit head-local geometry')
        from .surface import add_surface
        add_surface(payload, Path(source_obj), Path(texture) if texture is not None else None)
    elif texture is not None:
        raise ValueError('Texture requires the corresponding source corner-UV OBJ')
    if source_mask is not None:
        if source_obj is None:
            raise ValueError('Source components require the corresponding corner-UV OBJ')
        from .components import add_components
        add_components(payload, Path(source_mask),
                       None if left_eye_texture is None else Path(left_eye_texture),
                       None if right_eye_texture is None else Path(right_eye_texture))
    elif left_eye_texture is not None or right_eye_texture is not None:
        raise ValueError('Eye textures require explicit source component masks')
    encoded = json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n'
    if len(encoded.encode('utf-8')) > 16 * 1024 * 1024:
        raise ValueError('Combined source/rig/PNG artifact exceeds the native 16 MiB limit')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    return {"file": str(output.resolve()), "sha256": sha(output),
            "vertices": len(vertices), "triangles": len(faces), "hs2_parameter_mapping": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--head-local", action="store_true")
    parser.add_argument("--with-rig", action="store_true", help="Carry complete original FLAME LBS and pose-corrective state")
    parser.add_argument('--source-obj', type=Path, help='Exact source face/corner-UV OBJ; template positions are ignored')
    parser.add_argument('--texture', type=Path, help='Explicit external opaque PNG in the source UV layout; not inferred albedo')
    parser.add_argument('--source-mask', type=Path, help='Pinned authored FLAME component masks; preserves original face order')
    parser.add_argument('--left-eye-texture', type=Path, help='Explicit source-UV PNG for the original left eyeball')
    parser.add_argument('--right-eye-texture', type=Path, help='Explicit source-UV PNG for the original right eyeball')
    args = parser.parse_args()
    artifact = ModelArtifact(args.manifest, args.image_index)
    print(json.dumps(export(artifact, args.out, head_local=args.head_local, with_rig=args.with_rig,
                            source_obj=args.source_obj, texture=args.texture, source_mask=args.source_mask,
                            left_eye_texture=args.left_eye_texture, right_eye_texture=args.right_eye_texture)))


if __name__ == "__main__":
    main()
