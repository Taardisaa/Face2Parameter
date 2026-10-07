"""Literal whole-submesh operations. No UV guesses or anatomical region labels."""
from collections import Counter

import numpy as np

from tools.surface_calibration.core import ContractError, validate_reprojection
from .consensus import material


def geometric_definitions(mesh, front_camera, cameras, scene_meshes):
    """Freeze front support maximum and inventory actual indexed boundaries.

    The full asset-provided submesh is the region. An extremum need not be a
    nose, and an indexed boundary need not be a lip. Neither is claimed here.
    """
    if not mesh.world_policy_certified:
        raise ContractError("Geometric definition requires independent world-policy evidence")
    submeshes = sorted(set(mesh.submesh_ids.tolist()))
    if submeshes != [0]:
        raise ContractError("This whole-submesh definition requires exactly the actual submesh 0")
    x, y, width, height = front_camera.rect
    center = [x + width / 2 - .5, front_camera.height - y - height / 2 - .5]
    direction = -front_camera.ray(center).direction
    used_vertices = np.unique(mesh.triangles)
    support = mesh.vertices[used_vertices] @ direction
    ties = used_vertices[np.isclose(support, support.max(), atol=1e-10, rtol=0)]
    definition = {"id": "head_front_support_vertex", "kind": "literal_geometric_material_definition",
                  "semantic_definition": "Maximum directional support on actual full o_head submesh 0 in the certified front capture's toward-camera direction",
                  "anatomical_correspondence_validated": False, "is_named_nose_apex": False,
                  "region_source": "actual exported submesh 0; all referenced vertices; no UV or bone region",
                  "renderer_path": mesh.renderer_path, "source_geometry_sha256": mesh.source_hash,
                  "support_direction_world": direction.tolist(), "direction_source_view_id": front_camera.view_id,
                  "support_tie_tolerance_game_units": 1e-10, "maximizing_vertex_ids": ties.tolist(),
                  "unique_maximum": len(ties) == 1,
                  "independent_image_semantic_observation_present": False,
                  "semantic_or_detector_acceptance_claimed": False}
    if len(ties) == 1:
        vertex = int(ties[0])
        incident = np.flatnonzero(np.any(mesh.triangles == vertex, axis=1))
        triangle = int(incident[0])
        weights = (mesh.triangles[triangle] == vertex).astype(float)
        fixed = material(mesh, triangle, weights)
        definition.update(material=fixed,
                          transport_rule="Follow fixed source-bound vertex/triangle weights; do not recompute the extremum after deformation",
                          views=[validate_reprojection(fixed, camera, scene_meshes) for camera in cameras])
    edges = Counter(tuple(sorted(edge)) for triangle in mesh.triangles for edge in
                    [(triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])])
    boundary = [edge for edge, count in edges.items() if count == 1]
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), set()).add(int(b))
        adjacency.setdefault(int(b), set()).add(int(a))
    components, seen = [], set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        pending, vertices = [start], set()
        while pending:
            vertex = pending.pop()
            if vertex in vertices:
                continue
            vertices.add(vertex)
            pending.extend(adjacency[vertex] - vertices)
        seen |= vertices
        ordered = sorted(vertices)
        points = mesh.vertices[ordered]
        components.append({"component_id": len(components), "vertex_ids": ordered,
                           "vertex_count": len(ordered), "is_closed_degree_two_loop": all(len(adjacency[v]) == 2 for v in ordered),
                           "world_bbox": [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
                           "anatomical_region_label": None})
    return {"schema_version": 1, "support_definition": definition,
            "indexed_topology_boundary_inventory": {
                "region_source": "actual exported o_head submesh 0 ordered triangle indices",
                "source_geometry_sha256": mesh.source_hash, "renderer_path": mesh.renderer_path,
                "boundary_edge_count": len(boundary), "nonmanifold_edge_count": sum(n > 2 for n in edges.values()),
                "position_welding_or_uv_guess_applied": False, "components": components,
                "limits": "UV/material seams can create indexed boundaries. No component is labeled lip/eye/neck without independent asset or annotation evidence."}}
