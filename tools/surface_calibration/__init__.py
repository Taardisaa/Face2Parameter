"""Local surface-candidate geometry tools; no detector or game initialization."""
from .core import (
    Camera, ContractError, Ray, SurfaceMesh, all_intersections, follow,
    meshes_from_geometry, observe, validate_reprojection,
)

__all__ = [
    "Camera", "ContractError", "Ray", "SurfaceMesh", "all_intersections", "follow",
    "meshes_from_geometry", "observe", "validate_reprojection",
]
