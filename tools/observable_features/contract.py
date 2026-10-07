"""Strict input contract. Schema validity never certifies anatomy or visibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

Vec3 = tuple[StrictFloat, StrictFloat, StrictFloat]
Vec2 = tuple[StrictFloat, StrictFloat]
SHA = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, frozen=True)


class Claims(StrictModel):
    anatomical_correspondence_validated: Literal[False] = False
    image_visibility_validated: Literal[False] = False
    cross_view_fixed_material_point_validated: Literal[False] = False
    cross_base_semantic_equivalence_validated: Literal[False] = False
    likeness_validated: Literal[False] = False


class GeometryScope(StrictModel):
    path: str = Field(min_length=1)
    sha256: SHA
    head_id: StrictInt = Field(ge=0)
    renderer_path: str = Field(min_length=1)
    source_geometry_sha256: SHA
    frame_count: StrictInt = Field(ge=0)
    pose_signature: SHA
    native_face_values_sha256: SHA
    expression_configuration_sha256: SHA
    abmx_parameter_values_status: Literal['unverified_not_bound_by_this_contract']
    coordinate_space: Literal['renderer_baked_raw']
    units: Literal['game_units_unscaled']
    frame_policy_id: str = Field(min_length=1)
    frame_policy_certified: Literal[False] = False


class ImageScope(StrictModel):
    path: str = Field(min_length=1)
    sha256: SHA
    width: StrictInt = Field(gt=0)
    height: StrictInt = Field(gt=0)
    view_id: str = Field(min_length=1)
    camera_payload_path: str = Field(min_length=1)
    camera_payload_sha256: SHA
    coordinate_convention: Literal['PNG pixel centers top_left']
    camera_pixel_pose_certified: Literal[False] = False


class FullSubmesh(StrictModel):
    kind: Literal['full_asset_submesh']
    submesh_id: StrictInt = Field(ge=0)
    anatomical_region_label: None = None


class BaseFeature(StrictModel):
    id: str = Field(min_length=1)
    claims: Claims = Field(default_factory=Claims)
    certified_anatomical_label: None = None


class FormalSurface(BaseFeature):
    kind: Literal['formal_surface']
    operation: Literal['directional_span', 'directional_support_set', 'surface_area', 'plane_intersection_segments']
    region: FullSubmesh
    axis: Vec3 | None = None
    plane_offset: StrictFloat | None = None
    numerical_tolerance_game_units: StrictFloat = Field(gt=0)
    recomputation_rule: Literal['recompute_on_each_actual_surface']

    @model_validator(mode='after')
    def parameters(self):
        directional = self.operation != 'surface_area'
        if directional != (self.axis is not None):
            raise ValueError('Axis required only for directional/plane operations')
        if self.axis is not None and not np.isclose(np.linalg.norm(self.axis), 1., atol=1e-12, rtol=0):
            raise ValueError('Explicit unit axis required; implicit renormalization forbidden')
        if (self.operation == 'plane_intersection_segments') != (self.plane_offset is not None):
            raise ValueError('Plane offset required only for plane intersection')
        return self


class MaterialCandidate(BaseFeature):
    kind: Literal['fixed_material_candidate']
    triangle_id: StrictInt = Field(ge=0)
    ordered_vertex_ids: tuple[StrictInt, StrictInt, StrictInt]
    barycentric: Vec3
    semantic_hypothesis: str | None = None
    transport_rule: Literal['fixed_source_triangle_barycentric']

    @model_validator(mode='after')
    def weights(self):
        if any(x < 0 for x in self.ordered_vertex_ids) or len(set(self.ordered_vertex_ids)) != 3:
            raise ValueError('Three distinct nonnegative ordered vertex IDs required')
        if min(self.barycentric) < 0 or max(self.barycentric) > 1 or abs(sum(self.barycentric)-1) > 1e-12:
            raise ValueError('Nonnegative unit-sum barycentrics required')
        return self


class AppearancePixel(BaseFeature):
    kind: Literal['appearance_pixel']
    image: ImageScope
    visibility: Literal['visible', 'occluded', 'ambiguous']
    xy: Vec2 | None
    uncertainty_radius_px: StrictFloat | None
    appearance_definition: str = Field(min_length=1)
    skin_semantic_decision: Literal['unknown', 'texture', 'ambiguous']
    annotation_method: Literal['original_png_before_overlay']

    @model_validator(mode='after')
    def observable(self):
        if self.visibility == 'visible':
            if self.xy is None or self.uncertainty_radius_px is None or self.uncertainty_radius_px < 0:
                raise ValueError('Visible appearance needs coordinates and honest nonnegative uncertainty')
            if not (0 <= self.xy[0] < self.image.width and 0 <= self.xy[1] < self.image.height):
                raise ValueError('Pixel outside original PNG')
        elif self.xy is not None or self.uncertainty_radius_px is not None:
            raise ValueError('Occluded/ambiguous observations must preserve null coordinates and uncertainty')
        return self


class Unobservable(BaseFeature):
    kind: Literal['unobservable']
    intended_feature: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    value: None = None


Feature = Annotated[FormalSurface | MaterialCandidate | AppearancePixel | Unobservable, Field(discriminator='kind')]


class Manifest(StrictModel):
    schema_version: Literal[1]
    geometry: GeometryScope
    features: list[Feature] = Field(min_length=1)
    purpose: Literal['measurement_diagnostic_only']
    fixed_image_gate_max_error_px: Literal[2.0] = 2.0
    full_infrastructure_goal_complete: Literal[False] = False

    @model_validator(mode='after')
    def unique_ids(self):
        if len({f.id for f in self.features}) != len(self.features):
            raise ValueError('Duplicate feature IDs')
        return self


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def read_bound(path, expected):
    if not Path(path).is_absolute() or digest(path) != expected:
        raise ValueError('Absolute artifact path with matching byte SHA required')
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_sources(manifest: Manifest):
    scope = manifest.geometry
    snapshot = read_bound(scope.path, scope.sha256)
    if snapshot.get('schema_version') != 1 or snapshot.get('snapshot_kind') != 'maker_live_skinned_geometry':
        raise ValueError('Actual Maker schema1 geometry required')
    if (snapshot.get('character', {}).get('head_id') != scope.head_id or
        snapshot.get('frame_count') != scope.frame_count or snapshot.get('pose_signature') != scope.pose_signature):
        raise ValueError('Wrong base/head, pose or frame scope')
    if snapshot.get('frame_count_end') != scope.frame_count:
        raise ValueError('Geometry export crossed actual frame boundary')
    character = snapshot['character']
    values = np.asarray(character.get('shape_value_face'), dtype=float)
    if values.shape != (character.get('native_count'),) or not len(values) or not np.isfinite(values).all():
        raise ValueError('Actual finite native face array/count required')
    if (json_digest(character['shape_value_face']) != scope.native_face_values_sha256 or
        json_digest(character.get('expression')) != scope.expression_configuration_sha256 or
        not isinstance(character.get('expression'), dict)):
        raise ValueError('Native state or expression configuration changed')
    matches = [m for m in snapshot['meshes'] if m['renderer_path'] == scope.renderer_path]
    if len(matches) != 1 or matches[0].get('source_geometry_sha256') != scope.source_geometry_sha256:
        raise ValueError('Renderer/source identity mismatch or ambiguity')
    for feature in manifest.features:
        if isinstance(feature, AppearancePixel):
            image = feature.image
            if not Path(image.path).is_absolute() or digest(image.path) != image.sha256:
                raise ValueError('Image bytes/path changed')
            from PIL import Image
            with Image.open(image.path) as im:
                if im.format != 'PNG':
                    raise ValueError('Actual PNG encoding required, not a renamed image')
                if im.size != (image.width, image.height):
                    raise ValueError('PNG dimensions differ from declared scope')
                im.verify()
            capture = read_bound(image.camera_payload_path, image.camera_payload_sha256)
            # A single capture object is required, never an unbound multi-view aggregate.
            if capture.get('path') != image.path or capture.get('width') != image.width or capture.get('height') != image.height:
                raise ValueError('Actual single camera payload does not bind PNG')
            pair = capture.get('paired_geometry', {})
            if pair.get('sha256') != scope.sha256 or pair.get('pose_signature') != scope.pose_signature or pair.get('frame_count') != scope.frame_count:
                raise ValueError('Appearance and geometry scope differ')
            from tools.surface_calibration.core import Camera
            camera = Camera.from_capture(capture, diagnostic=True)
            if not camera.pose_pairing_validated:
                raise ValueError('Actual camera/geometry pose and same-frame pairing required')
    return snapshot, matches[0]
