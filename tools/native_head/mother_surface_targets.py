"""Source-bound regional surface targets for provisional mother authoring.

Original FLAME vertex masks and native skinning support prevent an ear from
matching the cheek. These are support regions, not certified anatomical cuts.
Vermilion candidates retain native interior shape under the authoring energy;
they are driven by separate lip anchors instead of snapping to either side of
FLAME's nearly contacting mouth surfaces.
"""
import pickle
from pathlib import Path

import numpy as np
from matplotlib.path import Path as PolygonPath

from tools.model_bridge.artifact import sha
from tools.native_head.mother_oriented_surface import OrientedSurface
from tools.native_head.mother_template_inputs import source_file


MASK_SHA = "ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492"


class RegionalTargets:
    def __init__(self, native, inverse, target, target_faces, mask_path, anchors):
        if sha(mask_path) != MASK_SHA:
            raise ValueError("Official FLAME region mask differs from audited source")
        masks = pickle.loads(Path(mask_path).read_bytes(), encoding="latin1")
        self.labels = np.full(int(inverse.max())+1, "general", dtype="U16")
        self.surfaces = {"general": OrientedSurface(target[target_faces])}
        self.receipt = dict(source_masks=source_file(mask_path), regions={},
            policy="Original source masks plus dominant native rig support; correspondence partitions, not anatomical cuts",
            anatomical_boundaries_certified=False)
        names = native["bone_names"].tolist()
        for side in ("L", "R"):
            bones = [i for i, name in enumerate(names) if "Ear" in name and name.endswith("_"+side)]
            support = (native["bone_w"]*np.isin(native["bone_idx"], bones)).sum(1)
            ids = np.flatnonzero(support > .5)
            sign = np.sign(native["verts"][ids, 0].mean())
            candidates = [key for key in ("left_ear", "right_ear")
                          if np.sign(target[np.asarray(masks[key], int), 0].mean()) == sign]
            if len(candidates) != 1:
                raise ValueError("Cannot disambiguate source ear masks in actual coordinates")
            key = candidates[0]
            selected = np.isin(target_faces, masks[key]).all(1)
            if not selected.any():
                raise ValueError("No complete source ear triangles")
            self.labels[inverse[ids]] = side+"_ear"
            self.surfaces[side+"_ear"] = OrientedSurface(target[target_faces[selected]])
            self.receipt["regions"][side+"_ear"] = dict(native_vertex_ids=ids.tolist(),
                native_bone_names=[names[i] for i in bones], source_mask=key,
                source_faces=target_faces[selected].tolist())
        for side in ('L', 'R'):
            bones = [i for i, name in enumerate(names) if name.startswith('cf_J_Eye') and name.endswith('_'+side)]
            support = (native['bone_w']*np.isin(native['bone_idx'], bones)).sum(1)
            ids = np.flatnonzero(support>.5)
            if not len(ids):
                raise ValueError('No native dominant eyelid rig support')
            sign = np.sign(native['verts'][ids,0].mean())
            keys = [key for key in ('left_eye_region','right_eye_region')
                    if np.sign(target[np.asarray(masks[key],int),0].mean())==sign]
            if len(keys)!=1:
                raise ValueError('Cannot disambiguate actual source eyelid sides')
            key = keys[0]
            selected = np.isin(target_faces,masks[key]).all(1)
            if not selected.any():
                raise ValueError('No source eyelid-region triangles')
            label = side+'_lid'
            self.labels[inverse[ids]] = label
            self.surfaces[label] = OrientedSurface(target[target_faces[selected]])
            self.receipt['regions'][label] = dict(native_vertex_ids=ids.tolist(),
                native_bone_names=[names[i] for i in bones],source_mask=key,
                source_faces=target_faces[selected].tolist(),
                policy='Dominant original lid-bone support; not a certified anatomical cut')
        # Original annotated donor pigment contour is a candidate vermilion
        # envelope. It is not reused as a model cut or an inner-mouth boundary.
        lip_uv = [a["uv"] for a in anchors["annotations"]
                  if "uv" in a and 48 <= a["target_flame_landmark"] <= 59]
        if len(lip_uv) != 12:
            raise ValueError("Expected complete annotated outer lip contour")
        pigment = PolygonPath(np.r_[lip_uv, lip_uv[:1]])
        lip_ids = np.flatnonzero(pigment.contains_points(native["uv"]))
        self.preserve_native_lips = np.unique(inverse[lip_ids])
        mouth_bones = [i for i, name in enumerate(names) if name.startswith('cf_J_Mouth')]
        mouth_support = (native['bone_w']*np.isin(native['bone_idx'], mouth_bones)).sum(1)
        mouth_ids = np.flatnonzero(mouth_support > 0)
        self.preserve_native_lips = np.union1d(self.preserve_native_lips, inverse[mouth_ids])
        self.receipt["candidate_vermilion_vertex_ids"] = lip_ids.tolist()
        self.receipt['native_mouth_rig_support_vertex_ids'] = mouth_ids.tolist()
        self.receipt['native_mouth_bone_names'] = [names[i] for i in mouth_bones]
        self.receipt["lip_policy"] = "No nearest-surface attraction in donor mouth rig support or pigment envelope; outer-lip anchors and ARAP, no removal"
        self.receipt['oriented_surface_policy'] = 'Closest complete target triangle in same normal hemisphere, using ARAP-transported source surface normals'

    def closest(self, positions, normals):
        result = np.empty_like(positions)
        for label, surface in self.surfaces.items():
            selected = np.flatnonzero(self.labels == label)
            if len(selected):
                result[selected] = surface.closest_oriented(positions[selected], normals[selected])[0]
        return result
