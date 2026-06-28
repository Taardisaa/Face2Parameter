"""Reference distribution of facial ratios over a sample of labels.json characters.

Builds the population statistics the harmony/fidelity reports score against: per-ratio robust center
(median) + spread (MAD), and a shrunk covariance for a joint Mahalanobis typicality percentile. Only the
cheap deform stages 1-4 run per sample (no skinning), so a few thousand samples build in seconds-minutes.

Cache -> data/hs2_head/ratio_reference.npz (gitignored, alongside the rig data).
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

from .landmarks import mesh_landmarks
from .ratios import ratio_vector

_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "data", "hs2_head")
REF_PATH = os.path.join(_DATA, "ratio_reference.npz")


def sample_label_vectors(path, n, seed):
    """Random sample of n 205-dim normalized vectors from labels.json (whole file loaded once)."""
    rng = random.Random(seed)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = list(data.keys())
    if n and n < len(keys):
        keys = rng.sample(keys, n)
    V = np.array([data[k] for k in keys], dtype=np.float64)
    return V


def build_reference(labels_path="data/labels.json",
                    template_card="tests/HS2ChaF_20240901192905747.png",
                    n=5000, seed=0, subset="HARMONY", out=REF_PATH):
    """Sample labels.json -> ratio matrix -> robust stats + Mahalanobis ecdf; cache to npz."""
    from ..hs2_mesh import _rig, fd_to_inputs
    from ..face_data_utils.utils import FaceData

    rig = _rig()
    fd = FaceData(template_card)                      # reused; set_from_vector overwrites all masked params
    V = sample_label_vectors(labels_path, n, seed)

    rows, names = [], None
    for v in V:
        fd.set_from_vector(np.asarray(v, np.float32), is_simplify=True, without_right=True,
                           denormalize=True, use_gaussian=False)
        shape_face, ab_data = fd_to_inputs(fd)
        vals, names = ratio_vector(mesh_landmarks(rig, shape_face, ab_data), subset)
        rows.append(vals)
    R = np.asarray(rows)                              # (N, K)

    med = np.median(R, axis=0)
    mad = 1.4826 * np.median(np.abs(R - med), axis=0)
    keep = mad > 1e-9                                 # drop degenerate ratios (param barely moves them)
    dropped = [nm for nm, k in zip(names, keep) if not k]
    R, med, mad = R[:, keep], med[keep], mad[keep]
    names = [nm for nm, k in zip(names, keep) if k]

    mean = R.mean(axis=0)
    cov = np.cov(R, rowvar=False)
    eps = 1e-6 * np.trace(cov) / cov.shape[0]         # shrink toward diagonal for a stable inverse
    cov_inv = np.linalg.inv(cov + eps * np.eye(cov.shape[0]))
    d = R - mean
    maha = np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv, d))

    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, ratio_names=np.array(names), median=med, mad=mad, mean=mean,
             cov=cov, cov_inv=cov_inv, maha_sorted=np.sort(maha),
             n_samples=len(R), seed=seed, subset=subset,
             template_name=os.path.basename(template_card), dropped=np.array(dropped))
    return out, names, len(R), dropped


def load_reference(path=REF_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no reference at {path}; build it first:\n"
            f"  .venv/Scripts/python.exe scripts/face_report.py --build-reference")
    z = np.load(path, allow_pickle=True)
    ref = {k: z[k] for k in z.files}
    ref["ratio_names"] = [str(x) for x in ref["ratio_names"]]
    ref["subset"] = str(ref["subset"])
    return ref
