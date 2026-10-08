"""Project an original photograph onto an unchanged MICA mesh for visual review.

Only camera parameters are fitted. A source-view texture match is construction,
not evidence that the head shape is correct. Unobserved faces stay gray.
Run in the existing WSL SMIRK environment.
"""
import argparse
import json
from pathlib import Path
import pickle

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
import torch
from pytorch3d.renderer import (
    BlendParams, FoVOrthographicCameras, HardPhongShader, Materials,
    MeshRasterizer, MeshRenderer, PointLights, RasterizationSettings,
    TexturesUV, TexturesVertex, look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.utils.camera_conversions import cameras_from_opencv_projection

from .artifact import ModelArtifact, host_path, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--image-index', type=int, default=1)
    p.add_argument('--face-mask', type=Path, required=True,
                   help='Trusted, topology-compatible authored FLAME_masks.pkl; texture region only')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    model = ModelArtifact(args.manifest, args.image_index)
    vertices, faces = model.mesh(head_local=True)
    path = host_path(model.image['input'])
    if sha(path) != model.image['input_sha256']:
        raise ValueError('Original photograph changed')
    args.out.mkdir(parents=True, exist_ok=False)
    original = np.asarray(Image.open(path).convert('RGB'))
    x0, y0, x1, y1 = model.arrays['detected_bbox']
    side = int(np.ceil(max(x1-x0, y1-y0)*1.55))
    origin = np.floor([(x0+x1-side)/2, (y0+y1-side)/2]).astype(int)
    photo = np.asarray(Image.fromarray(original).crop((*origin, *(origin+side))))
    target = model.arrays['detected_landmarks_5'].astype(float) - origin
    landmarks = model.arrays['geometry__landmarks_fan_3d'][0]
    points = np.vstack([landmarks[36:42].mean(0), landmarks[42:48].mean(0),
                        landmarks[30], landmarks[48], landmarks[54]]).astype(float)
    # Explicit approximate eye-center correspondence; five fitted anchors do
    # not become held-out accuracy measurements.
    center = side/2
    def intrinsics(focal):
        return np.array([[focal, 0, center], [0, focal, center], [0, 0, 1.]])
    focal = 2*side
    distance = focal*np.linalg.norm(points[0]-points[1])/np.linalg.norm(target[0]-target[1])
    initial = np.r_[np.pi, 0., 0., 0., 0., distance, np.log(focal)]
    def residual(parameters):
        predicted, _ = cv2.projectPoints(points, parameters[:3], parameters[3:6],
                                         intrinsics(np.exp(parameters[6])), None)
        return (predicted[:, 0]-target).ravel()
    fit = least_squares(residual, initial,
        bounds=(np.r_[[-2*np.pi]*3, -2., -2., .05, np.log(.8*side)],
                np.r_[[2*np.pi]*3, 2., 2., 3., np.log(8*side)]), max_nfev=250)
    if not fit.success:
        raise RuntimeError('Camera registration did not converge')
    rotation, _ = cv2.Rodrigues(fit.x[:3])
    translation = fit.x[3:6]; k = intrinsics(np.exp(fit.x[6]))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float32, device=device)
    camera = cameras_from_opencv_projection(tensor(rotation[None]), tensor(translation[None]),
        tensor(k[None]), tensor([[side, side]]))
    v = tensor(vertices)
    f = torch.as_tensor(faces, dtype=torch.int64, device=device)
    settings = RasterizationSettings(image_size=(side, side), faces_per_pixel=1,
                                     blur_radius=0., perspective_correct=True)
    plain = Meshes(verts=[v], faces=[f], textures=TexturesVertex(verts_features=[torch.full_like(v, .75)]))
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=settings)
    with torch.no_grad():
        fragments = rasterizer(plain)
    mask = fragments.pix_to_face[0, :, :, 0].cpu().numpy() >= 0
    visible_ids = torch.unique(fragments.pix_to_face[fragments.pix_to_face >= 0])
    # Keep every face; UV seam expansion lets hidden faces use a gray swatch.
    projected, _ = cv2.projectPoints(vertices.astype(float), fit.x[:3], translation, k, None)
    uv = projected[:, 0]/(side-1)
    uv[:, 1] = 1-uv[:, 1]
    textured_faces = f.clone()
    visible = torch.zeros(len(f), dtype=torch.bool, device=device); visible[visible_ids] = True
    with args.face_mask.open('rb') as stream:
        authored_mask = pickle.load(stream, encoding='latin1')
    face_vertices = np.unique(np.concatenate([np.asarray(authored_mask[key], dtype=int)
        for key in ['face', 'left_eyeball', 'right_eyeball']]))
    if face_vertices.min() < 0 or face_vertices.max() >= len(vertices):
        raise ValueError('Authored texture mask is outside original topology')
    in_face = np.zeros(len(vertices), dtype=bool); in_face[face_vertices] = True
    face_region = in_face[faces].all(1)
    visible &= torch.as_tensor(face_region, device=device)
    textured_faces[~visible] = len(vertices)
    texture = photo.astype(np.float32)/255
    texture[:3, :3] = .65
    uv = np.vstack([uv, [1/(side-1), 1-1/(side-1)]])
    textured = Meshes(verts=[v], faces=[f], textures=TexturesUV(
        maps=tensor(texture[None]), faces_uvs=textured_faces[None], verts_uvs=tensor(uv[None])))
    # Flat texture rendering has no added lighting: source lighting is baked in.
    flat = PointLights(device=device, ambient_color=((1., 1., 1.),),
        diffuse_color=((0., 0., 0.),), specular_color=((0., 0., 0.),))
    lights = PointLights(device=device, location=[[.3, .4, 1.]],
        ambient_color=((.55, .55, .55),), diffuse_color=((.65, .65, .65),),
        specular_color=((0., 0., 0.),))
    material = Materials(device=device, specular_color=((0., 0., 0.),))
    blend = BlendParams(background_color=(.94, .94, .92))
    def render(mesh, cam, light, size):
        renderer = MeshRenderer(MeshRasterizer(cameras=cam, raster_settings=RasterizationSettings(
            image_size=size, faces_per_pixel=1, blur_radius=0., perspective_correct=True)),
            HardPhongShader(device=device, cameras=cam, lights=light, materials=material, blend_params=blend))
        with torch.no_grad():
            return renderer(mesh)[0, ..., :3].cpu().numpy()
    gray = render(plain, camera, lights, (side, side))
    painted = render(textured, camera, flat, (side, side))
    overlay = photo.astype(float)/255
    overlay[mask] = .55*overlay[mask] + .45*gray[mask]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay8 = np.uint8(np.clip(overlay, 0, 1)*255)
    cv2.drawContours(overlay8, contours, -1, (0, 230, 210), max(1, side//250))
    r, t = look_at_view_transform(dist=1., elev=0., azim=35., at=((0., .02, -.035),), device=device)
    alternate_camera = FoVOrthographicCameras(device=device, R=r, T=t,
        min_x=-.145, max_x=.145, min_y=-.145, max_y=.145)
    alternate = render(textured, alternate_camera, flat, 512)
    font = FontProperties(fname='/mnt/c/Windows/Fonts/msyh.ttc')
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), facecolor='#f0f0eb')
    titles = ['原照片脸部区域', '固定模型叠图：青线为模型边界', '仅面部投影贴图：原相机', '同一贴图转到 35°：灰色无贴图']
    for ax, img, title in zip(axes, [photo, overlay8, painted, alternate], titles):
        ax.imshow(img); ax.set_title(title, fontproperties=font, fontsize=10); ax.axis('off')
    fig.suptitle('img-002 原始 MICA 网格：只估计相机，不调整头部形状', fontproperties=font)
    fig.text(.5, .04, '原相机贴图相似是投影构造，不能证明骨相正确；五点相机为估计，非真实标定',
        ha='center', fontproperties=font, fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .95)); fig.savefig(args.out/'projection_board.png', dpi=140); plt.close(fig)
    report = dict(manifest_sha256=sha(args.manifest), image_index=args.image_index,
        photo_sha256=model.image['input_sha256'], raw_mesh_sha256=model.image['sha256'],
        camera_matrix=k.tolist(), rotation=rotation.tolist(), translation=translation.tolist(),
        five_point_reprojection_residual=residual(fit.x).reshape(-1, 2).tolist(),
        anchor_correspondence='FLAME68 eye-ring means, nose30, mouth48/54 to detector five points; approximate eye centers',
        shape_changed=False, faces_deleted=0, camera_calibrated=False,
        source_view_texture_match_is_validation=False, unobserved_triangles='gray',
        texture_mask=dict(path=str(args.face_mask), sha256=sha(args.face_mask),
            meaning='Authored FLAME face plus eyeballs; texture-only restriction, not photo segmentation'),
        focal_at_bound=bool(np.any(fit.active_mask[6:])),
        texture='Projective UVs from original image, source illumination retained; not albedo',
        silhouette_independent=False, accuracy_certified=False, game_mutations=0)
    (args.out/'receipt.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(board=str(args.out/'projection_board.png'), shape_changed=False)))


if __name__ == '__main__':
    main()
