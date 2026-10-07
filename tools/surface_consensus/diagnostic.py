"""Pillow projection overlays for independent visual review, never labels."""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.surface_calibration.pixel_certificate import digest


def render_support_diagnostics(definition, observations, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 11)
    images, crops, records = [], [], []
    for view, observation in zip(definition["views"], observations, strict=True):
        source = Path(observation["path"])
        with Image.open(source) as original:
            annotated = original.convert("RGB")
        xy = view["projection"]["xy"]
        if xy is None:
            continue
        x, y = xy
        draw = ImageDraw.Draw(annotated)
        color = (0, 255, 255) if view["status"] == "projected_without_independent_observation" else (255, 80, 60)
        draw.line([(x-5, y), (x+5, y)], fill=(0, 0, 0), width=3)
        draw.line([(x, y-5), (x, y+5)], fill=(0, 0, 0), width=3)
        draw.line([(x-4, y), (x+4, y)], fill=color, width=1)
        draw.line([(x, y-4), (x, y+4)], fill=color, width=1)
        # Small ID rather than a long semantic label on the face.
        marker_id = "S" + str(definition["maximizing_vertex_ids"][0])
        draw.text((x+7, y-14), marker_id, font=small,
                  fill=color, stroke_width=1, stroke_fill=(0, 0, 0))
        draw.text((8, 8), f"view {observation['view_index']} yaw {observation['yaw']:+.0f} | geometry-only projection", font=font,
                  fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        destination = output / f"view_{observation['view_index']}_support_overlay.png"
        annotated.save(destination)
        left, top = math.floor(x) - 64, math.floor(y) - 64
        crop = annotated.crop((left, top, left+128, top+128))
        crop_path = output / f"view_{observation['view_index']}_support_crop128.png"
        crop.save(crop_path)
        images.append(annotated)
        crops.append(crop)
        records.append({"view_index": observation["view_index"], "yaw": observation["yaw"],
                        "source_png": str(source.resolve()), "source_png_sha256": digest(source),
                        "projected_xy_top_left_integer_centers": xy, "geometry_status": view["status"],
                        "overlay_path": str(destination.resolve()), "overlay_sha256": digest(destination),
                        "crop_path": str(crop_path.resolve()), "crop_sha256": digest(crop_path),
                        "crop_bounds_ltrb_exclusive": [left, top, left+128, top+128]})
    full = Image.new("RGB", (4*512, 2*512), (30, 30, 30))
    strip = Image.new("RGB", (128*len(crops), 160), (30, 30, 30))
    strip_draw = ImageDraw.Draw(strip)
    for i, (image, crop, record) in enumerate(zip(images, crops, records, strict=True)):
        full.paste(image, ((i % 4)*512, (i // 4)*512))
        strip.paste(crop, (i*128, 32))
        strip_draw.text((i*128+5, 3), f"v{record['view_index']} {record['yaw']:+.0f} deg", font=font, fill=(255, 255, 255))
        strip_draw.text((i*128+5, 18), "geometry only", font=small, fill=(255, 255, 255))
    full_path, strip_path = output / "support_7view_overlay_sheet.png", output / "support_7view_crop128_strip.png"
    full.save(full_path)
    strip.save(strip_path)
    return {"kind": "projected_material_point_visual_review_diagnostics",
            "anatomical_correspondence_validated": False, "independent_semantic_annotation": False,
            "original_pngs_modified": False, "point_id": definition["id"], "views": records,
            "sheet_path": str(full_path.resolve()), "sheet_sha256": digest(full_path),
            "crop_strip_path": str(strip_path.resolve()), "crop_strip_sha256": digest(strip_path),
            "marker_legend": "cyan: no geometric blocker; red: geometric visibility/clipping rejection. Neither is material visibility or anatomical evidence."}
