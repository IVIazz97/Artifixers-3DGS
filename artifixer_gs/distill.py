#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ArtiFixer3D with an external 3DGS trainer: build its training scene, then prepare ArtiFixer3D+.

Adapted from ArtiFixer's data_processing/artifixer3d.py, minus the 3DGUT training.

  handoff  COLMAP scene that any 3DGS trainer reads: real anchor photos + ArtiFixer predictions at
           their camera poses, PINHOLE cameras, the source points3D.bin for initialisation, and
           generated_images.txt (the prediction images; ArtiFixer3D supervises them with LPIPS only,
           see FastGS train.py --lpips_images).
  plus     render the PLY the trainer produced on every frame of the prepared split and write
           split_artifixer3d_plus.json for the second ArtiFixer pass (ArtiFixer3D+).
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.spatial.transform import Rotation

from artifixer_gs.colmap import CAMERA_MODEL_NAMES, Camera, read_colmap_extrinsics_binary, read_colmap_intrinsics_binary
from artifixer_gs.render import render_ply
from data_processing.camera_trajectories import (
    applied_transform_matrix,
    camera_intrinsics_for_frame,
    opengl_c2w_to_opencv_w2c,
    write_json,
)

GENERATED_IMAGES_FILENAME = "generated_images.txt"
POSE_TOLERANCE_DEGREES = 1e-3
POSE_TOLERANCE_TRANSLATION = 1e-4


@dataclass(frozen=True)
class PreparedScene:
    scene_id: str
    split_path: Path
    # transforms.json defining frame order, intrinsics and OpenGL c2w poses of every frame.
    transforms_path: Path
    # Prepared COLMAP root with the real photos (split image_root).
    colmap_dir: Path
    prompt_path: Path
    camera_scale: float
    has_gt: bool
    # Frames that keep their real photo; every other frame takes the ArtiFixer prediction.
    selected_indices: list[int]
    target_indices_path: Path | None
    frame_count: int


def resolve(split_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else split_root / path


def load_prepared_scene(split_path: Path, scene_id: str | None = None) -> PreparedScene:
    split_path = split_path.resolve()
    scenes = json.loads(split_path.read_text())["test"]
    if scene_id is None:
        assert len(scenes) == 1, f"--scene_id is required when {split_path} has {len(scenes)} scenes"
        scene_id = next(iter(scenes))
    meta = scenes[scene_id]
    root = split_path.parent
    transforms_path = resolve(root, meta["transforms_path"])
    selected_indices = json.loads(resolve(root, meta["selected_indices_path"]).read_text())
    return PreparedScene(
        scene_id=scene_id,
        split_path=split_path,
        transforms_path=transforms_path,
        colmap_dir=resolve(root, meta["image_root"]),
        prompt_path=resolve(root, meta["prompt_path"]),
        camera_scale=float(meta["camera_scale"]),
        has_gt=bool(meta.get("has_gt", True)),
        selected_indices=[int(index) for index in selected_indices],
        target_indices_path=resolve(root, meta["target_indices_path"]) if "target_indices_path" in meta else None,
        frame_count=len(json.loads(transforms_path.read_text())["frames"]),
    )


def generated_frame_indices(scene: PreparedScene) -> list[int]:
    selected = set(scene.selected_indices)
    assert len(selected) == len(scene.selected_indices), f"Duplicate selected indices: {scene.selected_indices}"
    return [index for index in range(scene.frame_count) if index not in selected]


def pinhole_camera(camera_id: int, width: int, height: int, fx: float, fy: float, cx: float, cy: float) -> Camera:
    return Camera(id=camera_id, model="PINHOLE", width=int(width), height=int(height), params=np.array([fx, fy, cx, cy], dtype=np.float64))


def pinhole_from_colmap(camera_id: int, camera: Camera, image_size: tuple[int, int]) -> Camera:
    """Source COLMAP camera scaled to image_size, as PINHOLE (3DGS trainers take no distortion)."""
    sx, sy = image_size[0] / camera.width, image_size[1] / camera.height
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        fx, fy, distortion = f, f, []
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        distortion = []
    elif camera.model in ("SIMPLE_RADIAL", "RADIAL"):
        f, cx, cy, *distortion = camera.params
        fx = fy = f
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, *distortion = camera.params
    else:
        assert False, f"Unsupported COLMAP camera model: {camera.model}"
    assert not any(distortion), f"Camera {camera.id} has distortion {distortion}; undistort the scene first"
    return pinhole_camera(camera_id, image_size[0], image_size[1], fx * sx, fy * sy, cx * sx, cy * sy)


def colmap_qvec(rotation: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(rotation).as_quat()
    qvec = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)
    return (qvec if qvec[0] >= 0 else -qvec) / np.linalg.norm(qvec)


def write_cameras(path: Path, cameras: list[Camera]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            fid.write(struct.pack("<iiQQ", camera.id, CAMERA_MODEL_NAMES[camera.model].model_id, camera.width, camera.height))
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def write_images(path: Path, images: list[tuple[int, np.ndarray, np.ndarray, int, str]]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id, qvec, tvec, camera_id, name in images:
            fid.write(struct.pack("<idddddddi", image_id, *qvec, *tvec, camera_id))
            fid.write(name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", 0))


def shared_camera_id(cameras: list[Camera], ids: dict[tuple, int], camera: Camera) -> int:
    key = (camera.width, camera.height, tuple(np.round(camera.params, 6)))
    if key not in ids:
        ids[key] = camera.id
        cameras.append(camera)
    return ids[key]


def link_or_resize(source: Path, target: Path, size: tuple[int, int]) -> None:
    """Symlink source, or write a resized copy when it is not size (ArtiFixer outputs its own resolution)."""
    if not source.is_file():
        raise FileNotFoundError(f"Missing frame: {source}")
    with PILImage.open(source) as image:
        if image.size == size:
            target.symlink_to(source.resolve())
        else:
            image.convert("RGB").resize(size, resample=PILImage.Resampling.LANCZOS).save(target)


def handoff(scene: PreparedScene, artifixer_frames_dir: Path, output_dir: Path) -> Path:
    """Write the trainer's COLMAP scene: anchors keep their exact source pose, predictions take the split pose."""
    generated = generated_frame_indices(scene)
    assert generated, "Every frame is an anchor: there is nothing to distill"
    missing = [index for index in generated if not (artifixer_frames_dir / f"{index:05d}.png").is_file()]
    assert not missing, f"Missing ArtiFixer predictions in {artifixer_frames_dir}: {missing[:10]} ({len(missing)} missing)"

    transforms = json.loads(scene.transforms_path.read_text())
    frames = transforms["frames"]
    applied_transform = applied_transform_matrix(transforms)
    sparse_dir = scene.colmap_dir / "sparse" / "0"
    source_images = {Path(image.name).name: image for image in read_colmap_extrinsics_binary(sparse_dir / "images.bin")}
    source_cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")

    if output_dir.exists():
        assert output_dir.is_dir() and not output_dir.is_symlink(), f"Expected a directory: {output_dir}"
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "sparse" / "0").mkdir(parents=True)

    selected = set(scene.selected_indices)
    cameras: list[Camera] = []
    camera_ids: dict[tuple, int] = {}
    images, generated_names = [], []
    for index, frame in enumerate(frames):
        w2c = opengl_c2w_to_opencv_w2c(frame["transform_matrix"], applied_transform)
        if index in selected:
            source = scene.colmap_dir / str(frame["file_path"])
            source_image = source_images[Path(str(frame["file_path"])).name]
            with PILImage.open(source) as image:
                size = image.size
            camera = pinhole_from_colmap(len(cameras) + 1, source_cameras[source_image.camera_id], size)
            qvec, tvec = source_image.qvec, source_image.tvec
            # The split pose must round-trip to the source COLMAP pose, or the conventions disagree.
            error = (Rotation.from_quat(np.roll(colmap_qvec(w2c[:3, :3]), -1)) * Rotation.from_quat(np.roll(qvec, -1)).inv()).magnitude()
            assert np.degrees(error) < POSE_TOLERANCE_DEGREES and np.linalg.norm(w2c[:3, 3] - tvec) < POSE_TOLERANCE_TRANSLATION, (
                f"Anchor frame {index} ({source.name}) does not match its COLMAP pose; regenerate the split"
            )
            name = f"frame_{index:05d}{source.suffix.lower()}"
            (output_dir / "images" / name).symlink_to(source.resolve())
        else:
            intrinsics = camera_intrinsics_for_frame(transforms, frame)
            assert not any(intrinsics[key] for key in ("k1", "k2", "p1", "p2")), f"Frame {index} has lens distortion"
            camera = pinhole_camera(
                len(cameras) + 1, intrinsics["w"], intrinsics["h"], intrinsics["fl_x"], intrinsics["fl_y"], intrinsics["cx"], intrinsics["cy"]
            )
            qvec, tvec = colmap_qvec(w2c[:3, :3]), w2c[:3, 3]
            name = f"frame_{index:05d}.png"
            link_or_resize(artifixer_frames_dir / f"{index:05d}.png", output_dir / "images" / name, (camera.width, camera.height))
            generated_names.append(name)
        camera_id = shared_camera_id(cameras, camera_ids, camera)
        images.append((index + 1, qvec, tvec, camera_id, name))

    write_cameras(output_dir / "sparse" / "0" / "cameras.bin", cameras)
    write_images(output_dir / "sparse" / "0" / "images.bin", images)
    shutil.copyfile((sparse_dir / "points3D.bin").resolve(), output_dir / "sparse" / "0" / "points3D.bin")
    (output_dir / GENERATED_IMAGES_FILENAME).write_text("\n".join(generated_names) + "\n")
    write_json(output_dir / "selected_indices.json", scene.selected_indices)
    print(
        f"Wrote {output_dir}: {len(selected)} photos + {len(generated_names)} ArtiFixer frames, "
        f"{len(cameras)} camera(s); LPIPS-only list {output_dir / GENERATED_IMAGES_FILENAME}",
        flush=True,
    )
    return output_dir


def path_for_split(split_path: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(split_path.parent.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def plus(scene: PreparedScene, ply: Path, render_dir: Path, output_split: Path) -> Path:
    """Render ply on every split frame and write the ArtiFixer3D+ split (same anchors, same targets)."""
    render_ply(ply=ply, trajectory_path=scene.transforms_path, output_dir=render_dir, selected_indices=scene.selected_indices)
    entry = {
        "scene_id": scene.scene_id,
        "transforms_path": path_for_split(output_split, scene.transforms_path),
        "image_root": path_for_split(output_split, scene.colmap_dir),
        "render_dir": path_for_split(output_split, render_dir / "renders"),
        "opacity_dir": path_for_split(output_split, render_dir / "opacity"),
        "selected_indices_path": path_for_split(output_split, render_dir / "selected_indices.json"),
        "prompt_path": path_for_split(output_split, scene.prompt_path),
        "reconstruction_checkpoint": path_for_split(output_split, ply),
        "camera_scale": scene.camera_scale,
    }
    if scene.target_indices_path is not None:
        entry["target_indices_path"] = path_for_split(output_split, scene.target_indices_path)
    if not scene.has_gt:
        entry["has_gt"] = False
    write_json(output_split, {"test": {scene.scene_id: entry}})
    print(f"Wrote ArtiFixer3D+ split {output_split}", flush=True)
    return output_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    h = sub.add_parser("handoff", help="Build the 3DGS trainer's COLMAP scene from photos + ArtiFixer frames.")
    h.add_argument("--split_path", type=Path, required=True, help="split.json written by artifixer_gs.prepare.")
    h.add_argument("--artifixer_frames_dir", type=Path, required=True, help="<save_dir>/<scene>/frames/batch_0000/pred")
    h.add_argument("--output_dir", type=Path, required=True)
    h.add_argument("--scene_id", default=None)
    p = sub.add_parser("plus", help="Render the trained PLY and write the ArtiFixer3D+ split.")
    p.add_argument("--split_path", type=Path, required=True)
    p.add_argument("--ply", type=Path, required=True, help="PLY trained on the handoff scene.")
    p.add_argument("--render_dir", type=Path, required=True)
    p.add_argument("--output_split", type=Path, required=True)
    p.add_argument("--scene_id", default=None)
    args = parser.parse_args()

    scene = load_prepared_scene(args.split_path, args.scene_id)
    if args.command == "handoff":
        handoff(scene, args.artifixer_frames_dir.resolve(), args.output_dir.resolve())
    else:
        plus(scene, args.ply.resolve(), args.render_dir.resolve(), args.output_split.resolve())


if __name__ == "__main__":
    main()
