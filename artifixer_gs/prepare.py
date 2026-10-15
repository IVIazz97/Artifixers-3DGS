#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a COLMAP scene and a trained 3DGS PLY for ArtiFixer inference.

Adapted from ArtiFixer's data_processing/prepare_colmap_artifixer_inputs.py: the 3DGUT training is
gone. The Gaussians come from any method that writes the standard 3DGS PLY (FastGS, 3DGS, ...) and
are rendered with gsplat. Phases:

  prepare  link the photos, write sparse/0 and nerfstudio/transforms.json (OpenGL c2w)
  render   render the PLY: every photo frame (anchors = --selected_image_names_file, targets = the
           rest), or the novel --trajectory_path (targets = the path, anchors = the selected photos)
  scale    metric scale with MoGe-2 (or --metric_scale)
  caption  caption.h5 from --caption text, or the Qwen3-VL captioner

The PLY must live in the world frame of --colmap_dir (true when it was trained on that COLMAP).
split.json is written once renders, scale and caption exist; model_eval.run_inference reads it.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from artifixer_gs.colmap import (
    CAMERA_MODEL_NAMES,
    Camera,
    Image,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_intrinsics_binary,
)
from artifixer_gs.render import render_outputs_complete, render_ply
from data_processing.camera_trajectories import (
    assert_target_only_trajectory,
    camera_intrinsics_for_frame,
    frame_with_camera_intrinsics,
    opencv_w2c_to_opengl_c2w,
    read_camera_trajectory,
    trajectory_frame_count,
    transforms_json,
    write_json,
)

DEFAULT_TEXT_ENCODER_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DEFAULT_CAPTIONING_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
DEFAULT_CAPTION_FILENAME = "caption.h5"
DEFAULT_CAMERA_CONDITIONING_SCALE = 0.01
SUPPORTED_CAMERA_MODELS = {"SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV"}

# Canonical pose contract for prepared ArtiFixer reconstructed_colmap splits: transforms.json stores
# NeRFStudio/OpenGL C2W transform_matrix values converted from the COLMAP OpenCV W2C poses.


@dataclass(frozen=True)
class ColmapScene:
    cameras: list[Camera]
    images: list[Image]


@dataclass(frozen=True)
class PreparedPaths:
    scene_id: str
    scene_root: Path
    # Prepared photos + sparse/0 + nerfstudio/transforms.json; the split's image_root.
    input_dir: Path
    # Renders of every photo frame (sparse-anchor mode).
    render_dir: Path
    prompt_dir: Path
    split_path: Path
    selected_indices_path: Path
    scale_dir: Path


@dataclass(frozen=True)
class PreparedTrajectory:
    root: Path
    transforms_path: Path
    render_trajectory_path: Path
    selected_indices_path: Path
    target_indices_path: Path
    render_dir: Path


def prepared_paths(scene_root: Path, scene_id: str) -> PreparedPaths:
    return PreparedPaths(
        scene_id=scene_id,
        scene_root=scene_root,
        input_dir=scene_root / "input" / scene_id,
        render_dir=scene_root / "renders" / "photos",
        prompt_dir=scene_root / "captions",
        split_path=scene_root / "split.json",
        selected_indices_path=scene_root / "selected_indices.json",
        scale_dir=scene_root / "metric_alignment",
    )


def resolve_colmap_paths(colmap_dir: Path) -> tuple[Path, Path]:
    colmap_dir = colmap_dir.resolve()
    image_dir = colmap_dir / "images"
    sparse_dir = colmap_dir / "sparse" / "0"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Missing COLMAP images directory: {image_dir}")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (sparse_dir / name).is_file():
            raise FileNotFoundError(f"Missing COLMAP sparse file: {sparse_dir / name}")
    return image_dir, sparse_dir


def read_colmap_scene(sparse_dir: Path) -> ColmapScene:
    cameras = read_colmap_intrinsics_binary(sparse_dir / "cameras.bin")
    return ColmapScene(
        cameras=[cameras[camera_id] for camera_id in sorted(cameras)],
        images=read_colmap_extrinsics_binary(sparse_dir / "images.bin"),
    )


def scene_relative_path(paths: PreparedPaths, path: Path) -> str:
    scene_root = paths.scene_root.resolve()
    path = path.resolve()
    try:
        return path.relative_to(scene_root).as_posix()
    except ValueError:
        return str(path)


def read_selected_image_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def resolve_selected_indices(args: argparse.Namespace, images: Sequence[Image]) -> list[int]:
    if args.selected_image_names_file is not None:
        name_to_index = {Path(image.name).name: index for index, image in enumerate(images)}
        names = read_selected_image_names(args.selected_image_names_file)
        basenames = [Path(name).name for name in names]
        missing = [name for name in basenames if name not in name_to_index]
        assert not missing, f"Selected image names are not in the COLMAP model: {missing}"
        selected = [name_to_index[name] for name in basenames]
    else:
        selected = list(range(len(images)))

    assert selected, "At least one anchor view is required"
    assert len(set(selected)) == len(selected), f"Selected indices contain duplicates: {selected}"
    assert min(selected) >= 0 and max(selected) < len(
        images
    ), f"Selected indices must be in [0, {len(images) - 1}], got {selected}"
    return selected


def require_unique_basenames(images: Sequence[Image]) -> None:
    basenames = [Path(image.name).name for image in images]
    assert len(set(basenames)) == len(basenames), "COLMAP image basenames must be unique"


def source_image_path(image_dir: Path, image: Image) -> Path:
    for candidate in (image_dir / image.name, image_dir / Path(image.name).name):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find image for COLMAP entry {image.name!r} under {image_dir}")


def scaled_camera_params(camera: Camera, sx: float, sy: float) -> np.ndarray:
    params = camera.params.astype(np.float64).copy()
    if camera.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        assert np.isclose(sx, sy), f"Camera {camera.id} requires uniform scaling for {camera.model}"
        params[0] *= sx
        params[1] *= sx
        params[2] *= sy
    elif camera.model in {"PINHOLE", "OPENCV"}:
        params[0] *= sx
        params[1] *= sy
        params[2] *= sx
        params[3] *= sy
    else:
        assert False, f"Unsupported camera model for image-resolution scaling: {camera.model}"
    return params


def scale_image_observations(image: Image, sx: float, sy: float) -> Image:
    xys = image.xys.astype(np.float64).copy()
    if len(xys):
        xys[:, 0] *= sx
        xys[:, 1] *= sy
    return image._replace(xys=xys)


def scale_colmap_scene_to_images(image_dir: Path, scene: ColmapScene) -> ColmapScene:
    camera_by_id = {camera.id: camera for camera in scene.cameras}
    image_info_by_camera: dict[int, tuple[int, int, str, Path]] = {}
    for image in scene.images:
        assert image.camera_id in camera_by_id, f"COLMAP image {image.name!r} references missing camera {image.camera_id}"
        image_path = source_image_path(image_dir, image)
        with PILImage.open(image_path) as pil_image:
            width, height = pil_image.size
        previous = image_info_by_camera.setdefault(image.camera_id, (width, height, image.name, image_path))
        assert previous[:2] == (width, height), (
            f"COLMAP camera {image.camera_id} is shared by images with different sizes: "
            f"{previous[0]}x{previous[1]} and {width}x{height}"
        )

    scaled_cameras = []
    scale_by_camera: dict[int, tuple[float, float]] = {}
    for camera in scene.cameras:
        image_info = image_info_by_camera.get(camera.id)
        if image_info is None or image_info[:2] == (int(camera.width), int(camera.height)):
            scaled_cameras.append(camera)
            scale_by_camera[camera.id] = (1.0, 1.0)
            continue
        width, height, image_name, image_path = image_info
        sx = width / float(camera.width)
        sy = height / float(camera.height)
        if not np.isclose(sx, sy, rtol=1e-2, atol=1e-3):
            raise AssertionError(
                f"COLMAP camera/image size mismatch for {image_name!r}: "
                f"camera {camera.id} is {int(camera.width)}x{int(camera.height)}, "
                f"but {image_path} is {width}x{height}"
            )
        sx = sy = float((sx + sy) * 0.5)
        print(
            f"Scaling COLMAP camera {camera.id} from {int(camera.width)}x{int(camera.height)} "
            f"to {width}x{height} to match input images",
            flush=True,
        )
        scaled_cameras.append(camera._replace(width=width, height=height, params=scaled_camera_params(camera, sx, sy)))
        scale_by_camera[camera.id] = (sx, sy)

    scaled_images = [scale_image_observations(image, *scale_by_camera[image.camera_id]) for image in scene.images]
    return ColmapScene(cameras=scaled_cameras, images=scaled_images)


def normalize_near_shared_camera_scene(scene: ColmapScene) -> ColmapScene:
    camera_by_id = {camera.id: camera for camera in scene.cameras}
    used_cameras = [camera_by_id[camera_id] for camera_id in sorted({image.camera_id for image in scene.images})]
    if len(used_cameras) <= 1:
        return scene

    models = {camera.model for camera in used_cameras}
    assert len(models) == 1, f"Cannot normalize mixed camera models: {sorted(models)}"
    model = next(iter(models))
    assert model in SUPPORTED_CAMERA_MODELS, f"Unsupported camera model for shared-camera normalization: {model}"

    widths = np.array([float(camera.width) for camera in used_cameras], dtype=np.float64)
    heights = np.array([float(camera.height) for camera in used_cameras], dtype=np.float64)
    assert widths.max() / widths.min() <= 1.05 and heights.max() / heights.min() <= 1.05, (
        "Refusing to normalize cameras with image dimensions differing by more than 5%"
    )

    target_width = int(round(np.median(widths)))
    target_height = int(round(np.median(heights)))
    scaled_params = []
    scale_by_camera: dict[int, tuple[float, float]] = {}
    for camera in used_cameras:
        sx = target_width / float(camera.width)
        sy = target_height / float(camera.height)
        scale_by_camera[camera.id] = (sx, sy)
        scaled_params.append(scaled_camera_params(camera, sx, sy))

    normalized_camera = used_cameras[0]._replace(
        id=1, width=target_width, height=target_height, params=np.median(np.stack(scaled_params), axis=0)
    )
    normalized_images = [
        scale_image_observations(image, *scale_by_camera[image.camera_id])._replace(camera_id=1)
        for image in scene.images
    ]
    print(f"Normalizing {len(used_cameras)} near-shared cameras to one {target_width}x{target_height} {model} camera", flush=True)
    return ColmapScene(cameras=[normalized_camera], images=normalized_images)


def write_colmap_cameras(path: Path, cameras: Sequence[Camera]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera in cameras:
            model = CAMERA_MODEL_NAMES[camera.model]
            fid.write(struct.pack("<iiQQ", camera.id, model.model_id, int(camera.width), int(camera.height)))
            fid.write(struct.pack("<" + "d" * len(camera.params), *camera.params))


def write_colmap_images(path: Path, images: Sequence[Image]) -> None:
    with path.open("wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image in images:
            fid.write(struct.pack("<idddddddi", image.id, *image.qvec, *image.tvec, image.camera_id))
            fid.write(image_basename(image).encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.xys)))
            for xy, point3d_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<ddq", float(xy[0]), float(xy[1]), int(point3d_id)))


def image_basename(image: Image) -> str:
    return Path(image.name).name


def reset_prepared_scene(path: Path) -> None:
    if path.exists():
        assert path.is_dir() and not path.is_symlink(), f"Expected prepared scene directory, got {path}"
        shutil.rmtree(path)
    (path / "images").mkdir(parents=True)
    (path / "sparse/0").mkdir(parents=True)


def prepare_images(
    source_image_dir: Path,
    target_image_dir: Path,
    images: Sequence[Image],
    cameras: Sequence[Camera],
    downscale_factor: int,
) -> None:
    """Symlink the photos, or write resized copies when the camera size differs from the file."""
    assert downscale_factor >= 1, f"Expected downscale factor >= 1, got {downscale_factor}"
    camera_by_id = {camera.id: camera for camera in cameras}
    for image in images:
        camera = camera_by_id[image.camera_id]
        source = source_image_path(source_image_dir, image).resolve()
        target = target_image_dir / image_basename(image)
        size = (
            max(1, int(round(float(camera.width) / downscale_factor))),
            max(1, int(round(float(camera.height) / downscale_factor))),
        )
        with PILImage.open(source) as pil_image:
            if pil_image.size == size:
                target.symlink_to(source)
            else:
                pil_image.resize(size, resample=PILImage.Resampling.LANCZOS).save(target)


def write_sparse_model(source_sparse_dir: Path, target_sparse_dir: Path, scene: ColmapScene) -> None:
    write_colmap_cameras(target_sparse_dir / "cameras.bin", scene.cameras)
    write_colmap_images(target_sparse_dir / "images.bin", scene.images)
    (target_sparse_dir / "points3D.bin").symlink_to((source_sparse_dir / "points3D.bin").resolve())


def image_to_nerfstudio_transform(image: Image) -> list[list[float]]:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = qvec_to_so3(image.qvec)
    world_to_camera[:3, 3] = image.tvec
    return opencv_w2c_to_opengl_c2w(world_to_camera).tolist()


def camera_intrinsics(camera: Camera) -> dict[str, float | int]:
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params
        values = {"fl_x": f, "fl_y": f, "cx": cx, "cy": cy}
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params
        values = {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy}
    elif camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = camera.params
        values = {"fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "k1": k1}
    elif camera.model == "RADIAL":
        f, cx, cy, k1, k2 = camera.params
        values = {"fl_x": f, "fl_y": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2}
    elif camera.model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = camera.params
        values = {"fl_x": fx, "fl_y": fy, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "p1": p1, "p2": p2}
    else:
        assert False, f"Unsupported camera model for ArtiFixer transforms: {camera.model}"
    return {
        "camera_model": "OPENCV",
        "w": int(camera.width),
        "h": int(camera.height),
        **{key: float(value) for key, value in values.items()},
    }


def shared_camera_intrinsics(scene: ColmapScene) -> dict[str, float | int]:
    camera_by_id = {camera.id: camera for camera in scene.cameras}
    used_cameras = [camera_by_id[camera_id] for camera_id in sorted({image.camera_id for image in scene.images})]
    unsupported = sorted({camera.model for camera in used_cameras} - SUPPORTED_CAMERA_MODELS)
    assert not unsupported, f"Unsupported camera models for ArtiFixer transforms: {unsupported}"

    intrinsics = camera_intrinsics(used_cameras[0])
    for camera in used_cameras[1:]:
        assert camera_intrinsics(camera) == intrinsics, "ArtiFixer COLMAP prep expects one shared intrinsic calibration"
    return intrinsics


def source_frame(image: Image) -> dict[str, object]:
    return {
        "file_path": f"images/{image_basename(image)}",
        "transform_matrix": image_to_nerfstudio_transform(image),
    }


def write_transforms_json(path: Path, scene: ColmapScene) -> None:
    write_json(path, transforms_json(shared_camera_intrinsics(scene), [source_frame(image) for image in scene.images]))


def eval_split_entry(
    paths: PreparedPaths,
    *,
    transforms_path: Path,
    render_dir: Path,
    metric_scale: float,
    ply: Path,
    target_indices_path: Path | None = None,
    has_gt: bool = True,
) -> dict[str, object]:
    entry = {
        "scene_id": paths.scene_id,
        "transforms_path": scene_relative_path(paths, transforms_path),
        "image_root": scene_relative_path(paths, paths.input_dir),
        "render_dir": scene_relative_path(paths, render_dir / "renders"),
        "opacity_dir": scene_relative_path(paths, render_dir / "opacity"),
        "selected_indices_path": scene_relative_path(paths, render_dir / "selected_indices.json"),
        "prompt_path": scene_relative_path(paths, paths.prompt_dir / paths.scene_id / DEFAULT_CAPTION_FILENAME),
        "reconstruction_checkpoint": scene_relative_path(paths, ply),
        "metric_scale": metric_scale,
        "camera_scale": metric_scale * DEFAULT_CAMERA_CONDITIONING_SCALE,
    }
    if target_indices_path is not None:
        entry["target_indices_path"] = scene_relative_path(paths, target_indices_path)
    if not has_gt:
        entry["has_gt"] = False
    return entry


def write_eval_split(paths: PreparedPaths, metric_scale: float, trajectory: PreparedTrajectory | None, ply: Path) -> None:
    if trajectory is None:
        entry = eval_split_entry(
            paths,
            transforms_path=paths.input_dir / "nerfstudio/transforms.json",
            render_dir=paths.render_dir,
            metric_scale=metric_scale,
            ply=ply,
        )
    else:
        entry = eval_split_entry(
            paths,
            transforms_path=trajectory.transforms_path,
            render_dir=trajectory.render_dir,
            metric_scale=metric_scale,
            ply=ply,
            target_indices_path=trajectory.target_indices_path,
            has_gt=False,
        )
    write_json(paths.split_path, {"test": {paths.scene_id: entry}})


def prepare_files(
    args: argparse.Namespace,
    paths: PreparedPaths,
    image_dir: Path,
    sparse_dir: Path,
    scene: ColmapScene,
    selected_indices: list[int],
) -> None:
    if paths.input_dir.exists() and not args.replace:
        print(f"Skipping prepare; found {paths.input_dir}", flush=True)
        return

    reset_prepared_scene(paths.input_dir)
    prepare_images(image_dir, paths.input_dir / "images", scene.images, scene.cameras, args.image_downscale_factor)
    write_sparse_model(sparse_dir, paths.input_dir / "sparse/0", scene)
    write_transforms_json(paths.input_dir / "nerfstudio/transforms.json", scene)
    write_json(paths.selected_indices_path, selected_indices)
    (paths.scene_root / "selected_images.txt").write_text(
        "\n".join(image_basename(scene.images[index]) for index in selected_indices) + "\n"
    )


def render_photos(args: argparse.Namespace, paths: PreparedPaths, selected_indices: list[int]) -> None:
    """Render every photo frame; the selected photos stay anchors, the rest become targets."""
    transforms_path = paths.input_dir / "nerfstudio/transforms.json"
    if not args.replace and render_outputs_complete(
        paths.render_dir, trajectory_frame_count(transforms_path), expected_selected_indices=selected_indices
    ):
        print(f"Skipping render; found complete outputs in {paths.render_dir}", flush=True)
        return
    render_ply(ply=args.ply, trajectory_path=transforms_path, output_dir=paths.render_dir, selected_indices=selected_indices)


def trajectory_paths(paths: PreparedPaths) -> PreparedTrajectory:
    root = paths.scene_root / "trajectory"
    return PreparedTrajectory(
        root=root,
        transforms_path=root / "transforms.json",
        render_trajectory_path=root / "trajectory.json",
        selected_indices_path=root / "selected_indices.json",
        target_indices_path=root / "target_indices.json",
        render_dir=paths.scene_root / "renders" / "trajectory",
    )


def write_trajectory_inputs(
    args: argparse.Namespace,
    paths: PreparedPaths,
    scene: ColmapScene,
    selected_indices: list[int],
) -> PreparedTrajectory:
    """transforms.json = path targets first, then the anchor photos; trajectory.json = targets only."""
    trajectory = trajectory_paths(paths)
    trajectory_data = read_camera_trajectory(args.trajectory_path)
    assert_target_only_trajectory(trajectory_data, str(args.trajectory_path))
    target_frames = [
        frame_with_camera_intrinsics(frame, camera_intrinsics_for_frame(trajectory_data, frame))
        for frame in trajectory_data["frames"]
    ]
    selected_start = len(target_frames)
    source_intrinsics = shared_camera_intrinsics(scene)
    selected_frames = [
        frame_with_camera_intrinsics(source_frame(scene.images[index]), source_intrinsics) for index in selected_indices
    ]

    if trajectory.root.exists():
        assert trajectory.root.is_dir() and not trajectory.root.is_symlink(), f"Expected trajectory directory, got {trajectory.root}"
        shutil.rmtree(trajectory.root)
    trajectory.root.mkdir(parents=True)
    write_json(trajectory.transforms_path, transforms_json(trajectory_data, target_frames + selected_frames))
    write_json(trajectory.render_trajectory_path, transforms_json(trajectory_data, target_frames))
    write_json(trajectory.selected_indices_path, list(range(selected_start, selected_start + len(selected_indices))))
    write_json(trajectory.target_indices_path, list(range(len(target_frames))))
    return trajectory


def render_trajectory(
    args: argparse.Namespace, paths: PreparedPaths, scene: ColmapScene, selected_indices: list[int]
) -> PreparedTrajectory:
    trajectory = write_trajectory_inputs(args, paths, scene, selected_indices)
    # Only the targets are rendered; selected_indices names the anchor photos' slots in transforms.json.
    selected_start = len(read_camera_trajectory(trajectory.render_trajectory_path)["frames"])
    render_ply(
        ply=args.ply,
        trajectory_path=trajectory.render_trajectory_path,
        output_dir=trajectory.render_dir,
        selected_indices=list(range(selected_start, selected_start + len(selected_indices))),
    )
    return trajectory


def read_metric_scale(path: Path) -> float:
    for line in path.read_text().splitlines():
        if line.startswith("Scale factor:"):
            return float(line.split(":", 1)[1].split()[0])
    assert False, f"Could not find scale factor in {path}"


def run_metric_alignment(args: argparse.Namespace, paths: PreparedPaths) -> None:
    scale_info = paths.scale_dir / "scale_info.txt"
    if args.metric_scale is not None:
        if scale_info.is_file() and not args.replace:
            existing_scale = read_metric_scale(scale_info)
            assert np.isclose(existing_scale, args.metric_scale), (
                f"Existing metric scale at {scale_info} is {existing_scale}, "
                f"but --metric_scale={args.metric_scale}. Re-run with --replace."
            )
            print(f"Skipping metric alignment; found {scale_info}", flush=True)
            return
        paths.scale_dir.mkdir(parents=True, exist_ok=True)
        scale_info.write_text(f"Scale factor: {args.metric_scale}\n")
        return

    if scale_info.is_file() and not args.replace:
        print(f"Skipping metric alignment; found {scale_info}", flush=True)
        return

    from data_processing.sparse_recon.metric_alignment import align_colmap_to_metric_scale

    paths.scale_dir.mkdir(parents=True, exist_ok=True)
    scale, _, _ = align_colmap_to_metric_scale(
        colmap_dir=args.colmap_dir / "sparse/0",
        image_dir=paths.input_dir / "images",
        output_dir=paths.scale_dir,
        debug=False,
        downsample_factor=1,
    )
    scale_info.write_text(f"Scale factor: {scale}\n")


def write_fixed_caption(caption: str, output_path: Path, text_encoder_model_id: str, max_sequence_length: int = 512) -> None:
    """caption.h5 from a given caption string, in generate_caption_hdf5's on-disk format."""
    import h5py
    import torch

    from data_processing.captioning.generate_captions import generate_text_embedding, get_text_encoder_and_tokenizer

    text_encoder, tokenizer = get_text_encoder_and_tokenizer(text_encoder_model_id)
    with torch.inference_mode():
        embedding = generate_text_embedding(caption, text_encoder, tokenizer, max_sequence_length)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as hf:
        dataset = hf.create_dataset("frame_00000", data=embedding)
        dataset.attrs["caption"] = caption
        dataset.attrs["image_indices"] = np.array([0])
    print(f"Wrote {output_path} embedding={embedding.shape}", flush=True)


def generate_caption(args: argparse.Namespace, paths: PreparedPaths) -> None:
    caption_path = paths.prompt_dir / paths.scene_id / DEFAULT_CAPTION_FILENAME
    if caption_path.is_file() and not args.replace:
        print(f"Skipping captioning; found {caption_path}", flush=True)
        return
    if args.caption is not None:
        write_fixed_caption(args.caption, caption_path, args.text_encoder_model_id)
        return

    from data_processing.captioning.generate_captions import generate_caption_hdf5

    generate_caption_hdf5(
        input_path=paths.input_dir,
        output_path=caption_path,
        dataset_downsample_factor=1,
        captioning_model_id=DEFAULT_CAPTIONING_MODEL_ID,
        text_encoder_model_id=args.text_encoder_model_id,
    )


def metric_scale(args: argparse.Namespace, paths: PreparedPaths) -> float | None:
    if args.metric_scale is not None:
        return args.metric_scale
    scale_info = paths.scale_dir / "scale_info.txt"
    return read_metric_scale(scale_info) if scale_info.is_file() else None


def parse_phases(value: str) -> set[str]:
    phases = {phase.strip() for phase in value.split(",") if phase.strip()}
    valid = {"prepare", "render", "scale", "caption"}
    unknown = phases - valid
    assert not unknown, f"Unknown phases: {sorted(unknown)}"
    assert phases, f"--phases must include at least one of {sorted(valid)}"
    return phases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", type=Path, required=True, help="Trained 3DGS PLY (standard INRIA attributes).")
    parser.add_argument(
        "--colmap_dir",
        type=Path,
        required=True,
        help="COLMAP scene the PLY was trained on: images/ and sparse/0/{cameras,images,points3D}.bin.",
    )
    parser.add_argument(
        "--output_root", type=Path, required=True, help="Scene output directory. Its name is the scene id."
    )
    parser.add_argument(
        "--selected_image_names_file",
        type=Path,
        default=None,
        help="Newline-delimited photo names kept as real anchors. Defaults to every photo.",
    )
    parser.add_argument(
        "--trajectory_path",
        type=Path,
        default=None,
        help="transforms-style JSON of novel target cameras (OpenGL c2w + intrinsics). Without it, the "
        "photo frames that are not anchors are the targets.",
    )
    parser.add_argument("--phases", default="prepare,render,scale,caption", help="Comma-separated phases.")
    parser.add_argument("--replace", action="store_true", help="Redo phases whose outputs already exist.")
    parser.add_argument("--caption", default=None, help="Scene caption text; skips the Qwen3-VL captioner.")
    parser.add_argument("--text_encoder_model_id", default=DEFAULT_TEXT_ENCODER_MODEL_ID, help="Wan text encoder.")
    parser.add_argument(
        "--image_downscale_factor",
        type=int,
        default=1,
        help="Downscale the prepared photos (and their renders) by this factor.",
    )
    parser.add_argument(
        "--normalize_near_shared_cameras",
        action="store_true",
        help="Collapse many near-identical COLMAP cameras into one fixed calibration.",
    )
    parser.add_argument("--metric_scale", type=float, default=None, help="Precomputed metric scale; skips MoGe.")
    return parser


def prepare_scene(args: argparse.Namespace) -> None:
    phases = parse_phases(args.phases)
    assert args.image_downscale_factor >= 1, "--image_downscale_factor must be >= 1"

    args.ply = args.ply.expanduser().resolve()
    args.colmap_dir = args.colmap_dir.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.selected_image_names_file is not None:
        args.selected_image_names_file = args.selected_image_names_file.expanduser().resolve()
    if args.trajectory_path is not None:
        args.trajectory_path = args.trajectory_path.expanduser().resolve()
        assert "render" in phases, "--trajectory_path requires the render phase"
    image_dir, sparse_dir = resolve_colmap_paths(args.colmap_dir)
    scene = read_colmap_scene(sparse_dir)
    require_unique_basenames(scene.images)
    scene = scale_colmap_scene_to_images(image_dir, scene)
    if args.normalize_near_shared_cameras:
        scene = normalize_near_shared_camera_scene(scene)
    if args.image_downscale_factor != 1:
        factor = 1.0 / args.image_downscale_factor
        scene = ColmapScene(
            cameras=[
                camera._replace(
                    width=max(1, int(round(camera.width * factor))),
                    height=max(1, int(round(camera.height * factor))),
                    params=scaled_camera_params(camera, factor, factor),
                )
                for camera in scene.cameras
            ],
            images=[scale_image_observations(image, factor, factor) for image in scene.images],
        )

    selected_indices = resolve_selected_indices(args, scene.images)
    paths = prepared_paths(args.output_root, args.output_root.name)
    paths.scene_root.mkdir(parents=True, exist_ok=True)

    if "prepare" in phases:
        prepare_files(args, paths, image_dir, sparse_dir, scene, selected_indices)
    assert paths.input_dir.is_dir(), f"Missing prepared scene {paths.input_dir}: run the prepare phase"

    trajectory = trajectory_paths(paths) if args.trajectory_path is not None else None
    if "render" in phases:
        if trajectory is None:
            render_photos(args, paths, selected_indices)
        else:
            trajectory = render_trajectory(args, paths, scene, selected_indices)
    if "scale" in phases:
        run_metric_alignment(args, paths)
    if "caption" in phases:
        generate_caption(args, paths)

    scale = metric_scale(args, paths)
    render_dir = paths.render_dir if trajectory is None else trajectory.render_dir
    frame_count = trajectory_frame_count(
        paths.input_dir / "nerfstudio/transforms.json" if trajectory is None else trajectory.render_trajectory_path
    )
    caption_ready = (paths.prompt_dir / paths.scene_id / DEFAULT_CAPTION_FILENAME).is_file()
    if scale is not None and caption_ready and render_outputs_complete(render_dir, frame_count):
        write_eval_split(paths, scale, trajectory, args.ply)
        print(f"split_path={paths.split_path}", flush=True)
    else:
        print(f"split.json not written yet: scale={scale} caption={caption_ready} renders={render_dir}", flush=True)
    print(f"scene_root={paths.scene_root} anchors={len(selected_indices)}", flush=True)


def main() -> None:
    prepare_scene(build_parser().parse_args())


if __name__ == "__main__":
    main()
