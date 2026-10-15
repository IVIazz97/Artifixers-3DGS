# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that artifixer_gs.render (gsplat) reproduces FastGS's own renders of a FastGS PLY.

FastGS test views of a COLMAP scene trained with `--eval` (every 8th image by name) at FastGS's
resolution (width 1600 when wider, `-r -1`), principal point at the image centre as 3DGS assumes.
Prints PSNR(gsplat, FastGS render) and PSNR of each against the photo. Run with a gsplat Python:

  PYTHONPATH=. python tests/check_render_vs_fastgs.py --colmap_dir .../mip360/kitchen \
      --fastgs_dir .../FastGS/output/mip360/kitchen_big --out_dir /tmp/check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from artifixer_gs.colmap import read_colmap_extrinsics_binary, read_colmap_intrinsics_binary
from artifixer_gs.render import render_frames
from data_processing.camera_trajectories import opencv_w2c_to_opengl_c2w, transforms_json


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return float(10 * np.log10(1 / max(mse, 1e-12)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--fastgs_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=30000)
    parser.add_argument("--max_views", type=int, default=8)
    args = parser.parse_args()

    images = sorted(read_colmap_extrinsics_binary(args.colmap_dir / "sparse/0/images.bin"), key=lambda image: Path(image.name).stem)
    cameras = read_colmap_intrinsics_binary(args.colmap_dir / "sparse/0/cameras.bin")
    test = [image for index, image in enumerate(images) if index % 8 == 0][: args.max_views]
    camera = cameras[test[0].camera_id]
    assert camera.model == "PINHOLE", camera.model
    width, height = int(camera.width), int(camera.height)
    down = width / 1600 if width > 1600 else 1.0  # FastGS utils/camera_utils.py loadCam
    w, h = int(width / down), int(height / down)
    fx, fy = camera.params[0] * w / width, camera.params[1] * h / height

    frames = []
    for image in test:
        w2c = np.eye(4)
        w2c[:3, :3] = image.qvec_to_so3()
        w2c[:3, 3] = image.tvec
        frames.append({"transform_matrix": opencv_w2c_to_opengl_c2w(w2c).tolist()})
    intrinsics = {"camera_model": "OPENCV", "w": w, "h": h, "fl_x": fx, "fl_y": fy, "cx": w / 2, "cy": h / 2}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    trajectory = args.out_dir / "test_views.json"
    trajectory.write_text(json.dumps(transforms_json(intrinsics, frames)))

    ply = args.fastgs_dir / "point_cloud" / f"iteration_{args.iteration}" / "point_cloud.ply"
    render_frames(ply, trajectory, args.out_dir)
    fastgs_dir = args.fastgs_dir / "test" / f"ours_{args.iteration}"
    rows = []
    for index, image in enumerate(test):
        ours = np.asarray(Image.open(args.out_dir / "renders" / f"{index:05d}.png").convert("RGB"))
        theirs = np.asarray(Image.open(fastgs_dir / "renders" / f"{index:05d}.png").convert("RGB"))
        gt = np.asarray(Image.open(fastgs_dir / "gt" / f"{index:05d}.png").convert("RGB"))
        assert ours.shape == theirs.shape, f"{ours.shape} vs {theirs.shape}"
        alpha = np.asarray(Image.open(args.out_dir / "opacity" / f"{index:05d}.png"))
        rows.append((image.name, psnr(ours, theirs), psnr(ours, gt), psnr(theirs, gt), float(alpha.mean() / 255)))
        print(f"{image.name}: gsplat-vs-fastgs {rows[-1][1]:.2f} dB | gsplat-vs-gt {rows[-1][2]:.2f} | fastgs-vs-gt {rows[-1][3]:.2f} | mean alpha {rows[-1][4]:.3f}")
    means = np.mean([row[1:] for row in rows], axis=0)
    print(f"MEAN gsplat-vs-fastgs {means[0]:.2f} dB | gsplat-vs-gt {means[1]:.2f} | fastgs-vs-gt {means[2]:.2f} | alpha {means[3]:.3f}")


if __name__ == "__main__":
    main()
