# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PSNR on the target (non-anchor) frames of a scripts/run.sh output, per stage, plus Gaussian stats.

Stages: base PLY render, ArtiFixer, ArtiFixer3D render, ArtiFixer3D+, and optionally the renders of a
3DGUT reconstruction of the same frames. Predictions are resized to the photo size before scoring.
Only for photo-target runs (split has ground truth).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from artifixer_gs.render import read_ply


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) / 255 - b.astype(np.float64) / 255) ** 2)
    return float(10 * np.log10(1 / max(mse, 1e-12)))


def load(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return np.asarray(image if image.size == size else image.resize(size, Image.Resampling.LANCZOS))


def pred_dir(root: Path) -> Path | None:
    return next(iter(sorted(root.glob("**/frames/batch_0000/pred"))), None) if root.is_dir() else None


def opacity_stats(opacity: np.ndarray) -> str:
    return (
        f"N={len(opacity):,}  opacity<0.05: {np.mean(opacity < 0.05):.1%}  <0.1: {np.mean(opacity < 0.1):.1%}  "
        f"median {np.median(opacity):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="OUT of scripts/run.sh")
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--threedgut_renders", type=Path, default=None, help="3DGUT renders/ of the same frames")
    parser.add_argument("--threedgut_ckpt", type=Path, default=None, help="3DGUT ckpt_N.pt for opacity stats")
    args = parser.parse_args()

    split = json.loads((args.out / "split.json").read_text())["test"]
    scene_id, meta = next(iter(split.items()))
    transforms = json.loads((args.out / meta["transforms_path"]).read_text())
    selected = set(json.loads((args.out / meta["selected_indices_path"]).read_text()))
    targets = [index for index in range(len(transforms["frames"])) if index not in selected]
    size = (int(transforms["w"]), int(transforms["h"]))
    image_root = args.out / meta["image_root"]

    stages = {
        "3DGUT recon": args.threedgut_renders,
        "FastGS recon": args.out / "renders" / "photos" / "renders",
        "ArtiFixer": pred_dir(args.out / "artifixer"),
        "ArtiFixer3D (FastGS)": args.out / "af3d" / "renders" / "renders",
        "ArtiFixer3D+": pred_dir(args.out / "artifixer3d_plus"),
    }
    print(f"{scene_id}: {len(targets)} target frames, {len(selected)} anchors, {size[0]}x{size[1]}")
    for name, directory in stages.items():
        if directory is None or not directory.is_dir():
            print(f"  {name:22s} missing")
            continue
        scores = [
            psnr(load(directory / f"{index:05d}.png", size), load(image_root / transforms["frames"][index]["file_path"], size))
            for index in targets
        ]
        print(f"  {name:22s} PSNR {np.mean(scores):6.2f} dB  ({directory})")

    print("Gaussians:")
    if args.threedgut_ckpt is not None and args.threedgut_ckpt.is_file():
        import torch

        density = torch.load(args.threedgut_ckpt, map_location="cpu", weights_only=False)["density"]
        print(f"  3DGUT MCMC       {opacity_stats(torch.sigmoid(density.detach()).squeeze(-1).numpy())}")
    for name, ply in (
        ("FastGS recon", Path(meta["reconstruction_checkpoint"])),
        ("FastGS AF3D", args.out / "af3d" / "fastgs" / "point_cloud" / "iteration_30000" / "point_cloud.ply"),
    ):
        ply = ply if ply.is_absolute() else args.out / ply
        if ply.is_file():
            print(f"  {name:16s} {opacity_stats(read_ply(ply)['opacities'])}")


if __name__ == "__main__":
    main()
