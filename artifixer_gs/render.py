# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a standard 3DGS PLY on a transforms.json camera path into ArtiFixer's input layout.

<out_dir>/renders/{i:05d}.png   RGB composited on black (as the ArtiFixer training renders)
<out_dir>/opacity/{i:05d}.png   accumulated alpha * 255, the latent mixing weight ArtiFixer uses
<out_dir>/selected_indices.json the frame indices that keep their real photo

Any method that writes the INRIA 3DGS PLY (3DGS, FastGS, Taming-3DGS, ...) works. gsplat does the
rasterization. If gsplat is not importable here, the render runs in the interpreter named by
GS_RENDER_PY (a Python with torch + gsplat) with this repo on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from data_processing.camera_trajectories import (
    applied_transform_matrix,
    camera_intrinsics_for_frame,
    opengl_c2w_to_opencv_w2c,
    read_camera_trajectory,
    write_json,
)

RENDER_FRAME_DIRS = ("renders", "opacity")
DISTORTION_KEYS = ("k1", "k2", "p1", "p2")
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_ply(path: Path) -> dict[str, np.ndarray]:
    """Read a binary little-endian INRIA 3DGS PLY into activated arrays.

    Returns means [N,3], quats [N,4] (wxyz, normalized), scales [N,3] (exp), opacities [N] (sigmoid)
    and sh [N,K,3] with K = (degree+1)^2, DC first."""
    with path.open("rb") as f:
        header = []
        while True:
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        assert header[0] == "ply", f"{path} is not a PLY file"
        assert "format binary_little_endian 1.0" in header, f"{path} must be binary_little_endian"
        types = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2"}
        count, fields, in_vertex = 0, [], False
        for line in header:
            parts = line.split()
            if parts[:2] == ["element", "vertex"]:
                count, in_vertex = int(parts[2]), True
            elif parts and parts[0] == "element":
                in_vertex = False
            elif in_vertex and parts[:1] == ["property"]:
                fields.append((parts[2], types[parts[1]]))
        data = np.fromfile(f, dtype=np.dtype(fields), count=count)

    names = data.dtype.names
    def stack(prefix: str) -> np.ndarray:
        keys = sorted((name for name in names if name.startswith(prefix)), key=lambda name: int(name.split("_")[-1]))
        return np.stack([data[key] for key in keys], axis=1).astype(np.float32)

    f_dc = stack("f_dc_")
    f_rest = stack("f_rest_") if any(name.startswith("f_rest_") for name in names) else np.zeros((count, 0), np.float32)
    n_coeffs = 1 + f_rest.shape[1] // 3
    degree = int(round(np.sqrt(n_coeffs))) - 1
    assert (degree + 1) ** 2 == n_coeffs, f"{path}: {f_rest.shape[1]} f_rest values is not a full SH band"
    # INRIA stores f_rest channel-major: (N, 3, K-1) flattened.
    sh = np.concatenate([f_dc[:, None, :], f_rest.reshape(count, 3, n_coeffs - 1).transpose(0, 2, 1)], axis=1)
    quats = stack("rot_")
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    return {
        "means": np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32),
        "quats": quats,
        "scales": np.exp(stack("scale_")),
        "opacities": 1.0 / (1.0 + np.exp(-data["opacity"].astype(np.float32))),
        "sh": sh,
        "sh_degree": np.array(degree),
    }


def trajectory_cameras(trajectory_path: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """OpenCV world-to-camera matrices [M,4,4] and intrinsics [M,3,3] of a transforms.json path."""
    transforms = json.loads(trajectory_path.read_text())
    applied_transform = applied_transform_matrix(transforms)
    viewmats, Ks, sizes = [], [], set()
    for index, frame in enumerate(transforms["frames"]):
        intrinsics = camera_intrinsics_for_frame(transforms, frame)
        distortion = [intrinsics[key] for key in DISTORTION_KEYS]
        assert not any(distortion), f"{trajectory_path} frame {index} has lens distortion {distortion}; 3DGS renders pinhole only"
        sizes.add((intrinsics["w"], intrinsics["h"]))
        viewmats.append(opengl_c2w_to_opencv_w2c(frame["transform_matrix"], applied_transform))
        Ks.append([[intrinsics["fl_x"], 0, intrinsics["cx"]], [0, intrinsics["fl_y"], intrinsics["cy"]], [0, 0, 1]])
    assert len(sizes) == 1, f"{trajectory_path} must use one resolution, got {sorted(sizes)}"
    width, height = sizes.pop()
    return np.asarray(viewmats, np.float32), np.asarray(Ks, np.float32), width, height


def render_frames(ply_path: Path, trajectory_path: Path, out_dir: Path, depth: bool = False) -> None:
    """Rasterize every path frame with gsplat and write the PNGs (needs torch + gsplat)."""
    import torch
    from gsplat import rasterization
    from PIL import Image

    device = torch.device("cuda")
    gaussians = read_ply(ply_path)
    sh_degree = int(gaussians.pop("sh_degree"))
    g = {key: torch.from_numpy(value).to(device) for key, value in gaussians.items()}
    viewmats, Ks, width, height = trajectory_cameras(trajectory_path)
    for name in (*RENDER_FRAME_DIRS, *(("depth",) if depth else ())):
        (out_dir / name).mkdir(parents=True, exist_ok=True)

    print(f"Rendering {len(viewmats)} frames of {trajectory_path} from {ply_path} ({len(g['means'])} Gaussians, SH {sh_degree})", flush=True)
    with torch.no_grad():
        for index in range(len(viewmats)):
            colors, alphas, _ = rasterization(
                means=g["means"],
                quats=g["quats"],
                scales=g["scales"],
                opacities=g["opacities"],
                colors=g["sh"],
                viewmats=torch.from_numpy(viewmats[index : index + 1]).to(device),
                Ks=torch.from_numpy(Ks[index : index + 1]).to(device),
                width=width,
                height=height,
                sh_degree=sh_degree,
                render_mode="RGB+ED" if depth else "RGB",
                rasterize_mode="classic",
                backgrounds=torch.zeros(1, 3, device=device),
            )
            rgb = colors[0, ..., :3]
            # Same quantization as torchvision.save_image / the 3DGUT renderer.
            rgb = (rgb * 255 + 0.5).clamp(0, 255).byte().cpu().numpy()
            alpha = (alphas[0, ..., 0] * 255).round().clamp(0, 255).byte().cpu().numpy()
            Image.fromarray(rgb).save(out_dir / "renders" / f"{index:05d}.png")
            Image.fromarray(alpha).save(out_dir / "opacity" / f"{index:05d}.png")
            if depth:
                np.save(out_dir / "depth" / f"{index:05d}.npy", colors[0, ..., 3].cpu().numpy())


def gsplat_available() -> bool:
    try:
        import gsplat  # noqa: F401
    except ImportError:
        return False
    return True


def render_outputs_complete(output_dir: Path, frame_count: int, *, expected_selected_indices: list[int] | None = None) -> bool:
    selected_indices_path = output_dir / "selected_indices.json"
    if not selected_indices_path.is_file():
        return False
    if expected_selected_indices is not None and json.loads(selected_indices_path.read_text()) != expected_selected_indices:
        return False
    return all(
        (output_dir / dirname / f"{index:05d}.png").is_file() for dirname in RENDER_FRAME_DIRS for index in range(frame_count)
    )


def render_ply(
    *,
    ply: Path,
    trajectory_path: Path,
    output_dir: Path,
    selected_indices: list[int],
    depth: bool = False,
) -> Path:
    """Render ply on trajectory_path into output_dir (cleared first) and record selected_indices."""
    ply, trajectory_path, output_dir = ply.resolve(), trajectory_path.resolve(), output_dir.resolve()
    if not ply.is_file():
        raise FileNotFoundError(f"Missing Gaussian PLY: {ply}")
    for name in (*RENDER_FRAME_DIRS, "depth"):
        if (output_dir / name).exists():
            shutil.rmtree(output_dir / name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if gsplat_available():
        render_frames(ply, trajectory_path, output_dir, depth)
    else:
        python = os.environ.get("GS_RENDER_PY")
        assert python, "gsplat is not importable here: set GS_RENDER_PY to a Python that has torch + gsplat"
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), os.environ.get("PYTHONPATH", "")])}
        command = [python, "-m", "artifixer_gs.render", "--ply", str(ply), "--trajectory", str(trajectory_path), "--out_dir", str(output_dir)]
        subprocess.run(command + (["--depth"] if depth else []), check=True, env=env, cwd=REPO_ROOT)

    frame_count = len(read_camera_trajectory(trajectory_path)["frames"])
    write_json(output_dir / "selected_indices.json", selected_indices)
    assert render_outputs_complete(output_dir, frame_count), f"Incomplete renders in {output_dir}"
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True, help="transforms.json-style camera path (OpenGL c2w).")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--depth", action="store_true", help="Also write depth/{i:05d}.npy (expected z-depth).")
    args = parser.parse_args()
    render_frames(args.ply.resolve(), args.trajectory.resolve(), args.out_dir.resolve(), args.depth)


if __name__ == "__main__":
    sys.exit(main())
