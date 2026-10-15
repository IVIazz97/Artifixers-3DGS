#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write a COLMAP scene restricted to some photos, to train a 3DGS on the anchors only.

images/ are symlinks, cameras.bin and points3D.bin are shared, images.bin keeps only the named
images (with their original ids and poses), so the result stays in the source world frame.
"""

from __future__ import annotations

import argparse
import shutil
import struct
from pathlib import Path

from artifixer_gs.colmap import read_colmap_extrinsics_binary


def write_subset(colmap_dir: Path, names: list[str], output_dir: Path) -> None:
    sparse = colmap_dir / "sparse" / "0"
    images = {Path(image.name).name: image for image in read_colmap_extrinsics_binary(sparse / "images.bin")}
    missing = [name for name in names if Path(name).name not in images]
    assert not missing, f"Not in {sparse / 'images.bin'}: {missing[:10]}"
    if output_dir.exists():
        assert output_dir.is_dir() and not output_dir.is_symlink(), f"Expected a directory: {output_dir}"
        shutil.rmtree(output_dir)
    (output_dir / "images").mkdir(parents=True)
    (output_dir / "sparse" / "0").mkdir(parents=True)
    kept = [images[Path(name).name] for name in names]
    with (output_dir / "sparse" / "0" / "images.bin").open("wb") as fid:
        fid.write(struct.pack("<Q", len(kept)))
        for image in kept:
            fid.write(struct.pack("<idddddddi", image.id, *image.qvec, *image.tvec, image.camera_id))
            fid.write(Path(image.name).name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(image.xys)))
            for xy, point3d_id in zip(image.xys, image.point3D_ids):
                fid.write(struct.pack("<ddq", float(xy[0]), float(xy[1]), int(point3d_id)))
            source = colmap_dir / "images" / Path(image.name).name
            (output_dir / "images" / Path(image.name).name).symlink_to(source.resolve())
    for name in ("cameras.bin", "points3D.bin"):
        shutil.copyfile(sparse / name, output_dir / "sparse" / "0" / name)
    print(f"Wrote {output_dir} with {len(kept)} of {len(images)} images", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colmap_dir", type=Path, required=True)
    parser.add_argument("--image_names_file", type=Path, required=True, help="Newline-delimited photo names to keep.")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()
    names = [line.strip() for line in args.image_names_file.read_text().splitlines() if line.strip()]
    write_subset(args.colmap_dir.resolve(), names, args.output_dir.resolve())


if __name__ == "__main__":
    main()
