# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""COLMAP binary/text readers, copied from 3DGRUT (threedgrut/datasets/utils.py) so this repo needs no 3DGRUT."""

from __future__ import annotations

import collections
import struct

import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_colmap_points3D_text(path):
    """
    Read points3D.txt file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []

    # Single file read
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                # Convert directly to numpy arrays while appending
                xyzs.append([float(x) for x in elems[1:4]])
                rgbs.append([int(x) for x in elems[4:7]])
                errors.append(float(elems[7]))

    # Convert lists to numpy arrays all at once
    return (
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def read_colmap_points3D_binary(path_to_model_file):
    """
    Read points3D.bin file from COLMAP output.
    Returns numpy arrays of xyz coordinates, RGB values, and reprojection errors.
    """
    # Pre-allocate lists for data
    xyzs = []
    rgbs = []
    errors = []

    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        for _ in range(num_points):
            # Read the point data
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            # Append coordinates, colors, and error
            xyzs.append(binary_point_line_properties[1:4])
            rgbs.append(binary_point_line_properties[4:7])
            errors.append(binary_point_line_properties[7])

            # Skip track length and elements as they're not used
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            fid.seek(8 * track_length, 1)

    # Convert lists to numpy arrays all at once
    return (
        np.array(xyzs, dtype=np.float64),
        np.array(rgbs, dtype=np.int32),
        np.array(errors, dtype=np.float64).reshape(-1, 1),
    )


def read_colmap_intrinsics_text(path):
    """
    Read camera intrinsics from a COLMAP text file.
    Args:
        path: Path to the cameras.txt file
    Returns:
        Dict of Camera objects indexed by camera ID
    """
    cameras = {}
    with open(path, "r") as fid:
        # Skip comment lines at the start
        lines = (line.strip() for line in fid)
        lines = (line for line in lines if line and not line.startswith("#"))
        for line in lines:
            # Unpack elements directly using split with maxsplit
            camera_id, model, width, height, *params = line.split()
            camera_id = int(camera_id)
            width, height = int(width), int(height)
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=np.array([float(p) for p in params]),
            )
    return cameras


def read_colmap_intrinsics_binary(path_to_model_file):
    """
    Read camera intrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the cameras.bin file
    Returns:
        Dict of Camera objects indexed by camera ID
    Raises:
        ValueError: If the number of cameras read doesn't match the expected count
        KeyError: If an invalid camera model ID is encountered
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        # Read number of cameras
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            # Read fixed-size camera properties
            camera_id, model_id, width, height = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            # Get camera model information
            try:
                camera_model = CAMERA_MODEL_IDS[model_id]
            except KeyError:
                raise KeyError(f"Invalid camera model ID: {model_id}")
            # Read camera parameters
            params = read_next_bytes(
                fid,
                num_bytes=8 * camera_model.num_params,
                format_char_sequence="d" * camera_model.num_params,
            )
            # Create camera object
            assert camera_id not in cameras, f"Camera ID {camera_id} already exists"
            cameras[camera_id] = Camera(
                id=camera_id,
                model=camera_model.model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    # Verify camera count
    assert len(cameras) == num_cameras, f"Expected {num_cameras} cameras, but read {len(cameras)}"
    return cameras


def qvec_to_so3(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


class Image(BaseImage):
    def qvec_to_so3(self):
        return qvec_to_so3(self.qvec)


def read_colmap_extrinsics_binary(path_to_model_file):
    """
    Read camera extrinsics from a COLMAP binary file.
    Args:
        path_to_model_file: Path to the images.bin file
    Returns:
        List of Image objects sorted by image name
    Raises:
        ValueError: If string parsing or data reading fails
    """
    images = []
    with open(path_to_model_file, "rb") as fid:
        # Read number of registered images
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            # Read image properties (id, rotation, translation, camera_id)
            props = read_next_bytes(fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id, *qvec_tvec, camera_id = props
            qvec = np.array(qvec_tvec[:4])
            tvec = np.array(qvec_tvec[4:7])
            # Read image name (null-terminated string)
            image_name = ""
            while True:
                current_char = read_next_bytes(fid, 1, "c")[0]
                if current_char == b"\x00":
                    break
                try:
                    image_name += current_char.decode("utf-8")
                except UnicodeDecodeError:
                    raise ValueError(f"Invalid character in image name at position {len(image_name)}")
            # Read 2D points
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            point_data = read_next_bytes(
                fid,
                num_bytes=24 * num_points2D,
                format_char_sequence="ddq" * num_points2D,
            )
            # Parse point data into coordinates and IDs
            xys = np.array([(point_data[i], point_data[i + 1]) for i in range(0, len(point_data), 3)])
            point3D_ids = np.array([int(point_data[i + 2]) for i in range(0, len(point_data), 3)])
            # Create image object
            images.append(
                Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
            )
    return sorted(images, key=lambda x: x.name)

