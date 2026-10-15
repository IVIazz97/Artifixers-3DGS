<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ArtiFixer-GS

[ArtiFixer](https://research.nvidia.com/labs/sil/projects/artifixer/) (de Lutio et al., SIGGRAPH 2026) for
any 3D Gaussian Splatting method that writes the standard 3DGS PLY: 3DGS, FastGS, Taming-3DGS, and others.

The original repository trains and renders with 3DGUT. The ArtiFixer diffusion model itself only reads
images: RGB renders, opacity maps and the photos along a camera path. This repository keeps the
diffusion model and its data preparation unchanged, renders the PLY with [gsplat](https://github.com/nerfstudio-project/gsplat),
and leaves 3DGS training to the method of your choice. The ArtiFixer3D distillation runs in FastGS
(branch `artifixer`, flag `--lpips_images`). There is no 3DGUT anywhere.

```
your 3DGS PLY ──prepare──► renders + opacity ──ArtiFixer──► fixed frames (video)
                                                              │
               ArtiFixer3D+ video ◄──ArtiFixer── renders ◄── FastGS (photos + fixed frames) ◄──handoff
```

## Pipeline

| stage | command | env | output |
|---|---|---|---|
| prepare | `python -m artifixer_gs.prepare` | ArtiFixer | `input/<id>/` (photos, COLMAP, `transforms.json`), `renders/{photos,trajectory}/{renders,opacity}/`, metric scale, `captions/<id>/caption.h5`, `split.json` |
| infer | `python -m model_eval.run_inference --evalset reconstructed_colmap` | ArtiFixer | `artifixer/.../frames/batch_0000/pred/*.png` + mp4 |
| handoff | `python -m artifixer_gs.distill handoff` | ArtiFixer | `af3d/input/`: COLMAP scene with anchor photos + ArtiFixer frames (PINHOLE), `generated_images.txt` |
| train | FastGS `train.py --lpips_images af3d/input/generated_images.txt` | FastGS | `af3d/fastgs/point_cloud/iteration_N/point_cloud.ply`, `gs_stats.json` |
| plus | `python -m artifixer_gs.distill plus` | ArtiFixer | `af3d/renders/`, `split_artifixer3d_plus.json` |
| infer_plus | `run_inference` on the plus split | ArtiFixer | `artifixer3d_plus/.../pred/*.png` + mp4 |

`scripts/run.sh` runs them all, with one log per stage in `$OUT/logs/`.

```bash
PLY=.../point_cloud.ply COLMAP=.../scene OUT=output/scene ANCHORS=.../train_images.txt \
CAPTION="A kitchen table ..." MODEL_ID=.../Wan2.1-T2V-1.3B-Diffusers CHECKPOINT_PT=.../artifixer-1.3b.pt \
FASTGS_REPO=.../FastGS FASTGS_PY=.../fastgs/bin/python bash scripts/run.sh
```

There are two kinds of path:

- **Sparse anchors** (`ANCHORS`): the listed photos stay real. The other photo frames are targets that
  ArtiFixer regenerates. The PLY should be trained on the anchors only. `python -m artifixer_gs.subset`
  writes that anchor-only COLMAP scene.
- **Novel path** (`TRAJ`): a `transforms.json` of target cameras, given as OpenGL camera-to-world
  matrices plus intrinsics. Every photo is an anchor.

`STAGES=handoff,train,plus,infer_plus` runs only part of the pipeline. The ArtiFixer stage alone
(PLY in, video out) is `STAGES=prepare,infer`.

### What ArtiFixer needs from the 3D model

- `renders/{i:05d}.png`: 8-bit RGB composited on **black**, the same as the training renders.
- `opacity/{i:05d}.png`: accumulated alpha × 255. ArtiFixer uses it as the per-pixel mix between the
  render latent and noise (`model_training/pipeline/pipeline_base.py`).
- `transforms.json` with the poses and intrinsics, and `split.json` pointing at all of the above.

`artifixer_gs.render` writes exactly this from a PLY. It reads the SH degree from the `f_rest` count,
applies sigmoid to opacity, exp to scale and normalises the wxyz quaternion, and uses gsplat's
`classic` mode with eps2d 0.3, which is the 3DGS rasterizer. If gsplat is not installed in the
ArtiFixer Python, the render runs as a subprocess in `GS_RENDER_PY`.

## FastGS: ArtiFixer3D training

Branch `artifixer` of FastGS adds `--lpips_images FILE` and `--lambda_lpips 0.1` to `train.py`.
Images listed in the file (the ArtiFixer frames) get `0.1 · LPIPS(vgg)` only. Every other image (the
real photos) keeps FastGS's `0.8 L1 + 0.2 D-SSIM`. This is the 3DGUT ArtiFixer3D recipe
(`colmap_3dgut_sparse_mcmc_lpips`: `lambda_reconlosses_override 0`): frame-to-frame inconsistencies in
the generated frames do not blur the model.

FastGS's multi-view densification and pruning are unchanged. That includes the final pruning of
Gaussians below opacity 0.1, which the MCMC strategy of the 3DGUT version does not have.
`gs_stats.json` reports the Gaussian count and the share of low-opacity Gaussians.

## Deviations from the 3DGUT version

- **Training from scratch only.** There is no `--base_checkpoint` continuation with rescaled schedules.
  ArtiFixer3D trains FastGS from the COLMAP points, as the default 3DGUT ArtiFixer3D does.
- **Pinhole only.** Lens distortion raises an error, so undistort the scene first. 3DGS trainers that
  read PINHOLE COLMAP ignore the principal point and assume the image centre. The renders here use the
  exact `cx, cy` (sub-pixel on mip360).
- **Renderer.** Renders come from gsplat, not 3DGUT. On mip360 kitchen they match FastGS's own renders
  at 49 dB PSNR (`tests/check_render_vs_fastgs.py`).
- **Captions.** `--caption TEXT` writes `caption.h5` with the Wan text encoder and skips the Qwen3-VL
  captioner.

## Files

- `artifixer_gs/`: `prepare.py` (adapted from `data_processing/prepare_colmap_artifixer_inputs.py`),
  `render.py`, `distill.py` (adapted from `data_processing/artifixer3d.py`), `subset.py`, and
  `colmap.py` / `camera_models.py` (copied from 3DGRUT so no 3DGRUT install is needed).
- `model_eval/`, `model_training/`, `data_processing/`: the ArtiFixer inference code, copied unchanged
  except for those two imports.
- `scripts/run.sh`: the whole pipeline. `scripts/leonardo/`: Leonardo env and the kitchen test job.
- `tests/check_render_vs_fastgs.py`: renderer equivalence. `tests/compare_targets.py`: PSNR per stage
  on the target frames, plus Gaussian opacity stats.

Models are not included: Wan2.1-T2V-1.3B-Diffusers, `artifixer-1.3b.pt` (`nvidia/ArtiFixer`) and
MoGe-2 (`MOGE_MODEL_PATH`).

## License

Apache-2.0, as ArtiFixer. See [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).
Please cite ArtiFixer:

```bibtex
@inproceedings{delutio2026artifixer,
    title={ArtiFixer: Enhancing and Extending 3D Reconstruction with Auto-Regressive Diffusion Models},
    author={de Lutio, Riccardo and Fischer, Tobias and Chang, Yen-Yu and Zhang, Yuxuan and
            Wu, Jay Zhangjie and Ren, Xuanchi and Shen, Tianchang and Tothova, Katarina and
            Gojcic, Zan and Turki, Haithem},
    booktitle={SIGGRAPH},
    year={2026}
}
```
