# Leonardo setup for ArtiFixer-GS. Source from a job script: three existing envs, no new installs.
#   PY            artifixer env (torch 2.11, diffusers): prepare, inference, handoff
#   GS_RENDER_PY  gsplat env (gsplat 1.5.3): PLY rendering, called as a subprocess
#   FASTGS_PY     fastgs++ env (FastGS rasterizer): training, in FASTGS_REPO on branch artifixer
# /leonardo_scratch/fast/IscrC_EditGS is at its quota: temp files and writable caches go to work.
export TMPDIR=/leonardo_work/IscrC_EditGS/tmp_amazzucc/afgs_job_${SLURM_JOB_ID:-$$}
mkdir -p "$TMPDIR"
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/env.sh artifixer >/dev/null 2>&1
export TRITON_CACHE_DIR="$TMPDIR/triton"
export AFGS=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer-GS
export PYTHONPATH="$AFGS"
export PY=$ENVS/artifixer/bin/python
export GS_RENDER_PY=$ENVS/gsplat/bin/python
export FASTGS_PY=$ENVS/fastgs++/bin/python
export FASTGS_REPO=/leonardo_work/IscrC_EditGS/repos/FastGS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes have no internet
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor"
CKPTS=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/checkpoints   # shared with the ArtiFixer repo
export MODEL_ID=$CKPTS/Wan2.1-T2V-1.3B-Diffusers
export CHECKPOINT_PT=$CKPTS/artifixer/artifixer-1.3b.pt
export MOGE_MODEL_PATH=$CKPTS/moge-2-vitl-normal/model.pt
cd "$AFGS"
