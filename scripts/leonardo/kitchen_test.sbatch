#!/bin/bash
#SBATCH --account=IscrC_ESGS
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --job-name=afgs_kitchen
#SBATCH --output=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer-GS/output/logs/%x_%j.log
# End-to-end test on mip360 kitchen (ds4), same anchors as the 3DGUT ArtiFixer runs:
# FastGS on the 244 training photos -> ArtiFixer on the 35 held-out views -> ArtiFixer3D (FastGS)
# -> ArtiFixer3D+, then PSNR of every stage on the held-out views against the photos.
# Usage: sbatch scripts/leonardo/kitchen_test.sbatch   (STAGES=... reruns part of run.sh)
set -euo pipefail
source /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer-GS/scripts/leonardo/env.sh
SCENE=kitchen
DATA=/leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/datasets/mip360_ds4/$SCENE
export COLMAP=$DATA ANCHORS=$DATA/train_images.txt OUT=$AFGS/output/$SCENE
export CAPTION="An indoor dining room with a wooden table covered by a woven grey placemat and a pink table runner, wooden chairs, a leafy green houseplant and a bright window in the background."
export METRIC_SCALE=0.17030343437852719                 # the 3DGUT kitchen run's MoGe scale
FASTGS_KITCHEN="--densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002"
export FASTGS_ARGS=$FASTGS_KITCHEN                      # mip360 kitchen values from FastGS's benchmark
BASE=$OUT/base
export PLY=$BASE/fastgs/point_cloud/iteration_30000/point_cloud.ply
mkdir -p "$OUT/logs"
echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader)"

if [ ! -f "$PLY" ]; then
  $PY -m artifixer_gs.subset --colmap_dir "$COLMAP" --image_names_file "$ANCHORS" --output_dir "$BASE/input"
  (cd "$FASTGS_REPO" && PYTHONPATH= $FASTGS_PY train.py -s "$BASE/input" -m "$BASE/fastgs" -r 1 \
     --iterations 30000 --save_iterations 30000 --test_iterations 30000 $FASTGS_KITCHEN) 2>&1 | tee "$OUT/logs/base_train.log"
fi
bash scripts/run.sh
$PY tests/compare_targets.py --out "$OUT" --colmap_dir "$COLMAP" \
  --threedgut_ckpt /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/recon/kitchen/3dgrut_runs/kitchen/kitchen/ours_30000/ckpt_30000.pt \
  --threedgut_renders /leonardo_work/IscrC_EditGS/repos/CVPR2027/ArtiFixer/output/recon/kitchen/recon_results/kitchen/reconstruction/kitchen/ours_30000/renders \
  | tee "$OUT/logs/compare.log"
echo AFGS_KITCHEN_DONE
