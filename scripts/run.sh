#!/usr/bin/env bash
# PLY -> ArtiFixer video -> ArtiFixer3D (FastGS) -> ArtiFixer3D+ video, on one GPU.
#
# Required:
#   PLY      trained 3DGS PLY (standard INRIA attributes), trained on COLMAP
#   COLMAP   COLMAP scene: images/ + sparse/0/{cameras,images,points3D}.bin (undistorted)
#   OUT      output scene root; its basename is the scene id
# Path (one of):
#   ANCHORS  file of photo names kept as real anchors; the other photos are ArtiFixer targets
#   TRAJ     transforms.json of novel target cameras (OpenGL c2w + intrinsics); every photo is an anchor
# Optional:
#   CAPTION       scene caption (else the Qwen3-VL captioner runs)
#   METRIC_SCALE  metres per scene unit (else MoGe-2 estimates it)
#   STAGES        subset of prepare,infer,handoff,train,plus,infer_plus (default: all)
#   AF3D_STEPS    FastGS iterations for ArtiFixer3D (30000); FASTGS_ARGS extra train.py flags
#   PY            Python with the ArtiFixer deps (default: python)
#   GS_RENDER_PY  Python with gsplat, when PY has none
#   FASTGS_REPO, FASTGS_PY   FastGS checkout (branch artifixer) and its Python
#   MODEL_ID, CHECKPOINT_PT  Wan2.1-T2V-1.3B-Diffusers dir and artifixer-1.3b.pt
#   NUM_VIEWS (6), INFER_ARGS extra run_inference flags
set -euo pipefail
: "${PLY:?set PLY}" "${COLMAP:?set COLMAP}" "${OUT:?set OUT}"
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
PY=${PY:-python}
STAGES=${STAGES:-prepare,infer,handoff,train,plus,infer_plus}; STAGES=${STAGES//+/,}
AF3D_STEPS=${AF3D_STEPS:-30000}
NUM_VIEWS=${NUM_VIEWS:-6}
: "${MODEL_ID:?set MODEL_ID}" "${CHECKPOINT_PT:?set CHECKPOINT_PT}"
has() { [[ ",$STAGES," == *",$1,"* ]]; }
mkdir -p "$OUT/logs"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/logs/run.log"; }
stage() {  # stage NAME cmd...: run with a per-stage log and time it
  local name=$1; shift; local t0=$SECONDS
  log "start $name"
  "$@" 2>&1 | tee "$OUT/logs/$name.log"
  log "done $name ($((SECONDS - t0)) s)"
}

if [ -n "${TRAJ:-}" ]; then
  PATH_ARGS=(--trajectory_path "$TRAJ"); RENDER_TRAJECTORY=trajectory
else
  PATH_ARGS=(${ANCHORS:+--selected_image_names_file "$ANCHORS"}); RENDER_TRAJECTORY=val_frames
fi
INFER=(-m model_eval.run_inference --evalset reconstructed_colmap --checkpoint_pt "$CHECKPOINT_PT"
       --model_id "$MODEL_ID" --num_views "$NUM_VIEWS" --max_neighbors_per_encode 1 --replace_if_exists
       ${INFER_ARGS:-})
pred_dir() { find "$1" -type d -path "*/frames/batch_0000/pred" | head -1; }

has prepare && stage prepare "$PY" -m artifixer_gs.prepare --ply "$PLY" --colmap_dir "$COLMAP" --output_root "$OUT" \
  "${PATH_ARGS[@]}" ${CAPTION:+--caption "$CAPTION"} ${METRIC_SCALE:+--metric_scale "$METRIC_SCALE"} \
  --text_encoder_model_id "$MODEL_ID"
has infer && stage infer "$PY" "${INFER[@]}" --save_dir "$OUT/artifixer" --split_path "$OUT/split.json" \
  --render_trajectory "$RENDER_TRAJECTORY"

if has handoff; then
  PRED=$(pred_dir "$OUT/artifixer"); [ -n "$PRED" ] || { log "no ArtiFixer predictions under $OUT/artifixer"; exit 1; }
  stage handoff "$PY" -m artifixer_gs.distill handoff --split_path "$OUT/split.json" --artifixer_frames_dir "$PRED" \
    --output_dir "$OUT/af3d/input"
fi
if has train; then
  : "${FASTGS_REPO:?set FASTGS_REPO}" "${FASTGS_PY:?set FASTGS_PY}"
  # -r 1: keep the prepared resolution (FastGS otherwise shrinks images wider than 1600 px).
  stage train bash -c "cd '$FASTGS_REPO' && PYTHONPATH= '$FASTGS_PY' train.py -s '$OUT/af3d/input' -m '$OUT/af3d/fastgs' \
    -r 1 --iterations $AF3D_STEPS --save_iterations $AF3D_STEPS --test_iterations $AF3D_STEPS \
    --lpips_images '$OUT/af3d/input/generated_images.txt' ${FASTGS_ARGS:-}"
fi
AF3D_PLY="$OUT/af3d/fastgs/point_cloud/iteration_$AF3D_STEPS/point_cloud.ply"
has plus && stage plus "$PY" -m artifixer_gs.distill plus --split_path "$OUT/split.json" --ply "$AF3D_PLY" \
  --render_dir "$OUT/af3d/renders" --output_split "$OUT/split_artifixer3d_plus.json"
# ArtiFixer3D+: the second pass regenerates only the targets, conditioned on the ArtiFixer3D renders.
has infer_plus && stage infer_plus "$PY" "${INFER[@]}" --save_dir "$OUT/artifixer3d_plus" \
  --split_path "$OUT/split_artifixer3d_plus.json" --render_trajectory "$RENDER_TRAJECTORY" \
  --neighbor_selection_mode covisibility

log "outputs: ArtiFixer $(pred_dir "$OUT/artifixer" || true)"
log "         ArtiFixer3D PLY $AF3D_PLY ($(cat "$OUT/af3d/fastgs/gs_stats.json" 2>/dev/null | tr -d '\n ' || echo 'no stats'))"
log "         ArtiFixer3D+ $(pred_dir "$OUT/artifixer3d_plus" 2>/dev/null || true)"
