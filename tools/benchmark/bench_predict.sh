#!/usr/bin/env bash
# Time `subcellseg predict` on the variance example images and compare with SEG_STRUCT.
# usage: scripts/bench_predict.sh <device: cuda|cpu> <output dir> [image ids...]
# Needs `subcellseg` (the install under test) and the venv with bioio-ome-zarr/matplotlib
# for compare_to_gt.py; set COMPARE_PY to that venv's python (default: SubCellSeg/.venv).
set -u
dev=$1; out=$2; shift 2; ids=${@:-40 6919 11370}
here=$(cd "$(dirname "$0")" && pwd)
COMPARE_PY=${COMPARE_PY:-$here/../../.venv/bin/python}
base=https://animatedcell-test-data.s3.us-west-2.amazonaws.com/variance
mkdir -p "$out"; log=$out/bench_$dev.log; : > "$log"
echo "host $(hostname)  $(date -Is)  device=$dev" | tee -a "$log"
echo "subcellseg=$(command -v subcellseg)" | tee -a "$log"
subcellseg info 2>&1 | tee -a "$log"
for n in $ids; do
  echo "=== $n ===" | tee -a "$log"
  # sample GPU memory every 0.5 s while predict runs
  if [ "$dev" != cpu ]; then
    ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.5; done ) > "$out/${n}_gpumem.txt" 2>/dev/null &
    mon=$!
  fi
  s=$(date +%s.%N)
  subcellseg predict --input "$base/$n.zarr" --channel "Labeled Structure" --output "$out" --save-prob --device "$dev" --overwrite 2>&1 | tee -a "$log"
  e=$(date +%s.%N)
  [ "$dev" != cpu ] && { kill $mon 2>/dev/null; wait $mon 2>/dev/null; peak=$(sort -n "$out/${n}_gpumem.txt" | tail -1); } || peak=NA
  printf "RESULT %s predict_wall_s=%.1f peak_gpu_mem_MiB=%s\n" "$n" "$(echo "$e - $s" | bc)" "$peak" | tee -a "$log"
  "$COMPARE_PY" "$here/compare_to_gt.py" "$base/$n.zarr" "$out/${n}_seg.tif" "$out/${n}_prob.tif" --png "$out/${n}_compare.png" 2>&1 | tee -a "$log"
done
echo "done $(date -Is)" | tee -a "$log"
