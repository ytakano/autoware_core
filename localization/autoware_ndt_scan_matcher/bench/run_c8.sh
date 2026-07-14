#!/usr/bin/env bash
# C8 parallel-feasibility campaign: {search_00, legal_worst, legal_osc} x k{1,2,4} x {cpp,rust},
# each engine in a SEPARATE process invocation (avoids libgomp pinning the main thread and
# contaminating a subsequent rayon run). Isolated cores 2,4,6,8; per-worker pinning: OpenMP via
# GOMP_CPU_AFFINITY, rayon via NDT_PIN_RAYON_WORKERS (+init_thread_pool). Randomized cell order,
# bracketed by a fixed-work calibration spin (throttle guard).
set -u
B=/autoware_workspace/build/autoware_ndt_scan_matcher
PKG=/autoware_workspace/src/core/autoware_core/localization/autoware_ndt_scan_matcher
OUT=/tmp/claude-1000/-autoware-workspace/71311fc2-8e43-4949-b870-61869e8957ec/scratchpad/c8
rm -rf "$OUT"; mkdir -p "$OUT"
ITERS=${ITERS:-100}; WARMUP=${WARMUP:-10}

calib() {
  taskset -c 2 python3 - <<'PY'
import time
t=time.perf_counter_ns(); x=0
for i in range(60_000_000): x=(x*1103515245+12345)&0xffffffff
print(time.perf_counter_ns()-t, x)
PY
}

cell() { # fixture k engine
  local fx=$1 k=$2 e=$3 mask aff
  case $k in 1) mask=2; aff="2";; 2) mask=2,4; aff="2 4";; 4) mask=2,4,6,8; aff="2 4 6 8";; esac
  if [ "$e" = rust ]; then
    RAYON_NUM_THREADS=$k NDT_PIN_RAYON_WORKERS=1 WCET_ENGINE=rust \
      WCET_ITERS=$ITERS WCET_WARMUP=$WARMUP WCET_THREADS=$k \
      taskset -c $mask "$B/ndt_bench_replay" --fixture "$OUT/${fx}_k${k}_${e}.json" \
        "$PKG/bench/fixtures/${fx}.ndtfix" >>"$OUT/run.log" 2>&1
  else
    OMP_NUM_THREADS=$k GOMP_CPU_AFFINITY="$aff" WCET_ENGINE=cpp \
      WCET_ITERS=$ITERS WCET_WARMUP=$WARMUP WCET_THREADS=$k \
      taskset -c $mask "$B/ndt_bench_replay" --fixture "$OUT/${fx}_k${k}_${e}.json" \
        "$PKG/bench/fixtures/${fx}.ndtfix" >>"$OUT/run.log" 2>&1
  fi
}

echo "calib_before $(calib | awk '{print $1}')" | tee "$OUT/calib.txt"
cells=$(python3 -c "
import random; random.seed(20260714)
c=[(f,k,e) for f in ('search_00','legal_worst','legal_osc') for k in (1,2,4) for e in ('cpp','rust')]
random.shuffle(c); print(' '.join('%s:%d:%s'%x for x in c))")
echo "order: $cells" | tee -a "$OUT/calib.txt"
for spec in $cells; do
  IFS=: read fx k e <<< "$spec"
  echo "  cell $fx k=$k $e"; cell "$fx" "$k" "$e"
done
echo "calib_after $(calib | awk '{print $1}')" | tee -a "$OUT/calib.txt"
echo DONE
