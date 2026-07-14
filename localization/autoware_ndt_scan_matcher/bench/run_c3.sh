#!/usr/bin/env bash
# C3 Profile-A synthetic re-measurement at matched n (plan/paper_fix2.md): the three synthetic
# bridge inputs (search_00, legal_worst, legal_osc) under Profile A (production-representative:
# CFS, unpinned, SMT on, 3.2 GHz reference clock -- NO taskset), 3 sessions x 1000 samples
# (pooled n=3000, matching the Profile-B synthetic B-leg), so the bridge A/B max comparison is
# same-n. Each session is bracketed by a fixed-work calibration spin (throttle guard).
set -u
B=/autoware_workspace/build/autoware_ndt_scan_matcher
PKG=/autoware_workspace/src/core/autoware_core/localization/autoware_ndt_scan_matcher
OUT=/tmp/claude-1000/-autoware-workspace/71311fc2-8e43-4949-b870-61869e8957ec/scratchpad/c3
rm -rf "$OUT"; mkdir -p "$OUT"
ITERS=${ITERS:-1000}; WARMUP=${WARMUP:-10}; SESSIONS=${SESSIONS:-3}

calib() {
  python3 - <<'PY'
import time
t=time.perf_counter_ns(); x=0
for i in range(60_000_000): x=(x*1103515245+12345)&0xffffffff
print(time.perf_counter_ns()-t)
PY
}

for s in $(seq 1 "$SESSIONS"); do
  echo "session $s calib_before $(calib)" | tee -a "$OUT/calib.txt"
  # Profile A: no taskset (CFS placement), both engines, one invocation over the 3 synthetics.
  WCET_ITERS=$ITERS WCET_WARMUP=$WARMUP WCET_ENGINE=both \
    "$B/ndt_bench_replay" --fixture "$OUT/session_${s}.json" \
      "$PKG/bench/fixtures/search_00.ndtfix" \
      "$PKG/bench/fixtures/legal_worst.ndtfix" \
      "$PKG/bench/fixtures/legal_osc.ndtfix" >>"$OUT/run.log" 2>&1
  echo "session $s calib_after $(calib)" | tee -a "$OUT/calib.txt"
done

# Pool the sessions and merge with the original Profile-A real-frame legs into the A-leg doc
# consumed by paper/scripts/assemble_bridge.py (campaign_runs is gitignored; bridge.json is the
# committed artifact).
python3 - "$OUT" "$PKG" <<'PY'
import json, glob, subprocess, sys, os
out, pkg = sys.argv[1], sys.argv[2]
pool = {}
for f in sorted(glob.glob(f"{out}/session_*.json")):
    for fx, v in json.load(open(f))["fixtures"].items():
        for eng in ("cpp", "rust"):
            pool.setdefault((fx, eng), []).extend(v[eng]["samples_ms"])
sess1 = json.load(open(f"{out}/session_1.json"))
old = json.load(open(f"{pkg}/bench/campaign_runs/profileA/session-1/warm.json"))
merged = {"benchmark": "WCET fixture replay (Profile A, C3 matched-n synthetics)",
          "meta": dict(old["meta"])}
merged["meta"]["iters"] = "synthetics 3x1000 pooled (n=3000); real frames n=100 (session-1)"
mani = dict(old["meta"].get("manifest", {}))
mani["experiment_id"] = "profileA/c3_pooled"
mani["run_timestamp"] = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
                                        capture_output=True, text=True).stdout.strip()
mani["note"] = ("C3: synthetic bridge inputs re-measured at matched n=3000 (3 sessions x 1000, "
                "CFS/unpinned, SMT on, 3.2 GHz); real-frame legs copied from the original "
                "Profile-A session-1 (n=100, already matched to their B-leg).")
merged["meta"]["manifest"] = mani
merged["fixtures"] = {}
for fx in ("search_00", "legal_worst", "legal_osc"):
    scaf = sess1["fixtures"][fx]
    e = {k: scaf[k] for k in ("n_map", "n_tiles", "n_source", "max_iterations", "iter_match")
         if k in scaf}
    for eng in ("cpp", "rust"):
        e[eng] = {"iteration_num": scaf[eng]["iteration_num"],
                  "allocs_per_align": scaf[eng].get("allocs_per_align", -1),
                  "samples_ms": pool[(fx, eng)]}
    merged["fixtures"][fx] = e
for fx in ("real_slowest", "real_median"):
    merged["fixtures"][fx] = old["fixtures"][fx]
dst = f"{pkg}/bench/campaign_runs/profileA/c3_pooled"
os.makedirs(dst, exist_ok=True)
json.dump(merged, open(f"{dst}/warm.json", "w"))
print(f"wrote {dst}/warm.json")
PY
echo DONE | tee -a "$OUT/calib.txt"
echo "next: paper/scripts/assemble_bridge.py <bridgeB/session-3/warm.json> ; make tables"
