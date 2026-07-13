# WCET Measurement Campaign (Profiles A/B)

Orchestrates the controlled-engine measurement campaign of
`plan/ndt_timing_measurement_policy.md` around `ndt_bench_replay --fixture`:
environment verification with abort rules, warm/cold series, one-resource-at-a-time
co-runners, randomized fixture/engine order, sample tiers, sessions, and the policy's
reproducibility manifest in every output JSON. Configuration: `campaign_config.json`
(benchmark CPU, tiers, budgets, abort thresholds — `abort.nominal_khz` is host-specific).

## One-time host preparation (root)

```sh
# performance governor on the benchmark CPU (repeat for its siblings if desired)
echo performance | sudo tee /sys/devices/system/cpu/cpu2/cpufreq/scaling_governor
# offline the SMT sibling of the benchmark CPU
echo 0 | sudo tee /sys/devices/system/cpu/cpu3/online
# core isolation: add isolcpus=2 (and nohz_full=2 rcu_nocbs=2) to the kernel cmdline, or
#   sudo cset shield --cpu 2 --kthread=on
# core isolation (preferred): isolcpus=2 nohz_full=2 rcu_nocbs=2 on the kernel cmdline +
# reboot. NOTE: governor and the sibling-offline setting reset on reboot -- re-apply them;
# verify-env will catch it. (cset does not work on cgroup-v2 systems.)
# move IRQ affinities away from cpu2 (best-effort loop; cpu3 offline -> 0-1,4-15):
for irq in /proc/irq/[0-9]*; do echo 0-1,4-15 | sudo tee $irq/smp_affinity_list >/dev/null 2>&1 || true; done
```

`verify-env` names the exact knob for anything still wrong. Per-cell abort rules: a
**calibration guard** (fixed-work FP spin on the benchmark CPU, compared to the session
baseline; `abort.max_calib_slowdown`) is the primary throttling detector — on
nohz_full/isolated cores the kernel's frequency reporting goes stale (this host reports a
constant 400 MHz under load while the core actually boosts), so `scaling_cur_freq` is
recorded as telemetry and only judged when the readings vary (`abort.min_freq_frac` ×
`abort.nominal_khz`); plus a thermal ceiling (`abort.max_temp_c`).

## Build

```sh
colcon build --packages-select autoware_ndt_scan_matcher \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DNDT_USE_RUST=ON -DNDT_BUILD_BENCH=ON
```

## Run

```sh
cd src/core/autoware_core/localization/autoware_ndt_scan_matcher/bench
python3 wcet_campaign.py verify-env          # must pass (exit 0) before a real session
python3 wcet_campaign.py plan                # run matrix + duration estimate
python3 wcet_campaign.py run --session 1     # repeat for sessions 2..N on other days
python3 wcet_campaign.py merge campaign_runs/session-1
```

Raw per-cell JSONs and `.cell.json` sidecars (env snapshots, in-series frequency/thermal
samples, per-cell manifest) are preserved under `campaign_runs/<session>/<series>/`; `merge`
pairs the single-engine cells per fixture, re-asserts iteration equality (the fairness
requirement), and writes one `wcet.json`-shaped file per series with `meta.manifest`.

Replay-side controls (also usable standalone): `WCET_ENGINE=cpp|rust|both`,
`WCET_EVICT_BYTES=N` (between-sample cache-eviction approximation for the cold series),
plus the existing `WCET_ITERS` / `WCET_WARMUP`.

Dev smoke test (container, unprepared host): `run --allow-env-mismatch --max-cells N` with a
reduced config — the override is recorded in the manifest as `env_check: "overridden"` and
such data must never feed the paper.

## Profile A (production-representative) and the bridge experiment

`campaign_config_profileA.json` sets `"profile": "A"`: verify-env then *inverts* the
isolation checks (isolation must be **absent**, SMT online; a present `isolcpus` is a
failure), cells run **unpinned** under normal CFS, and frequency/calibration excursions are
recorded in the `.cell.json` sidecar (`profile_a_recorded_excursions`) instead of aborting
(the thermal ceiling and replay failures still abort). The manifest carries
`measurement_profile: "A"` and `affinity_mask: none (CFS placement)`. Keep the same
frequency pin as Profile B so A/B ratios isolate scheduling, not speed regimes.

The bridge legs (policy: Bridge Experiment) are assembled into `paper/data/bridge.json` by
`paper/scripts/assemble_bridge.py <b_leg_warm.json>`; the two real-drive inputs are frozen
from the archived capture via the engine example
(`wcet_frame --freeze <capture_dir> <seq> <guess_track.bin> <out.ndtfix>`; capture archived
at `~/autoware_ista_data/`). Watch for transient EC/firmware clamps around the 4.65M-point
map builds of the real fixtures — the calibration guard aborts the series; wait until the
fixed-work spin returns to the session baseline and re-run (a gated retry loop suffices).
