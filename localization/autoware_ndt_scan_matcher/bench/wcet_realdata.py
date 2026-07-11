#!/usr/bin/env python3
# Copyright 2024 Autoware Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Merge the two real-drive capture replays into one per-frame dataset (stdlib-only).

Inputs:
  rust.json  from `wcet_frame --capture DIR` (WCET_JSON): per-frame counters + in-crate ms
  cpp.json   from `ndt_bench_replay --capture out.json DIR`: per-frame cpp_ms/rust_ms + iter match

Output (-o): realdata.json  { "frames": [ {seq, n_source, n_tiles, iteration_num, counters...,
             cpp_ms, rust_ms (replay binary, same harness as cpp), rust_ms_crate, match} ] }

Also prints the operational-envelope summary (P / iter / K̄ / max-K / kd-per-pt / ms
distributions + equal-work match rate) for a quick sanity read.
"""

import argparse
import json
import sys


def pct(sorted_xs, p):
    return sorted_xs[min(int(p * (len(sorted_xs) - 1)), len(sorted_xs) - 1)]


def dist(name, xs, unit=""):
    xs = sorted(xs)
    print(
        f"  {name:12} p50 {pct(xs, 0.5):10.2f}  p99 {pct(xs, 0.99):10.2f}  "
        f"max {xs[-1]:10.2f} {unit}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rust_json")
    ap.add_argument("cpp_json")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    with open(args.rust_json, encoding="utf-8") as fh:
        rust = {f["seq"]: f for f in json.load(fh)["frames"]}
    with open(args.cpp_json, encoding="utf-8") as fh:
        cpp = {f["seq"]: f for f in json.load(fh)["frames"]}

    frames = []
    mismatch = 0
    for seq in sorted(rust):
        r = rust[seq]
        c = cpp.get(seq)
        if c is None:
            continue
        if r["iteration_num"] != c["iter_rust"]:
            # The two replays must agree with themselves (same map order, same inputs).
            print(f"WARN: rust replay iter differs between passes at seq {seq}", file=sys.stderr)
        if not c["match"]:
            mismatch += 1
        fr = {
            "seq": seq,
            "n_source": r["n_source"],
            "n_tiles": r["n_tiles"],
            "iteration_num": r["iteration_num"],
            "counters": r.get("counters"),
            "cpp_ms": c["cpp_ms"],
            "rust_ms": c["rust_ms"],
            "rust_ms_crate": r["ms"],
            "match": c["match"],
        }
        frames.append(fr)

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"frames": frames}, fh)
    n = len(frames)
    print(f"merged {n} frames -> {args.out}; equal-work: {n - mismatch}/{n}")

    if frames and frames[0]["counters"]:
        dist("P", [f["n_source"] for f in frames], "pts")
        dist("iterations", [f["iteration_num"] for f in frames])
        dist(
            "K-bar",
            [
                f["counters"]["sum_neighbors"] / max(f["counters"]["points_processed"], 1)
                for f in frames
            ],
        )
        dist("max-K", [f["counters"]["max_neighbors"] for f in frames])
        dist(
            "kd/pt",
            [
                f["counters"]["kd_nodes_visited"] / max(f["counters"]["points_processed"], 1)
                for f in frames
            ],
        )
        dist("cpp", [f["cpp_ms"] for f in frames], "ms")
        dist("rust", [f["rust_ms"] for f in frames], "ms")
    return 1 if mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
