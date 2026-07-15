#!/usr/bin/env python3
"""Merge chunked ndt_bench_replay trace JSON files and validate the sequence domain."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("chunks", nargs="+", type=Path)
    args = parser.parse_args()

    by_seq = {}
    for chunk in args.chunks:
        with chunk.open(encoding="utf-8") as stream:
            document = json.load(stream)
        if document.get("schema_version") != 2:
            raise SystemExit(f"{chunk}: expected schema_version 2")
        for frame in document.get("frames", []):
            seq = frame["seq"]
            if seq in by_seq:
                raise SystemExit(f"duplicate seq {seq} in {chunk}")
            by_seq[seq] = frame

    expected = list(range(args.expected))
    actual = sorted(by_seq)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SystemExit(f"sequence domain mismatch: missing={missing[:10]} extra={extra[:10]}")

    output = {
        "schema_version": 2,
        "frames": [by_seq[seq] for seq in expected],
        "meta": {
            "experiment": "work-shape trace conformance, real-drive open-loop replay",
            "trace_digest": "SHA-256",
            "shape_fields": ["grid_ordinal", "voxel_id"],
            "payload_fields": ["grid_ordinal", "voxel_id", "mean", "inverse_covariance"],
        },
    }
    with args.output.open("w", encoding="utf-8") as stream:
        json.dump(output, stream, separators=(",", ":"))
        stream.write("\n")
    print(f"merged {len(actual)} frames -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
