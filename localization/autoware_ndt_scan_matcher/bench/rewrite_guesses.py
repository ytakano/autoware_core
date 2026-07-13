#!/usr/bin/env python3
"""Rewrite a capture directory's per-frame guesses from a frozen guess track (stdlib-only).

The open-loop real-data protocol (paper Sec. V realdata; plan/ndt_timing_measurement_policy.md)
replays the drive with a FROZEN NDT-odometry guess track so both engines see identical,
EKF-independent inputs. The capture sidecar frames carry the original production guesses; this
tool produces a rewritten copy whose frame guesses (the leading 16 x f32 = 64 bytes of each
frame_<seq>.bin) come from the track file (64 bytes/frame, row-major f32, seq order — the
WCET_GUESS_OUT dump format). tiles/ and params.bin are symlinked, frames are copied+patched.

Usage: rewrite_guesses.py <capture_dir> <guess_track.bin> <out_dir>
"""

import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    src = pathlib.Path(sys.argv[1])
    track_path = pathlib.Path(sys.argv[2])
    dst = pathlib.Path(sys.argv[3])
    track = track_path.read_bytes()
    frames = sorted((src / "frames").iterdir())
    if len(track) != 64 * len(frames):
        print(
            f"track/frame count mismatch: {len(track)} bytes for {len(frames)} frames",
            file=sys.stderr,
        )
        return 1
    (dst / "frames").mkdir(parents=True, exist_ok=True)
    for name in ("params.bin", "tiles"):
        link = dst / name
        if not link.exists():
            link.symlink_to(src / name)
    for fp in frames:
        seq = int(fp.stem.split("_")[1])
        data = fp.read_bytes()
        (dst / "frames" / fp.name).write_bytes(track[seq * 64 : (seq + 1) * 64] + data[64:])
    print(f"rewrote {len(frames)} frames -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
