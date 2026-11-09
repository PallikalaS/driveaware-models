#!/usr/bin/env python3
"""Side-by-side: full frame (with box) | center crop | dynamic upper-body crop.

Mahitha runs this on 3-4 clips (include texting held low) and posts the
images on Discord, then the team picks the crop for extract_frames.py.

  python scripts/crop_compare.py clip1.mp4 clip2.mp4 --out results/crop_compare --n 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.crop import FaceFinder, center_crop, dynamic_crop, upper_body_box  # noqa: E402

PANEL = 320


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def compare(frame, finder):
    face = finder.find(frame)
    full = frame.copy()
    h, w = full.shape[:2]
    # show where the center crop would be
    s = min(h, w)
    cv2.rectangle(full, ((w - s) // 2, (h - s) // 2), ((w + s) // 2, (h + s) // 2), (0, 200, 255), 2)
    if face:
        b = upper_body_box(face, w, h)
        cv2.rectangle(full, (face.x, face.y), (face.x + face.w, face.y + face.h), (255, 120, 0), 2)
        cv2.rectangle(full, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 220, 0), 2)
    scale = PANEL / h
    full = cv2.resize(full, (int(w * scale), PANEL))
    cc = cv2.resize(center_crop(frame), (PANEL, PANEL))
    dc, used = dynamic_crop(frame, face)
    dc = cv2.resize(dc, (PANEL, PANEL))
    return np.hstack([
        label(full, "full: orange=center, green=dynamic"),
        label(cc, "center crop"),
        label(dc, "dynamic crop" if used else "dynamic (NO FACE -> center)"),
    ])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--out", default="results/crop_compare")
    ap.add_argument("--n", type=int, default=4, help="frames per clip")
    ap.add_argument("--face-model")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    finder = FaceFinder(args.face_model)
    for clip in args.clips:
        cap = cv2.VideoCapture(clip)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        for i, pos in enumerate(np.linspace(total * 0.15, total * 0.9, args.n).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
            ok, frame = cap.read()
            if not ok:
                continue
            dst = out / f"{Path(clip).stem}_{i}.jpg"
            cv2.imwrite(str(dst), compare(frame, finder))
            print(dst)
        cap.release()


if __name__ == "__main__":
    main()
