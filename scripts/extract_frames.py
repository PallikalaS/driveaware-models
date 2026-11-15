#!/usr/bin/env python3
"""Extract Model A training frames and write a pre-filled labels.csv.

What it does
  * team clips (.mp4, named name_activity_condition_tN.mp4): sample at --fps,
    skip the first 3 s of "safe" intro (unless the clip IS safe), crop, resize to
    224x224, drop blurry frames and near-duplicates.
  * public dataset (optional): a folder of class sub-folders with images
    (State Farm: imgs/train/c0..c9; DMD: export one folder per action label).
  * labels.csv gets one row per saved frame with subclass, app_class and a
    subject-based split already filled. Mahitha checks/fixes the subclass column.

Example (from the driveaware-models folder, Drive mounted / synced locally):
  python scripts/extract_frames.py \
      --team "~/Drive/DriveAware/Team_raw" "~/Drive/DriveAware/Team_test_only" \
      --public "~/Drive/DriveAware/StateFarm_raw/imgs/train" --public-source statefarm \
      --out "~/Drive/DriveAware/Frames" --crop dynamic --fps 2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from driveaware_data.crop import FaceFinder, center_crop, dhash, dynamic_crop, hamming, sharpness  # noqa: E402
from driveaware_data.labels import (SUBCLASS_TO_APP, assign_split, find_videos, map_public_label,  # noqa: E402
                                   parse_clip_name)

FIELDS = ["frame_path", "source", "subject", "clip", "t_s", "condition",
          "subclass", "app_class", "split", "crop_mode", "sharpness"]
INTRO_S = 5.0   # default for --skip-start-s: 3 s looking ahead + ~2 s to start the activity
IMG_EXT = {".jpg", ".jpeg", ".png"}


def crop_frame(frame, mode, finder):
    if mode == "center":
        return center_crop(frame), "center"
    out, used_face = dynamic_crop(frame, finder.find(frame))
    return out, ("dynamic" if used_face else "center_fallback")


def load_spans(path) -> dict:
    """Optional activity_spans.csv: clip,start_s,end_s (several rows per clip allowed).
    For a listed clip only frames inside a span are kept, because in some clips the activity
    happens on and off and the frames in between show normal driving."""
    spans: dict = {}
    if not path:
        return spans
    with open(Path(path).expanduser(), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            stem = Path(r["clip"].strip()).stem.lower()
            spans.setdefault(stem, []).append((float(r["start_s"]), float(r["end_s"])))
    return spans


def team_val_clips(paths, held_out, val_subjects, frac: float, seed: int = 663) -> set:
    """Whole clips (never frames of one clip in two splits) per training person go to val, so
    Model A's validation score reflects our phone camera, not only State Farm."""
    import random
    by_person: dict = {}
    for p in paths:
        try:
            c = parse_clip_name(p.name)
        except ValueError:
            continue
        if assign_split(c.name, held_out, val_subjects) == "train":
            by_person.setdefault(c.name, []).append(p)
    out = set()
    if frac <= 0:
        return out
    for person, ps in sorted(by_person.items()):
        ps = sorted(ps)
        random.Random(f"{seed}-{person}").shuffle(ps)
        out.update(ps[: max(1, round(len(ps) * frac))])
    return out


def process_clip(path: Path, out_dir: Path, args, finder, held_out, val_subjects,
                 spans=None, force_val=False):
    clip = parse_clip_name(path.name)
    keep = (spans or {}).get(path.stem.lower())
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"  ! cannot open {path.name}", file=sys.stderr)
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / args.fps))
    rows, last_hash, idx = [], None, 0
    split = assign_split(clip.name, held_out, val_subjects)
    if force_val and split == "train":
        split = "val"
    dest = out_dir / "team" / split / clip.activity
    dest.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / src_fps
        idx += 1
        if (idx - 1) % step:
            continue
        if clip.activity != "safe" and t < args.skip_start_s:
            continue
        if keep is not None and clip.activity != "safe" and not any(a <= t <= b for a, b in keep):
            continue
        img, mode = crop_frame(frame, args.crop, finder)
        sharp = sharpness(img)
        if sharp < args.min_sharpness:
            continue
        h = dhash(img)
        if last_hash is not None and hamming(h, last_hash) <= args.dup_bits:
            continue
        last_hash = h
        name = f"{path.stem}_{t:07.2f}s.jpg"
        cv2.imwrite(str(dest / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append({
            "frame_path": str((dest / name).relative_to(out_dir)), "source": "team",
            "subject": clip.name, "clip": path.name, "t_s": f"{t:.2f}",
            "condition": clip.condition, "subclass": clip.activity,
            "app_class": SUBCLASS_TO_APP[clip.activity], "split": split,
            "crop_mode": mode, "sharpness": f"{sharp:.1f}",
        })
    cap.release()
    return rows


def load_statefarm_subjects(root: Path) -> dict[str, str]:
    """State Farm ships driver_imgs_list.csv (subject,classname,img) next to imgs/."""
    for cand in [root / "driver_imgs_list.csv", root.parent / "driver_imgs_list.csv",
                 root.parent.parent / "driver_imgs_list.csv"]:
        if cand.exists():
            with cand.open() as f:
                return {r["img"]: r["subject"] for r in csv.DictReader(f)}
    return {}


def process_public(root: Path, source: str, out_dir: Path, args, finder, rng_val_frac):
    import random
    subjects = load_statefarm_subjects(root) if source == "statefarm" else {}
    all_subj = sorted(set(subjects.values()))
    random.Random(663).shuffle(all_subj)
    n_val = max(1, int(len(all_subj) * rng_val_frac)) if all_subj else 0
    val_subj = set(all_subj[:n_val])
    rows = []
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sub = map_public_label(source, cls_dir.name)
        if sub is None:
            continue
        imgs = sorted(p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXT)
        if args.public_limit:
            imgs = imgs[: args.public_limit]
        for p in imgs:
            subj = subjects.get(p.name, "unknown")
            # Public data is never test: test = held-out TEAM members only.
            split = "val" if subj in val_subj else "train"
            dest = out_dir / source / split / sub
            dest.mkdir(parents=True, exist_ok=True)
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            img, mode = crop_frame(frame, args.crop, finder)
            cv2.imwrite(str(dest / p.name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({
                "frame_path": str((dest / p.name).relative_to(out_dir)), "source": source,
                "subject": subj, "clip": "", "t_s": "", "condition": "",
                "subclass": sub, "app_class": SUBCLASS_TO_APP[sub], "split": split,
                "crop_mode": mode, "sharpness": f"{sharpness(img):.1f}",
            })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", nargs="*", default=[], help="folders with team .mp4 clips")
    ap.add_argument("--public", help="public dataset root (class sub-folders)")
    ap.add_argument("--public-source", choices=["statefarm", "dmd"], default="statefarm")
    ap.add_argument("--public-limit", type=int, default=0, help="max images per class (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", choices=["dynamic", "center"], default="dynamic")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--min-sharpness", type=float, default=30.0)
    ap.add_argument("--dup-bits", type=int, default=3, help="dHash distance treated as duplicate")
    ap.add_argument("--held-out", default="mahitha,vanessa")
    ap.add_argument("--val-subjects", default="", help="team members used for validation")
    ap.add_argument("--face-model", help="optional MediaPipe blaze_face_short_range.tflite")
    ap.add_argument("--skip-start-s", type=float, default=INTRO_S,
                    help="drop the first seconds of every non-safe clip (people look ahead first)")
    ap.add_argument("--spans", help="optional activity_spans.csv (clip,start_s,end_s): keep only those times")
    ap.add_argument("--team-val-frac", type=float, default=0.15,
                    help="share of each training person's clips (whole clips) used as validation")
    args = ap.parse_args(argv)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    held_out = {s.strip().lower() for s in args.held_out.split(",") if s.strip()}
    val_subjects = {s.strip().lower() for s in args.val_subjects.split(",") if s.strip()}
    finder = FaceFinder(args.face_model)

    rows = []
    spans = load_spans(args.spans)
    all_clips = [p for folder in args.team for p in find_videos(folder)]
    val_clips = team_val_clips(all_clips, held_out, val_subjects, args.team_val_frac)
    if val_clips:
        print(f"team validation clips ({len(val_clips)}): " + ", ".join(sorted(p.name for p in val_clips)))
    for folder in args.team:
        for p in find_videos(folder):
            print(f"clip {p.name}")
            try:
                rows += process_clip(p, out_dir, args, finder, held_out, val_subjects,
                                     spans=spans, force_val=p in val_clips)
            except ValueError as e:
                print(f"  ! skipped: {e}", file=sys.stderr)
    if args.public:
        print(f"public dataset {args.public} ({args.public_source})")
        rows += process_public(Path(args.public).expanduser(), args.public_source, out_dir, args, finder, 0.15)

    labels = out_dir / "labels.csv"
    with labels.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # per-class counts (commit this file to the repo, not the images)
    counts: dict[tuple[str, str], int] = {}
    for r in rows:
        counts[(r["split"], r["subclass"])] = counts.get((r["split"], r["subclass"]), 0) + 1
    with (out_dir / "class_counts.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "subclass", "frames"])
        for (s, c), n in sorted(counts.items()):
            w.writerow([s, c, n])
    print(f"\n{len(rows)} frames → {labels}")
    for (s, c), n in sorted(counts.items()):
        print(f"  {s:5s} {c:16s} {n}")


if __name__ == "__main__":
    main()
