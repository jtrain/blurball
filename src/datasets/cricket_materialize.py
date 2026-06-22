"""Materialize cricket clips from annotated videos into the WASB/Tennis layout.

The annotation pipeline (``label_ball.py``) keeps every frame inside the source
video and records only ball positions in ``annotations.json``. The training code
(``datasets/tennis.py``, which ``cricket`` reuses) instead expects each clip as a
folder of integer-named frames plus a ``Label.csv``::

    <root>/<match>/<clip>/0000.png
                          0001.png
                          ...
                          Label.csv   # file name,visibility,x-coordinate,y-coordinate

This module bridges the two: it decodes only the frames of the clips you ask for,
writes them in the expected layout, and is idempotent so frames stay inside the
videos until a clip is actually selected for a run.

Annotation conventions (see label_ball.py):
  * a clip key is ``"<video>:<start_frame>"``; ``start_frame`` is absolute in the video.
  * each label's ``frame`` is a *local* offset, so absolute frame = start_frame + frame.
  * ``vis``: 0 = no ball, 1 = visible, 2 = motion-blurred. 1 and 2 are "ball present"
    and line up with ``visible_flags: [1, 2]`` in the dataset config.
  * coordinates are in native video resolution (no resize), matching the frames we decode.

Frame selection within a clip (this is the important part):
  * A frame is only exported if it carries an explicit label. Frames with *no*
    label are "undecided" (the ball may well be visible, we just never adjudicated
    them) and must never be invented as vis=0 -- they are excluded.
  * Because the model stacks ``frames_in`` *consecutive* frames, an undecided frame
    is a hard boundary: it splits the clip into separate runs so we never stitch
    frames that have a real, unlabelled frame between them.
  * A "duplicate" frame (label_ball's ``duplicate_frames``) is the same image shown
    twice -- no new content. It is skipped, but it is NOT a boundary: the frames on
    either side are the true consecutive captures and are joined into one run
    (e.g. 7, 8, [9=dup], 10  ->  run 7, 8, 10).
  * Each maximal run of consecutively-labelled frames (duplicates removed) becomes
    its own clip dir; runs shorter than ``min_frames`` are dropped.

CLI:
    python -m datasets.cricket_materialize \\
        --clip "batting002.mkv:341"  --match batting002_mkv_341 \\
        --clip "batting002.mkv:5067" --match batting002_mkv_5067 \\
        --out-root /path/to/datasets/cricket
"""

import argparse
import json
import logging
import os
import os.path as osp
import re

import cv2
import pandas as pd

log = logging.getLogger(__name__)

CSV_COLUMNS = ["file name", "visibility", "x-coordinate", "y-coordinate"]


def _sanitize(name: str) -> str:
    """Turn a clip key like 'batting002.mkv:341' into a filesystem-safe token."""
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def _build_runs(clip: dict, ext: str, min_frames: int):
    """Split a clip into runs of consecutively-labelled frames.

    Returns a list of ``(abs_start_frame, rows, abs_indices)`` tuples, one per run:
      * ``abs_start_frame`` -- absolute video frame the run starts at (for naming).
      * ``rows`` -- Label.csv rows, frames reindexed 0..N-1 within the run.
      * ``abs_indices`` -- absolute video frame index for each row, in order.

    Duplicates are removed without breaking a run; undecided (unlabelled) frames
    break the run and are never emitted.
    """
    labels = {int(l["frame"]): l for l in clip["labels"]}
    if not labels:
        raise ValueError("clip has no labels")
    dups = set(clip.get("duplicate_frames", []))
    start_frame = clip["start_frame"]
    lo, hi = min(labels), max(labels)

    runs, current = [], []
    for local in range(lo, hi + 1):
        if local in dups:
            continue  # duplicate frame: skip, but do NOT break the run
        lab = labels.get(local)
        if lab is None:
            # undecided frame (real content, no decision): hard boundary
            if current:
                runs.append(current)
                current = []
            continue
        current.append((local, lab))
    if current:
        runs.append(current)

    out = []
    for run in runs:
        if len(run) < min_frames:
            continue
        rows, abs_indices = [], []
        for out_idx, (local, lab) in enumerate(run):
            if lab["vis"] > 0 and lab["x"] is not None:
                vis, x, y = int(lab["vis"]), float(lab["x"]), float(lab["y"])
            else:
                vis, x, y = 0, 0.0, 0.0  # explicit "no ball" decision
            rows.append(
                {
                    "file name": f"{out_idx:04d}{ext}",
                    "visibility": vis,
                    "x-coordinate": x,
                    "y-coordinate": y,
                }
            )
            abs_indices.append(start_frame + local)
        out.append((start_frame + run[0][0], rows, abs_indices))
    return out


def materialize_clip(
    clip_key: str,
    annotations: dict,
    vids_dir: str,
    out_root: str,
    match: str = None,
    ext: str = ".png",
    min_frames: int = 3,
    force: bool = False,
):
    """Extract one annotated clip into ``out_root/<match>/<run>/`` sub-clips.

    A single annotated clip may yield several runs (split at undecided frames),
    each written as its own clip dir named by the run's absolute start frame.
    Returns the list of clip dirs written/kept. Skips already-materialized runs
    (unless ``force``), so this is safe to call lazily before every run.
    """
    if clip_key not in annotations:
        raise KeyError(f"clip key not found in annotations: {clip_key}")
    clip = annotations[clip_key]
    match = match or _sanitize(clip_key)

    runs = _build_runs(clip, ext, min_frames)
    if not runs:
        log.warning("no runs >= %d frames for %s; nothing to export", min_frames, clip_key)
        return []

    video_path = osp.join(vids_dir, clip["video"])
    if not osp.isfile(video_path):
        raise FileNotFoundError(f"source video not found: {video_path}")

    clip_dirs = []
    cap = None
    try:
        for abs_start, rows, abs_indices in runs:
            clip_name = f"{abs_start:06d}"
            clip_dir = osp.join(out_root, match, clip_name)
            csv_path = osp.join(clip_dir, "Label.csv")
            if osp.isfile(csv_path) and not force:
                log.info("skip (already materialized): %s", clip_dir)
                clip_dirs.append(clip_dir)
                continue

            if cap is None:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"cannot open video: {video_path}")
            os.makedirs(clip_dir, exist_ok=True)

            next_abs = None
            for row, abs_idx in zip(rows, abs_indices):
                # Frames within a run skip over duplicates, so seek when not adjacent.
                if abs_idx != next_abs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_idx)
                ok, frame = cap.read()
                next_abs = abs_idx + 1
                if not ok or frame is None:
                    raise RuntimeError(f"failed to read frame {abs_idx} from {video_path}")
                cv2.imwrite(osp.join(clip_dir, row["file name"]), frame)

            pd.DataFrame(rows, columns=CSV_COLUMNS).to_csv(csv_path, index=False)
            log.info("materialized %d frames -> %s", len(rows), clip_dir)
            clip_dirs.append(clip_dir)
    finally:
        if cap is not None:
            cap.release()

    return clip_dirs


def list_clips(annotations: dict, vids_dir: str, ext: str = ".png", min_frames: int = 3):
    """Return [(key, match, video, video_ok, n_runs, n_frames)] for every clip.

    ``n_runs``/``n_frames`` reflect the same run-splitting used by
    ``materialize_clip``, so the counts match exactly what would be exported.
    """
    out = []
    for key, clip in annotations.items():
        video_ok = osp.isfile(osp.join(vids_dir, clip["video"]))
        try:
            runs = _build_runs(clip, ext, min_frames) if video_ok else []
        except (ValueError, KeyError):
            runs = []
        n_frames = sum(len(rows) for _, rows, _ in runs)
        out.append((key, _sanitize(key), clip["video"], video_ok, len(runs), n_frames))
    return out


def _matches_from_config(config_path: str):
    """Read train + test match names from a dataset yaml, in order, de-duplicated."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    names = []
    for split in ("train", "test"):
        for m in cfg.get(split, {}).get("matches", []) or []:
            if m not in names:
                names.append(str(m))
    return names


def _print_clip_list(annotations, vids_dir, ext, min_frames, show_all):
    rows = list_clips(annotations, vids_dir, ext=ext, min_frames=min_frames)
    usable = [r for r in rows if r[3] and r[4] > 0]
    shown = rows if show_all else usable
    shown = sorted(shown, key=lambda r: (-r[5], r[1]))

    print(
        f"{'match name (for cricket.yaml)':40s} {'runs':>4} {'frames':>6}  video"
    )
    print("-" * 78)
    for key, match, video, ok, nruns, nfr in shown:
        flag = "" if (ok and nruns > 0) else "  [skip: no video]" if not ok else "  [skip: <min]"
        print(f"{match:40s} {nruns:>4} {nfr:>6}  {video}{flag}")
    print("-" * 78)
    print(
        f"total clips: {len(rows)} | usable: {len(usable)} | "
        f"missing video: {sum(1 for r in rows if not r[3])}"
    )


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        default=osp.expanduser(
            "~/Documents/CricketVideos/splitting/dataset/annotations.json"
        ),
    )
    parser.add_argument(
        "--vids",
        default=osp.expanduser("~/Documents/CricketVideos/splitting/vids"),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list available clips (match names + run/frame counts) and exit",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="with --list, also show unusable clips (missing video / too short)",
    )
    parser.add_argument("--out-root")
    parser.add_argument(
        "--config",
        default=osp.join(
            osp.dirname(__file__), "..", "configs", "dataset", "cricket.yaml"
        ),
        help="dataset yaml whose train/test matches are materialized when no "
        "--clip is given (default: configs/dataset/cricket.yaml)",
    )
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        dest="clips",
        help="clip key 'video:start_frame'; repeatable. If omitted, the matches "
        "listed in --config are materialized instead.",
    )
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        dest="matches",
        help="match dir name for the matching --clip (optional, repeatable)",
    )
    parser.add_argument("--ext", default=".png")
    parser.add_argument(
        "--min-frames",
        type=int,
        default=3,
        help="drop runs shorter than this (defaults to model frames_in=3)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with open(args.annotations) as f:
        annotations = json.load(f)

    if args.list:
        _print_clip_list(annotations, args.vids, args.ext, args.min_frames, args.all)
        return

    if not args.out_root:
        parser.error("--out-root is required when materializing (omit only with --list)")

    if args.clips:
        # explicit clip keys (optionally with matching --match names)
        clip_keys = list(args.clips)
        match_names = [
            args.matches[i] if i < len(args.matches) else None
            for i in range(len(clip_keys))
        ]
    else:
        # default: materialize exactly the matches referenced by the config split
        match_names = _matches_from_config(args.config)
        by_match = {_sanitize(k): k for k in annotations}
        clip_keys = []
        kept_matches = []
        for m in match_names:
            if m not in by_match:
                log.warning("match %r in config has no matching clip; skipping", m)
                continue
            clip_keys.append(by_match[m])
            kept_matches.append(m)
        match_names = kept_matches
        log.info(
            "materializing %d matches from %s", len(clip_keys), osp.abspath(args.config)
        )

    for clip_key, match in zip(clip_keys, match_names):
        materialize_clip(
            clip_key=clip_key,
            annotations=annotations,
            vids_dir=args.vids,
            out_root=args.out_root,
            match=match,
            ext=args.ext,
            min_frames=args.min_frames,
            force=args.force,
        )


if __name__ == "__main__":
    _main()
