"""
Week 4: Tracking
Adds a tracker (ByteTrack) to the Week 2 person detector.

What it produces:
  output_tracked.mp4      video with a box, "ID <n> <conf>" label and path trail per person
  track_log.csv           one row per person per frame (frame, time, ID, confidence, box)
  id_summary.csv          one row per ID (first frame, last frame, how long it was seen)
  switch_candidates/      snapshots of frames where a likely ID switch happened
"""

import csv
import os
import shutil
import subprocess
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output_tracked.mp4"
LOG_CSV = "track_log.csv"
SUMMARY_CSV = "id_summary.csv"
SWITCH_DIR = "switch_candidates"

MODEL_PATH = "yolo11n.pt"     # Week 3: fast enough to run on every frame
TRACKER = "bytetrack.yaml"    # Week 4 slide 8. The Ultralytics default is BoT-SORT ("botsort.yaml")
PERSON_CLASS_ID = 0           # COCO class 0 = person

# We do NOT pass conf=0.4 (our Week 3 threshold) to model.track().
# ByteTrack's second step matches LOW-confidence boxes to tracks that were left over,
# which keeps partly hidden people from being dropped. Cutting boxes at 0.4 before the
# tracker would switch that step off. Ultralytics sends boxes down to 0.1 to the tracker,
# and ByteTrack only STARTS a new ID from a box with confidence >= 0.25.

DRAW_TRAILS = True            # draw each person's path over the last TRAIL_LENGTH frames
TRAIL_LENGTH = 40
SHOW_WINDOW = False           # True = watch it live while it runs (press q to stop)

# ID switch search: a NEW ID that appears close to where another ID was LOST a moment ago,
# with a box of similar size (so a small far-away person is not paired with a big close one)
SWITCH_LOOKBACK_SECONDS = 3.0
SWITCH_MAX_DISTANCE = 0.75    # center distance, in average box heights
SWITCH_MAX_SIZE_RATIO = 1.5   # the taller box may be at most 1.5x the shorter one

# Tracks shorter than this are reported separately; they are usually flicker or broken tracks
SHORT_TRACK_SECONDS = 1.0


# ==========================================
# VIDEO FORMAT HELPERS
# ".mp4" is only the container. The video inside can use a codec OpenCV cannot read
# (often HEVC/H.265 from phones), and OpenCV's own "mp4v" output does not play in
# browsers, VS Code or some Windows players. A bundled ffmpeg (imageio-ffmpeg) fixes both.
# ==========================================
def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def to_h264(ffmpeg, src, dst):
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", src,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           "-pix_fmt", "yuv420p", "-an", dst]
    return subprocess.run(cmd).returncode == 0 and os.path.exists(dst)


def open_readable_video(path):
    """Opens the video. If OpenCV cannot decode it, converts it to H.264 first."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find '{path}'. Put your video in this folder and name it {INPUT_VIDEO}.")

    cap = cv2.VideoCapture(path)
    ok, _ = cap.read() if cap.isOpened() else (False, None)
    if ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return cap, path
    cap.release()

    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError(f"OpenCV cannot read '{path}' (unsupported codec). "
                           "Run: pip install imageio-ffmpeg   and try again.")
    fixed = os.path.splitext(path)[0] + "_h264.mp4"
    print(f"[INFO] OpenCV cannot read '{path}'. Converting it to H.264 -> {fixed}")
    if not to_h264(ffmpeg, path, fixed):
        raise RuntimeError(f"Could not convert '{path}'. Try another video.")
    return cv2.VideoCapture(fixed), fixed


def color_for_id(track_id):
    """A fixed, bright color per ID, so the same person keeps the same color."""
    hue = (track_id * 37) % 180
    bgr = cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_label(frame, text, x, y, color, s=1.0):
    """Label with a filled background. s scales text for big videos (s = 3 for 4K)."""
    font_scale, thick, pad = 0.6 * s, max(1, int(2 * s)), int(4 * s)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
    y = max(y, th + 2 * pad)
    cv2.rectangle(frame, (x, y - th - 2 * pad), (x + tw + pad, y + pad), color, cv2.FILLED)
    cv2.putText(frame, text, (x + pad // 2, y - pad), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick)


# ==========================================
# MAIN TRACKING PIPELINE
# ==========================================
def run_person_tracking(input_video_path=INPUT_VIDEO, output_video_path=OUTPUT_VIDEO):
    # 1. Load the pretrained YOLO11 Nano model (same as Week 2)
    model = YOLO(MODEL_PATH)

    # 2. Open the input video
    cap, input_video_path = open_readable_video(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps > 240:   # missing or broken metadata
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = max(1.0, min(width, height) / 720)          # drawing scale: 1 for 720p/1080p, 3 for 4K
    line = max(2, int(2 * s))

    # 3. Video writer. OpenCV writes "mp4v"; at the end it is re-encoded to H.264 so it plays everywhere
    raw_output = os.path.splitext(output_video_path)[0] + "_raw.mp4"
    out = cv2.VideoWriter(raw_output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    os.makedirs(SWITCH_DIR, exist_ok=True)
    for old in os.listdir(SWITCH_DIR):            # clear snapshots from an earlier run
        if old.endswith(".jpg"):
            os.remove(os.path.join(SWITCH_DIR, old))

    print(f"[INFO] Tracking: {input_video_path} -> {output_video_path}")
    print(f"[INFO] {width}x{height} at {fps:.2f} FPS, {total_frames} frames, tracker = {TRACKER}")

    # Tracking state
    seen_ids = set()                                # Week 4 slide 9: count = len(seen_ids)
    first_seen = {}                                 # id -> frame number
    last_seen = {}                                  # id -> frame number
    last_box = {}                                   # id -> (x1, y1, x2, y2) when last seen
    frames_seen = defaultdict(int)                  # id -> number of frames it appeared in
    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    switch_candidates = []
    replaced_ids = set()                            # lost IDs already linked to a new ID
    lookback_frames = int(SWITCH_LOOKBACK_SECONDS * fps)

    log_file = open(LOG_CSV, "w", newline="")
    log = csv.writer(log_file)
    log.writerow(["frame", "time_s", "track_id", "confidence", "x1", "y1", "x2", "y2"])

    frame_no = 0                                     # frames are numbered from 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        time_s = (frame_no - 1) / fps

        # 4. Track people. persist=True keeps IDs between frames (slide 8)
        results = model.track(frame, persist=True, classes=[PERSON_CLASS_ID],
                              tracker=TRACKER, verbose=False)
        boxes = results[0].boxes

        ids_now = []
        new_ids = []
        if boxes.id is not None:                     # None when nobody is being tracked
            for box in boxes:
                track_id = int(box.id[0])            # slide 8: read the ID with box.id
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if track_id not in seen_ids:
                    seen_ids.add(track_id)
                    first_seen[track_id] = frame_no
                    new_ids.append(track_id)
                ids_now.append(track_id)
                frames_seen[track_id] += 1
                last_seen[track_id] = frame_no
                last_box[track_id] = (x1, y1, x2, y2)
                trails[track_id].append(((x1 + x2) // 2, y2))   # feet position

                log.writerow([frame_no, f"{time_s:.2f}", track_id, f"{conf:.3f}", x1, y1, x2, y2])

                # 5. Draw box and "ID <n> <conf>" label (same style as Week 2)
                color = color_for_id(track_id)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, line)
                draw_label(frame, f"ID {track_id} {conf:.2f}", x1, y1, color, s)

        # Path trails (slide 1: follow a person's path over time)
        if DRAW_TRAILS:
            for track_id in ids_now:
                pts = list(trails[track_id])
                for a, b in zip(pts, pts[1:]):
                    cv2.line(frame, a, b, color_for_id(track_id), line)

        # 6. Look for ID switches: a new ID close to an ID that was lost a moment ago
        for new_id in new_ids:
            if first_seen[new_id] == 1:
                continue                             # everyone in frame 1 is new, not a switch
            nx1, ny1, nx2, ny2 = last_box[new_id]
            ncx, ncy, nh = (nx1 + nx2) / 2, (ny1 + ny2) / 2, ny2 - ny1
            best = None
            for old_id in seen_ids:
                if old_id == new_id or old_id in ids_now or old_id in replaced_ids:
                    continue
                gap = frame_no - last_seen[old_id]
                if gap > lookback_frames:
                    continue
                ox1, oy1, ox2, oy2 = last_box[old_id]
                ocx, ocy, oh = (ox1 + ox2) / 2, (oy1 + oy2) / 2, oy2 - oy1
                size_ratio = max(nh, oh, 1) / max(min(nh, oh), 1)
                if size_ratio > SWITCH_MAX_SIZE_RATIO:
                    continue                         # very different size: a different person
                dist = ((ncx - ocx) ** 2 + (ncy - ocy) ** 2) ** 0.5 / max((nh + oh) / 2, 1)
                if dist <= SWITCH_MAX_DISTANCE and (best is None or dist < best[1]):
                    best = (old_id, dist, gap)
            if best:
                old_id, dist, gap = best
                replaced_ids.add(old_id)             # one lost ID can only be replaced once
                ox1, oy1, ox2, oy2 = last_box[old_id]
                snap = frame.copy()
                cv2.rectangle(snap, (ox1, oy1), (ox2, oy2), (255, 255, 255), line)
                draw_label(snap, f"old ID {old_id} (last seen frame {last_seen[old_id]})",
                           ox1, oy2 + int(22 * s), (255, 255, 255), s)
                name = f"frame{frame_no:05d}_ID{old_id}_to_ID{new_id}.jpg"
                cv2.imwrite(os.path.join(SWITCH_DIR, name), snap)
                switch_candidates.append({
                    "frame": frame_no, "time_s": time_s, "old_id": old_id, "new_id": new_id,
                    "old_last_frame": last_seen[old_id], "gap_frames": gap, "snapshot": name,
                })

        # 7. Info panel: frame number (to note the ID switch), people now, unique count
        panel = f"Frame {frame_no}  |  {time_s:5.1f}s  |  People now: {len(ids_now)}  |  Unique IDs: {len(seen_ids)}"
        pscale = 0.7 * s
        if width < 900 * s:                          # portrait video: shorter panel so it fits
            panel = f"Frame {frame_no} | {time_s:.1f}s | Now: {len(ids_now)} | Unique: {len(seen_ids)}"
        (pw, ph), _ = cv2.getTextSize(panel, cv2.FONT_HERSHEY_SIMPLEX, pscale, line)
        pad = int(10 * s)
        cv2.rectangle(frame, (0, 0), (pw + 2 * pad, ph + 2 * pad), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, panel, (pad, ph + pad), cv2.FONT_HERSHEY_SIMPLEX, pscale, (255, 255, 255), line)

        out.write(frame)

        if SHOW_WINDOW:
            cv2.imshow("Week 4 tracking (q to stop)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_no % 100 == 0:
            pct = f" ({100 * frame_no / total_frames:.0f}%)" if total_frames > 0 else ""
            print(f"[INFO] frame {frame_no}{pct}  unique IDs so far: {len(seen_ids)}")

    # Clean up
    cap.release()
    out.release()
    log_file.close()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()

    # Re-encode the output to H.264 so it plays in any player
    ffmpeg = find_ffmpeg()
    if ffmpeg and to_h264(ffmpeg, raw_output, output_video_path):
        os.remove(raw_output)
    else:
        shutil.move(raw_output, output_video_path)
        print("[WARN] Saved with the mp4v codec. If it will not play, open it in VLC "
              "or run: pip install imageio-ffmpeg")

    # Per-ID summary
    short_limit = int(SHORT_TRACK_SECONDS * fps)
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["track_id", "first_frame", "last_frame", "first_time_s", "last_time_s", "frames_seen"])
        for tid in sorted(seen_ids):
            w.writerow([tid, first_seen[tid], last_seen[tid],
                        f"{(first_seen[tid] - 1) / fps:.2f}", f"{(last_seen[tid] - 1) / fps:.2f}",
                        frames_seen[tid]])
    short_ids = sorted(t for t in seen_ids if frames_seen[t] < short_limit)

    # ==========================================
    # RESULTS SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("TRACKING RESULTS")
    print("=" * 60)
    print(f"Frames processed          : {frame_no}")
    print(f"Unique people (tracked)   : {len(seen_ids)}      <- len(seen_ids)")
    print(f"IDs seen < {SHORT_TRACK_SECONDS:.0f}s ({short_limit} frames): {len(short_ids)}  {short_ids if short_ids else ''}")
    print(f"IDs seen >= {SHORT_TRACK_SECONDS:.0f}s           : {len(seen_ids) - len(short_ids)}")
    print(f"\nLikely ID switches        : {len(switch_candidates)}")
    for c in switch_candidates:
        print(f"  - Frame {c['frame']} ({c['time_s']:.1f}s): ID {c['old_id']} lost at frame "
              f"{c['old_last_frame']}, new ID {c['new_id']} started {c['gap_frames']} frames later "
              f"in the same place -> {SWITCH_DIR}/{c['snapshot']}")
    if not switch_candidates:
        print("  None found automatically. Watch the video for a person whose ID number changes,")
        print("  or for two people swapping IDs when they cross.")
    print("\nNow count the people in the video by hand and compare with the tracked count.")
    print(f"Outputs: {output_video_path}, {LOG_CSV}, {SUMMARY_CSV}, {SWITCH_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    run_person_tracking()
