# Week 4: Person Tracking with ByteTrack

This adds a tracker to my Week 2 YOLO person detector. Each person gets an ID that stays the same across frames. The script counts unique people with tracking, compares that with a manual count and finds ID switches.

## What it does

1. Runs `yolo11n` on every frame, keeping only people (COCO class 0).
2. Uses `model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml")` to give each person an ID. `persist=True` keeps IDs between frames.
3. Draws each person's box with the label `ID <n> <confidence>`, using one colour per ID, and a trail of their path.
4. Shows the frame number, people in view and unique IDs so far at the top of the video.
5. Counts unique people by storing every ID in a set: `count = len(seen_ids)`.
6. Logs every ID in every frame to a CSV and flags likely ID switches: a new ID that appears where another ID of similar size was lost a moment before.

## How to run

```bash
pip install -r requirements.txt
python track_people.py
```

Put the video in this folder as `input.mp4` first.

## Outputs

| File | What it is |
|---|---|
| `output_tracked.mp4` | Video with IDs, boxes, trails and the counter |
| `track_log.csv` | One row per person per frame: frame, time, ID, confidence, box |
| `id_summary.csv` | One row per ID: first frame, last frame, frames seen |
| `switch_candidates/` | Snapshot of each likely ID switch; the white box is where the old ID was last seen |

## Video

A 9.3-second street clip filmed in portrait (2160×3840, 30 FPS, 279 frames). The tracker sees 7 to 13 people in each frame. Some people are close to the camera and large, and others are far away and small.

## Results

| | Count |
|---|---|
| Manual count (people I counted by watching the video) | **[FILL IN]** |
| Unique IDs from the tracker (`len(seen_ids)`) | **44** |
| Unique IDs that lasted 1 second or more | **19** |
| Difference (tracker − manual) | **[FILL IN]** |

### Why the counts are different

The tracker counted 44 unique IDs, but I counted [FILL IN] people by hand. The count is too high because tracks kept breaking: 25 of the 44 IDs lasted less than 1 second.

- **Large, clearly visible people were tracked well.** IDs 1, 2, 3, 6 and 7 kept the same ID for the whole video (273–279 of 279 frames).
- **Small, far-away people kept getting new IDs.** Their boxes are only about 130–340 px tall in a 3840 px frame, and their confidence is low (often 0.2–0.35). The detector misses them for a few frames, and when they are detected again they get a new ID. For example, IDs 49, 69 and 202 appear in the same spot in the background.
- **A person at the left edge of the frame got a chain of IDs:** 56 → 72 → 88 → 97 → 109, and 51 → 94 → 154 → 173. All of these boxes are at the far left (x ≈ 50–220 px), and some last only 2–8 frames. [CHECK in the video: this person is partly cut off by the edge of the frame and/or overlaps ID 3.] When only part of a person is visible, their box changes size from frame to frame, so it no longer overlaps the tracker's predicted box enough to keep the same ID.

Counting only IDs that lasted at least 1 second gives 19, which is closer to the manual count. This shows that most of the extra IDs are short broken tracks, not extra people.

## ID switch

- **Frame:** 87 (2.9 s)
- **What happened:** A person standing far away in the background had **ID 49** from frame 53 to frame 71. The detector lost them for 16 frames. At frame 87 they were detected again in the same spot (less than 0.05 box-heights away) but got a new ID, **ID 69**. ID 69 was lost again at frame 92.
- **Why I think it happened:**
  1. **The person is far away and small.** Their box is only about 140–180 px tall in a 3840 px frame, and the detector's confidence is only about 0.24–0.27. Small, far-away people are hard to detect (Week 3), so the detector missed them for 16 frames.
  2. **ByteTrack could not match the old track.** ByteTrack keeps a lost track for 30 frames, so ID 49 still existed at frame 87. But ByteTrack multiplies box overlap by detection confidence when matching. With a confidence of about 0.27, the new box needs about 74% overlap with the tracker's predicted box to match, which is very hard for a tiny box. So ByteTrack started a new ID instead.
  3. **ByteTrack does not compare appearance.** It only uses motion and box overlap. DeepSORT also compares how the person looks, so it might have recognised the same person and kept ID 49.

![ID switch at frame 87](switch_candidates/frame00087_ID49_to_ID69.jpg)

## Notes

- **No confidence threshold is passed to the tracker.** In Week 3 I used 0.4, but ByteTrack's second step matches low-confidence boxes to the tracks left over, which keeps partly hidden people from being lost. Ultralytics passes boxes down to 0.1 to the tracker, and ByteTrack only starts a new ID from a box at 0.25 or above.
- **ID numbers skip** (my IDs go up to 202 even though there are only 44). ByteTrack uses up an ID number for short detections it never confirms. This is why the count uses `len(seen_ids)` and not the highest ID number.
- **ByteTrack vs BoT-SORT:** the Ultralytics default tracker is BoT-SORT. I chose ByteTrack (`tracker="bytetrack.yaml"`) because it is fast and handles partly hidden people well.
- The output is re-encoded to H.264 with `imageio-ffmpeg` so it plays in browsers, VS Code and Windows. If OpenCV cannot read the input video (for example HEVC from a phone), the script converts it to `input_h264.mp4` first.
