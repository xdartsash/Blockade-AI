# -*- coding: utf-8 -*-
"""Review ESP box proposals and assign each visible player a class manually.

Usage:
  python -I tools/review_overlay_labels.py overlay_review/<session>

Keys while reviewing a green proposal: 0=enemy, 1=teammate, s=skip,
 u=undo previous, q=quit. After proposals or on a frame without them,
 a=draw a box around a visible player, Enter=save and continue.
Red proposals are intentionally excluded: they indicate occluded players.
"""
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
WINDOW = "Review visible players: 0 enemy | 1 teammate | s skip | u undo | q quit"


def yolo_line(box, class_id, width, height):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def draw_frame(frame, candidates, active=None, note=""):
    image = frame.copy()
    colors = {0: (0, 0, 255), 1: (255, 100, 0), None: (0, 220, 255)}
    for index, item in enumerate(candidates):
        x1, y1, x2, y2 = item["box"]
        class_id = item.get("class_id")
        color = (0, 255, 255) if index == active else colors[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        label = "ENEMY" if class_id == 0 else "TEAMMATE" if class_id == 1 else "CHECK"
        cv2.putText(image, f"{index + 1}: {label}", (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    if note:
        cv2.rectangle(image, (0, 0), (image.shape[1], 30), (20, 20, 20), -1)
        cv2.putText(image, note[:120], (8, 21), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path, help="one overlay_review session folder")
    parser.add_argument("--out", type=Path, default=ROOT / "dataset" / "raw",
                        help="YOLO raw root; output is created as a session subfolder")
    parser.add_argument("--session-name", default=None,
                        help="optional output folder name; default is source session name")
    args = parser.parse_args()

    source = args.session.resolve()
    originals = source / "originals"
    cleaned = source / "experimental_cleaned"
    records = sorted(source.glob("frame_*.json"))
    if not source.is_dir() or not originals.is_dir() or not records:
        parser.error("session must contain frame_*.json and originals/")

    out_dir = args.out.resolve() / (args.session_name or source.name)
    if out_dir.exists() and any(out_dir.iterdir()):
        parser.error(f"output already contains files: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    stop = False
    try:
        for record_path in records:
            data = json.loads(record_path.read_text(encoding="utf-8"))
            proposals = [b for b in data.get("boxes", []) if b.get("color") == "green"]
            stem = record_path.stem
            source_image = originals / f"{stem}.png"
            clean_image = cleaned / f"{stem}.png"
            image_path = clean_image if clean_image.is_file() else source_image
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[SKIP] unreadable image: {image_path}")
                continue
            height, width = frame.shape[:2]
            candidates = [{"box": list(map(int, b["xyxy"])), "class_id": None}
                          for b in proposals]
            index = 0
            while index < len(candidates):
                view = draw_frame(frame, candidates, active=index,
                                  note=f"{stem}  |  0 enemy  1 teammate  s skip  u undo  q quit")
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(0) & 0xFF
                if key in (ord("0"), ord("1")):
                    candidates[index]["class_id"] = int(chr(key))
                    index += 1
                elif key == ord("s"):
                    index += 1
                elif key == ord("u") and index > 0:
                    index -= 1
                    candidates[index]["class_id"] = None
                elif key == ord("q"):
                    stop = True
                    break
            if stop:
                break

            while True:
                view = draw_frame(frame, candidates,
                                  note=f"{stem}  |  a add missed visible player  Enter save  q quit")
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("a"):
                    roi = cv2.selectROI(WINDOW, frame, showCrosshair=True, fromCenter=False)
                    x, y, w, h = map(int, roi)
                    if w > 0 and h > 0:
                        preview = frame.copy()
                        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 220, 255), 2)
                        cv2.imshow(WINDOW, preview)
                        while True:
                            cls_key = cv2.waitKey(0) & 0xFF
                            if cls_key in (ord("0"), ord("1")):
                                candidates.append({"box": [x, y, x + w, y + h],
                                                  "class_id": int(chr(cls_key))})
                                break
                            if cls_key == ord("s"):
                                break
                elif key in (10, 13):
                    labels = [yolo_line(c["box"], c["class_id"], width, height)
                              for c in candidates if c["class_id"] in (0, 1)]
                    if labels:
                        target_image = out_dir / f"{stem}.png"
                        target_label = out_dir / f"{stem}.txt"
                        if target_image.exists() or target_label.exists():
                            raise FileExistsError(f"refusing to overwrite {stem} in {out_dir}")
                        if not cv2.imwrite(str(target_image), frame):
                            raise OSError(f"could not write {target_image}")
                        target_label.write_text("\n".join(labels) + "\n", encoding="utf-8")
                        saved += 1
                        print(f"[SAVED] {target_image.name}: {len(labels)} labels")
                    else:
                        print(f"[SKIP] {stem}: no labeled visible players")
                    break
                elif key == ord("q"):
                    stop = True
                    break
            if stop:
                break
    finally:
        cv2.destroyAllWindows()
    print(f"[DONE] labeled frames={saved}; output={out_dir}")
    print("Review every visible player; green means visible, not team. Red/occluded proposals were excluded.")


if __name__ == "__main__":
    main()


