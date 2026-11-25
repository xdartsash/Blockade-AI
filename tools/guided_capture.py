"""Propose raw-frame review candidates using the project's current player model."""
import json
import os


class GuidedCapture:
    def __init__(self, project):
        cfg = json.loads((project / "config.json").read_text(encoding="utf-8-sig"))
        self.weights = project / cfg["yolo_path"]
        if not self.weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {self.weights}")

        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        import numpy as np
        import torch
        from ultralytics import YOLO

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        self.size = int(cfg.get("yolo_imgsz", 512))
        self.device = cfg.get("yolo_device", "cpu")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("[YOLO] CUDA is unavailable, using CPU", flush=True)
            self.device = "cpu"
        self.model = YOLO(str(self.weights))
        if self.model.names != {0: "player"}:
            raise ValueError("Guided capture requires a YOLO model with the single class 'player'")
        print(f"[YOLO] {self.weights.name} | {self.device} | {self.size} px", flush=True)

        # Initialise inference before the player switches to the game.
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        try:
            self.propose(frame)
        except torch.cuda.OutOfMemoryError:
            if self.device == "cpu":
                raise
            self.device = "cpu"
            self.model = YOLO(str(self.weights))
            print("[YOLO] CUDA is out of memory, using CPU", flush=True)
            self.propose(frame)

    def propose(self, frame):
        result = self.model.predict(
            frame, imgsz=self.size, conf=0.15, device=self.device, max_det=50, verbose=False
        )[0]
        height, width = frame.shape[:2]
        proposals = []
        for left, top, right, bottom, score, _ in result.boxes.data.cpu().tolist():
            cx = (left + right) / 2
            cy = (top + bottom) / 2
            if cx < width * 0.14 and cy < height * 0.18:
                continue
            proposals.append({
                "class_id": 0,
                "xyxy": [round(left, 2), round(top, 2), round(right, 2), round(bottom, 2)],
                "confidence": round(score, 4),
            })
        return proposals


def capture_reason(proposals, frame_height, periodic):
    if periodic:
        return "periodic"
    if any(box["xyxy"][3] - box["xyxy"][1] <= 0.10 * frame_height for box in proposals):
        return "small"
    if any(box["confidence"] < 0.35 for box in proposals):
        return "uncertain"
    return None
