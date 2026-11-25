"""Live preview of the BLOCKADE player detector, or render one screenshot."""

import argparse
import ctypes
import os
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
PROJECT_TOOLS = Path(r"C:\Users\Dartsash\Desktop\Blockade AI\tools")
WINDOW = "YOLO preview"
BOX_COLOR = (95, 237, 82)
FAST_SIZE = 320
DETAIL_SIZE = 512


def load_model(weights):
    work = HERE.parent / "work"
    os.environ["YOLO_CONFIG_DIR"] = str(work / "ultralytics_blockade")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["POLARS_FORCE_PKG"] = "compat"
    sys.path.insert(0, str(work / "blockade_ml_compat"))
    import torch
    from ultralytics import YOLO, settings

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    settings.update({"sync": False})
    model = YOLO(str(weights))
    if model.names != {0: "player"}:
        raise ValueError("Ожидалась модель с одним классом player.")
    return model


def detect(frame, model, args, size):
    return model.predict(
        frame, imgsz=size, conf=args.conf, device=args.device, verbose=False
    )[0]


def draw_boxes(frame, prediction):
    marked = frame.copy()
    height, width = marked.shape[:2]
    scale = max(0.52, min(1.0, width / 1400))
    thickness = 2 if width >= 900 else 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    coords = prediction.boxes.xyxy.cpu().tolist()
    confidence = prediction.boxes.conf.cpu().tolist()
    for (x1, y1, x2, y2), score in zip(coords, confidence):
        left, right = np.clip(np.rint((x1, x2)).astype(int), 0, width - 1)
        top, bottom = np.clip(np.rint((y1, y2)).astype(int), 0, height - 1)
        if right <= left or bottom <= top:
            continue
        cv2.rectangle(marked, (left, top), (right, bottom), (14, 16, 14), thickness + 2)
        cv2.rectangle(marked, (left, top), (right, bottom), BOX_COLOR, thickness)
        label = f"player {score:.0%}"
        (label_width, label_height), baseline = cv2.getTextSize(label, font, scale, thickness)
        x = max(0, min(left, width - label_width - 8))
        y = top - 4 if top >= label_height + 7 else top + label_height + 4
        cv2.rectangle(
            marked, (x, y - label_height - 3), (x + label_width + 8, y + baseline),
            (14, 16, 14), cv2.FILLED,
        )
        cv2.putText(marked, label, (x + 4, y), font, scale, BOX_COLOR, thickness, cv2.LINE_AA)
    return marked


def preview(frame, status, shortcut="F9: size  |  F10: close"):
    """Fit the image without stretching. The header sits outside the game frame."""
    width, height, header = 584, 438, 37
    display = np.full((height + header, width, 3), (24, 23, 22), np.uint8)
    if frame is not None:
        src_h, src_w = frame.shape[:2]
        factor = min(width / src_w, height / src_h)
        w, h = max(1, round(src_w * factor)), max(1, round(src_h * factor))
        resized = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        x, y = (width - w) // 2, header + (height - h) // 2
        display[y:y + h, x:x + w] = resized
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(display, status, (12, 24), font, 0.53, (244, 244, 244), 1, cv2.LINE_AA)
    shortcut_width = cv2.getTextSize(shortcut, font, 0.49, 1)[0][0]
    cv2.putText(display, shortcut, (width - shortcut_width - 12, 23), font, 0.49, (174, 184, 194), 1, cv2.LINE_AA)
    return display


def preview_closed():
    key = cv2.waitKey(12) & 0xFF
    if key in (27, ord("q"), ord("Q")):
        return True
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def run_image(args, model):
    frame = cv2.imdecode(np.frombuffer(args.image.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Не удалось прочитать изображение: {args.image}")
    result = detect(frame, model, args, args.imgsz or DETAIL_SIZE)
    marked = draw_boxes(frame, result)
    print(f"На снимке рамок: {len(result.boxes)}", flush=True)
    if args.save:
        ok, encoded = cv2.imencode(args.save.suffix.lower(), marked)
        if not ok:
            raise ValueError(f"Не удалось сохранить изображение: {args.save}")
        args.save.write_bytes(encoded.tobytes())
        print(f"Готово: {args.save}", flush=True)
        return
    cv2.imshow(WINDOW, preview(marked, f"Screenshot | players: {len(result.boxes)}", "Q / Esc: close"))
    try:
        while not preview_closed():
            pass
    finally:
        cv2.destroyAllWindows()


def run_live(args, model):
    import win32api
    import win32gui

    sys.path.insert(0, str(PROJECT_TOOLS))
    from collect_overlay import ScreenCapture, client_region

    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
    except (AttributeError, OSError):
        pass

    capture = ScreenCapture("auto")
    game_title = args.window.strip().casefold()
    size = args.imgsz or FAST_SIZE
    f9_was_held = False
    previous_frame_time = fps = None
    frame_with_boxes = None
    status = f"{size} px | switch to {args.window}"
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(WINDOW, preview(None, status))
    try:
        while not preview_closed():
            if win32api.GetAsyncKeyState(0x79) & 0x8001:
                break
            hwnd = win32gui.GetForegroundWindow()
            title = win32gui.GetWindowText(hwnd).strip().casefold() if hwnd else ""
            f9_state = win32api.GetAsyncKeyState(0x78)
            if f9_state & 0x8001 and not f9_was_held and title in (game_title, WINDOW.casefold()):
                size = DETAIL_SIZE if size == FAST_SIZE else FAST_SIZE
                previous_frame_time = fps = None
                print(f"Размер входа YOLO: {size} px", flush=True)
            f9_was_held = bool(f9_state & 0x8000)
            if title != game_title:
                previous_frame_time = fps = None
                waiting = f"{size} px | switch to {args.window}"
                if waiting != status:
                    status = waiting
                    cv2.imshow(WINDOW, preview(frame_with_boxes, status))
                continue
            try:
                region = client_region(hwnd, win32gui)
            except win32gui.error:
                continue
            if region is None:
                continue
            try:
                frame = capture.grab(region)
            except (RuntimeError, OSError, ValueError):
                if capture.cam is None:
                    raise
                capture.close()
                capture = ScreenCapture("mss")
                continue
            if frame is None:
                if frame_with_boxes is None and status != "Waiting for a game frame":
                    status = "Waiting for a game frame"
                    cv2.imshow(WINDOW, preview(None, status))
                continue
            if win32gui.GetForegroundWindow() != hwnd:
                continue
            result = detect(frame, model, args, size)
            frame_with_boxes = draw_boxes(frame, result)
            now = perf_counter()
            if previous_frame_time is not None:
                instant = 1 / (now - previous_frame_time)
                fps = instant if fps is None else 0.75 * fps + 0.25 * instant
            previous_frame_time = now
            speed = f"{fps:.1f} FPS" if fps is not None else f"{result.speed['inference']:.0f} ms"
            status = f"{size} px | {speed} | players: {len(result.boxes)}"
            cv2.imshow(WINDOW, preview(frame_with_boxes, status))
    finally:
        capture.close()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Предпросмотр рамок YOLO для BLOCKADE")
    parser.add_argument("--weights", type=Path, default=HERE / "blockade_player_yolo26n_targeted.pt")
    parser.add_argument("--window", default="Blockade", help="Точное название окна игры")
    parser.add_argument("--device", default="cpu", help="cpu или 0 для NVIDIA")
    parser.add_argument("--imgsz", type=int, help="Размер входа модели. По умолчанию 320 в игре и 512 для снимка.")
    parser.add_argument("--conf", type=float, default=0.30, help="Минимальная уверенность")
    parser.add_argument("--image", type=Path, help="Вместо игры проверить один снимок")
    parser.add_argument("--save", type=Path, help="Сохранить результат для --image")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Не найдены веса: {args.weights}")
    if args.image and not args.image.is_file():
        parser.error(f"Не найден снимок: {args.image}")
    if args.save and not args.image:
        parser.error("--save используется вместе с --image")
    if not 0 < args.conf <= 1 or (args.imgsz is not None and args.imgsz < 32):
        parser.error("Уверенность должна быть в (0, 1], размер — не меньше 32")
    print("Загружаю YOLO…", flush=True)
    cv2.setNumThreads(1)
    model = load_model(args.weights)
    if args.image:
        run_image(args, model)
    else:
        print("Переключись в BLOCKADE. F9 меняет размер 320/512, F10 закрывает проверку.", flush=True)
        run_live(args, model)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
