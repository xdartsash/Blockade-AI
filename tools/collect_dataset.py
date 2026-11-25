# -*- coding: utf-8 -*-
"""
tools/collect_dataset.py — сбор кадров из BLOCKADE для датасета YOLO
=====================================================================

Что делает
----------
Захватывает экран (dxcam, фолбэк mss) и сохраняет кадры в dataset/raw/.
Два режима:
  1) Ручной:   жмёшь F8 — сохраняется один кадр (удобно для редких сцен).
  2) Авто:     каждые N секунд сам сохраняет кадр (для фонового накопления).
F2 — стоп, F9 — пауза/продолжить авто-режим.

Как пользоваться
----------------
1) Разверни BLOCKADE на весь экран, зайди на карту (лучше ту, где будешь
   обучать; собери кадры со ВСЕХ карт/режимов, где хочешь играть).
2) Запусти:
       python -I tools/collect_dataset.py --every 1.5 --max 400
3) Играй как обычно: ходи, встречай врагов, лови разные дистанции и ракурсы.
   Особенно важны кадры, где враг МЕЛКИЙ (вдали) и где он ПЕРЕКРЫТ наполовину.
4) F2 — стоп. Кадры лежат в dataset/raw/.

Дальше — разметка (см. README.md, раздел «Датасет YOLO»):
   X-AnyLabeling / labelImg → YOLO-формат, классы 0=enemy 1=teammate.

ВАЖНО для качества датасета
---------------------------
• 300-800 кадров с 1000+ боксов достаточно для старта (у прежней модели было
  ~1226 боксов — она уже работала).
• НЕ сохраняй 100 почти одинаковых кадров подряд — разнообразие важнее объёма.
  --every 1.5 даёт хороший разброс; между кадрами двигай камерой.
• Собирай и свои, и чужие: класс teammate нужен, чтобы бот не стрелял по своим.
• Мёртвые тела/лежащие модели НЕ размечай как enemy (иначе бот будет стрелять
  по трупам).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "dataset", "raw")

VK_F2 = 0x71   # стоп
VK_F8 = 0x77   # сохранить кадр
VK_F9 = 0x78   # пауза авто


def key_pressed(vk: int) -> bool:
    import win32api
    return bool(win32api.GetAsyncKeyState(vk) & 0x8000)


class Grabber:
    """dxcam (быстро, через Desktop Duplication), фолбэк mss."""

    def __init__(self):
        self._cam = None
        self._mss = None
        try:
            import dxcam
            self._cam = dxcam.create(output_idx=0, output_color="BGR")
            print("[ЗАХВАТ] dxcam")
        except Exception as e:
            print(f"[ЗАХВАТ] dxcam недоступен ({e}) — использую mss")
            import mss
            self._mss = mss.mss()

    def grab(self):
        if self._cam is not None:
            f = self._cam.grab()
            if f is None:
                return None
            return f  # уже BGR
        mon = self._mss.monitors[1]
        raw = self._mss.grab(mon)
        import numpy as np
        import cv2
        f = np.asarray(raw)          # BGRA
        return cv2.cvtColor(f, cv2.COLOR_BGRA2BGR)

    def close(self):
        if self._cam is not None:
            try:
                self._cam.release()
            except Exception:
                pass
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Сбор кадров для датасета YOLO")
    ap.add_argument("--every", type=float, default=1.5,
                    help="секунд между авто-кадрами (0 = только ручные F8)")
    ap.add_argument("--max", type=int, default=400,
                    help="максимум кадров, после — авто-стоп")
    ap.add_argument("--countdown", type=int, default=5,
                    help="секунд до старта (успеть развернуть игру)")
    ap.add_argument("--out", default=OUT_DIR, help="папка для кадров")
    ap.add_argument("--skip-hud", action="store_true",
                    help="обрезать нижние 12%% и верхние 6%% кадра (HUD/killfeed), "
                         "чтобы YOLO не училась на элементах интерфейса")
    args = ap.parse_args()

    import cv2

    if args.out == OUT_DIR:
        args.out = os.path.join(args.out, time.strftime("session_%Y%m%d_%H%M%S"))
    os.makedirs(args.out, exist_ok=True)

    # Продолжить нумерацию, если кадры уже есть
    existing = [f for f in os.listdir(args.out) if f.startswith("cap_") and f.endswith(".jpg")]
    n = max((int(f[4:-4]) for f in existing if f[4:-4].isdigit()), default=0)
    print(f"[СБОР] папка {args.out}, уже есть кадров: {n}")

    for i in range(args.countdown, 0, -1):
        print(f"[СБОР] старт через {i}... разверни игру", end="\r")
        time.sleep(1)
    print(" " * 60, end="\r")

    grabber = Grabber()
    print(f"[СБОР] F8 = кадр | F9 = пауза авто | F2 = стоп | авто каждые {args.every}с, максимум {args.max}")

    last_auto = 0.0
    paused = False
    saved_now = 0
    try:
        while True:
            if key_pressed(VK_F2):
                print("\n[СБОР] F2 — стоп.")
                break
            if key_pressed(VK_F9):
                paused = not paused
                print(f"[СБОР] авто-режим: {'ПАУЗА' if paused else 'ПРОДОЛЖЕН'}")
                time.sleep(0.3)

            do_save = False
            if key_pressed(VK_F8):
                do_save = True
                time.sleep(0.25)  # дебаунс
            elif (args.every > 0 and not paused
                  and time.monotonic() - last_auto >= args.every):
                do_save = True

            if do_save:
                frame = grabber.grab()
                last_auto = time.monotonic()
                if frame is None:
                    time.sleep(0.05)
                    continue
                if args.skip_hud:
                    h, w = frame.shape[:2]
                    frame = frame[int(h * 0.06):int(h * 0.88), :]
                n += 1
                saved_now += 1
                name = f"cap_{n:05d}.jpg"
                cv2.imwrite(os.path.join(args.out, name), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                print(f"[СБОР] сохранён {name} (всего за сессию: {saved_now})")
                if args.max and saved_now >= args.max:
                    print("[СБОР] достигнут --max, стоп.")
                    break
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[СБОР] Ctrl+C — стоп.")
    finally:
        grabber.close()
        print(f"[СБОР] готово. Кадры: {args.out}")
        print("[СБОР] следующий шаг — разметка (см. README.md, раздел «Датасет YOLO»).")


if __name__ == "__main__":
    main()
