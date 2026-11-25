# -*- coding: utf-8 -*-
"""
tools/check_assets.py — проверка шаблонов UI в assets/
=======================================================

Показывает, какие шаблоны есть, какого они размера, и (если BLOCKADE открыта)
проверяет, находятся ли они на текущем экране — с тем же порогом и бинаризацией,
что использует blockade_ai.py. Помогает понять, почему бот «не видит» смерть
или killfeed.

Запуск
------
    python -I tools/check_assets.py            # только наличие файлов
    python -I tools/check_assets.py --live     # + поиск на текущем экране

--live: за 5 секунд переключись на нужный экран игры.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

FILES = [
    ("aponima_feed.png", "killfeed со своим ником (фраги)", True),
    ("respawn_lmb.png", "надпись возрождения (смерть)", False),
    ("team_select.png", "кнопка АВТО (меню команды)", False),
    ("battle_button.png", "кнопка Бой (лобби)", False),
]


def grab_screen():
    try:
        import dxcam
        cam = dxcam.create(output_idx=0, output_color="BGR")
        f = cam.grab()
        try:
            cam.release()
        except Exception:
            pass
        if f is not None:
            return f
    except Exception:
        pass
    import mss
    import numpy as np
    import cv2
    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[1])
    return cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="искать шаблоны на текущем экране")
    ap.add_argument("--thresh", type=float, default=0.75)
    args = ap.parse_args()

    import cv2

    print("[ПРОВЕРКА] assets/:")
    ok_all = True
    shots = {}
    for fname, desc, required in FILES:
        p = os.path.join(ASSETS, fname)
        if os.path.exists(p):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [ОШИБКА] {fname}: файл есть, но не читается как картинка")
                ok_all = False
                continue
            print(f"  [OK]  {fname:<20} {img.shape[1]}×{img.shape[0]}px  — {desc}")
            shots[fname] = img
        else:
            mark = "НЕТ!" if required else "нет"
            print(f"  [{mark:^5}] {fname:<20} — {desc}")
            if required:
                ok_all = False

    if not ok_all:
        print("\n[ПРОВЕРКА] обязательных шаблонов не хватает. Сними их:")
        print("           python -I tools/make_assets.py")

    if not args.live:
        print("\n[ПРОВЕРКА] живой тест: python -I tools/check_assets.py --live")
        return

    if not shots:
        print("[ПРОВЕРКА] нечего искать — нет ни одного шаблона")
        return

    print("\n[ПРОВЕРКА] Через 5 секунд сниму экран — переключись на игру...")
    for i in range(5, 0, -1):
        print(f"  {i}...", end="\r")
        time.sleep(1)
    screen = grab_screen()
    if screen is None:
        print("[ОШИБКА] не удалось снять экран")
        return
    print(" " * 40)

    for fname, img in shots.items():
        h, w = screen.shape[:2]
        th, tw = img.shape[:2]
        if tw > w or th > h:
            print(f"  {fname}: шаблон БОЛЬШЕ экрана ({tw}×{th} > {w}×{h}) — "
                  f"пересними на текущем разрешении")
            continue
        # Как в blockade_ai.py: бинаризация по V-каналу, matchTemplate
        sv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)[:, :, 2]
        tv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, sm = cv2.threshold(sv, 150, 255, cv2.THRESH_BINARY)
        _, tm = cv2.threshold(tv, 150, 255, cv2.THRESH_BINARY)
        res = cv2.matchTemplate(sm, tm, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        found = score >= args.thresh
        print(f"  {fname}: score={score:.2f} порог={args.thresh} → "
              f"{'НАЙДЕН на экране ' + str(loc) if found else 'не найден'}")
        if not found and fname == "aponima_feed.png":
            print("    (killfeed появляется только пару секунд после фрага — "
                  "проверяй сразу после убийства)")


if __name__ == "__main__":
    main()
