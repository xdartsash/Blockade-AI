# -*- coding: utf-8 -*-
"""
tools/make_assets.py — помощник для снятия шаблонов UI (assets/)
================================================================

Делает скриншот BLOCKADE, затем по очереди просит выделить мышкой рамкой каждый
нужный элемент интерфейса и сохраняет его в assets/ как PNG. Это те шаблоны, по
которым бот определяет «я убил», «я умер», «это меню команды», «это кнопка Бой».

Запуск
------
    python -I tools/make_assets.py

Через 5 секунд (успевай развернуть нужный экран игры) скрипт снимет экран.
Дальше откроется окно: тяни рамку вокруг элемента — Enter/пробел сохранить,
c — перерисовать рамку, s — пропустить элемент, q — выйти.

Какие элементы и что вырезать — см. assets/README.md.

Если играешь на другом мониторе/в окне — скрипт всё равно снимает весь основной
экран; просто выдели нужный кусок.
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

# (имя файла, человекочитаемое описание, обязательно?)
TARGETS = [
    ("aponima_feed.png",
     "строку killfeed со СВОИМ ником (момент, когда ты кого-то убил)", True),
    ("respawn_lmb.png",
     "надпись «нажмите ЛКМ для возрождения» с экрана смерти", False),
    ("team_select.png",
     "жёлтую кнопку «АВТО» в меню выбора команды", False),
    ("battle_button.png",
     "кнопку «Бой» в лобби (если есть)", False),
]


def grab_screen():
    """Полный кадр основного экрана (BGR). dxcam → фолбэк mss."""
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
    except Exception as e:
        print(f"[ASSETS] dxcam недоступен ({e}) — пробую mss")
    import mss
    import numpy as np
    import cv2
    with mss.mss() as sct:
        raw = sct.grab(sct.monitors[1])
    return cv2.cvtColor(np.asarray(raw), cv2.COLOR_BGRA2BGR)


def crop_interactive(img, title):
    """Возвращает кроп, выбранный рамкой, или None (пропуск/выход).
    Кнопка: 'q' — выход из всего, 's' — пропустить элемент."""
    import cv2
    r = cv2.selectROI(title, img, fromCenter=False, showCrosshair=True)
    x, y, w, h = [int(v) for v in r]
    if w < 3 or h < 3:
        return None
    return img[y:y + h, x:x + w]


def main():
    ap = argparse.ArgumentParser(description="Снятие шаблонов UI в assets/")
    ap.add_argument("--countdown", type=int, default=5)
    ap.add_argument("--only", default=None,
                    help="снять только один файл, например aponima_feed.png")
    args = ap.parse_args()

    import cv2
    os.makedirs(ASSETS, exist_ok=True)

    targets = TARGETS
    if args.only:
        targets = [t for t in TARGETS if t[0] == args.only]
        if not targets:
            print(f"[ASSETS] нет такого элемента: {args.only}")
            print("[ASSETS] доступны: " + ", ".join(t[0] for t in TARGETS))
            sys.exit(1)

    print("[ASSETS] Сейчас сделаю скриншот. Убедись, что на экране виден НУЖНЫЙ "
          "элемент (killfeed/смерть/меню).")
    for i in range(args.countdown, 0, -1):
        print(f"[ASSETS] скриншот через {i}...", end="\r")
        time.sleep(1)
    shot = grab_screen()
    if shot is None:
        print("[ОШИБКА] не удалось снять экран")
        sys.exit(1)
    print(" " * 60)
    print(f"[ASSETS] скриншот снят: {shot.shape[1]}×{shot.shape[0]}px")
    print("[ASSETS] Тяни рамку вокруг элемента → Enter сохранить, c заново, "
          "s пропустить, q выйти.")

    # Подсказка: если разные элементы на разных экранах, один скриншот их не
    # покроет. Поэтому даём возможность переснять между элементами.
    for idx, (fname, desc, required) in enumerate(targets, 1):
        while True:
            print(f"\n[{idx}/{len(targets)}] {fname} — {desc}"
                  f"{' (ОБЯЗАТЕЛЬНО)' if required else ''}")
            print("    Выдели рамкой. Или: n = переснять экран, s = пропустить, "
                  "q = выйти.")
            crop = crop_interactive(shot, f"Выдели: {fname}")
            key = cv2.waitKey(1) & 0xFF

            # selectROI уже вернул кроп или None. Обработаем пересъём/пропуск
            # через отдельный ввод, т.к. selectROI блокирует.
            if crop is None:
                print("    [ASSETS] рамка не выбрана.")
                cmd = input("    n=переснять, s=пропустить, q=выйти, Enter=повторить: ").strip().lower()
                if cmd == "q":
                    cv2.destroyAllWindows()
                    return
                if cmd == "s":
                    if required:
                        print("    [ВНИМАНИЕ] элемент обязательный — без него "
                              "награда за фраги не работает. Пропускаю, но "
                              "вернись позже (python -I tools/make_assets.py "
                              f"--only {fname}).")
                    break
                if cmd == "n":
                    print("    [ASSETS] переснимаю через 5с — покажи нужный экран")
                    time.sleep(5)
                    shot = grab_screen()
                    continue
                continue

            out = os.path.join(ASSETS, fname)
            cv2.imwrite(out, crop)
            print(f"    [OK] сохранено {out} ({crop.shape[1]}×{crop.shape[0]}px)")
            more = input("    Переснять экран для следующего элемента? (y/N): ").strip().lower()
            if more == "y":
                print("    [ASSETS] переснимаю через 5с — покажи следующий экран")
                time.sleep(5)
                shot = grab_screen()
            break

    cv2.destroyAllWindows()
    print("\n[ASSETS] Готово. Проверь: python -I tools/check_assets.py")


if __name__ == "__main__":
    main()
