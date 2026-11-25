# -*- coding: utf-8 -*-
"""
tools/smoke_test.py — сквозная проверка стека БЕЗ запуска игры
===============================================================

Проверяет, что всё восстановленное/установленное реально работает:
  1. импорт blockade_ai.Config и detectors.HudDetectors;
  2. HUD-детекторы не падают на синтетических кадрах и корректно реагируют
     на «живой HUD» / «смерть» / «+N под прицелом»;
  3. YOLO делает предсказание на GPU/CPU (скачает yolo26n.pt, ~6 МБ, один раз);
  4. dxcam способен снять экран;
  5. pydirectinput доступен (без реальных нажатий).

Запуск:
    python -I tools/smoke_test.py

Ничего в игре не нажимает и экран не захватывает надолго — безопасно гонять
в любой момент.
"""
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL, WARN = "[OK]  ", "[НЕТ!]", "[ВНИМ.]"
fails = []
warns = []


def check(ok, label, detail=""):
    if ok is None:
        print(f"{WARN} {label}" + (f" — {detail}" if detail else ""))
        warns.append(label)
    elif ok:
        print(f"{PASS} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"{FAIL} {label}" + (f" — {detail}" if detail else ""))
        fails.append(label)


def main():
    import numpy as np
    import cv2

    print("=" * 68)
    print("  SMOKE TEST — Blockade AI (игра не запускается)")
    print("=" * 68)

    # ── 1. Импорт Config и detectors ────────────────────────────────────────
    try:
        from blockade_ai import Config
        check(True, "blockade_ai.Config импортируется")
    except Exception as e:
        check(False, "blockade_ai.Config импортируется", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    try:
        from detectors import HudDetectors, KillDetector, PlusDetector, DeathDetector
        check(True, "detectors.HudDetectors импортируется")
    except Exception as e:
        check(False, "detectors.HudDetectors импортируется", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    cfg = Config()
    cfg.kill_debug = False
    cfg.plus_debug = False
    cfg.death_debug = False
    cfg.kill_debug_save = False   # не мусорим debug_kills/ во время теста

    hud = HudDetectors(cfg)
    check(True, "HudDetectors создан", f"kill={'шаблон' if hud.kill.tpl_mask is not None else 'нет шаблона'}, "
                                       f"plus={'on' if hud._plus_enabled else 'off'}")
    if hud.kill.tpl_mask is None:
        check(None, "Шаблон killfeed", "assets/aponima_feed.png отсутствует — "
                                       "награда за фраги не будет работать. "
                                       "Сними: python -I tools/make_assets.py")

    # ── 2. Детектор смерти на синтетике ─────────────────────────────────────
    # Кадр 1280x720. «Живой HUD»: белое число HP в death_hud_right_roi/core_roi.
    # «Смерть»: число убрано + панель респавна залита светлым.
    W, H = 1280, 720
    alive = np.full((H, W, 3), 40, np.uint8)     # тёмный игровой фон
    rx = cfg.death_hud_right_roi                 # [0.50, 0.925, 0.60, 1.0]
    cx = cfg.death_hud_core_roi
    dr = cfg.death_roi

    def paint_hud(img):
        # правая часть числа HP: много белого (~23%+ площади ROI)
        x0, y0, x1, y1 = [int(rx[0] * W), int(rx[1] * H), int(rx[2] * W), int(rx[3] * H)]
        sub = img[y0:y1, x0:x1]
        sub[:] = 255
        # центр числа HP
        x0, y0, x1, y1 = [int(cx[0] * W), int(cx[1] * H), int(cx[2] * W), int(cx[3] * H)]
        img[y0:y1, x0:x1] = 255

    def paint_death_panel(img):
        x0, y0, x1, y1 = [int(dr[0] * W), int(dr[1] * H), int(dr[2] * W), int(dr[3] * H)]
        img[y0:y1, x0:x1] = 200   # светлая панель респавна

    alive_frame = alive.copy()
    paint_hud(alive_frame)

    dead_frame = alive.copy()
    paint_death_panel(dead_frame)   # HUD числа НЕТ, панель залита

    d = hud.death
    d.hard_reset()
    r, c, f = d.signals(alive_frame)
    check(r >= cfg.death_hud_right_max, "DeathDetector.signals: живой HUD виден",
          f"right={r:.1f}% core={c:.1f}% fill={f:.2f}")

    # живой кадр не должен давать смерть
    t = time.monotonic()
    alive_dead = any(d.on_death_screen(alive_frame, t + i * 0.05, 1000 + i) for i in range(5))
    check(not alive_dead, "DeathDetector: живой кадр → смерть не детектируется")

    # мёртвый кадр должен дать смерть после hold-кадров
    d.hard_reset()
    got = False
    for i in range(6):
        if d.on_death_screen(dead_frame, t + i * 0.05, 2000 + i):
            got = True
            break
    check(got, "DeathDetector: мёртвый кадр → смерть детектируется")

    # уровневый сигнал: пока мёртв, остаётся True
    still = d.on_death_screen(dead_frame, t + 0.5, 2001)
    check(still, "DeathDetector: уровень удерживается, пока мёртв")

    # re-arm: живой HUD несколько кадров → снова жив
    for i in range(cfg.death_min_hud_alive_frames + 2):
        d.on_death_screen(alive_frame, t + 1.0 + i * 0.05, 3000 + i)
    check(not d.on_death_screen(alive_frame, t + 2.0, 3100),
          "DeathDetector: re-arm после живого HUD (нет двойного счёта)")

    # идемпотентность по step_counter: два вызова с одним sc не двигают hold
    d.hard_reset()
    a = d.on_death_screen(dead_frame, t, 5000)
    b = d.on_death_screen(dead_frame, t, 5000)
    check(a == b, "DeathDetector: идемпотентность по step_counter", f"{a} == {b}")

    # ── 3. Плюс-детектор на синтетике ───────────────────────────────────────
    # Рисуем «+1» в plus_roi: якорь «+» на plus_anchor_frac ширины окна.
    p = hud.plus
    p.reset()
    base = np.full((H, W, 3), 30, np.uint8)
    pr = cfg.plus_roi
    ax = int(cfg.plus_anchor_frac * W)
    ay = int((pr[1] + pr[3]) * 0.5 * H)
    COL = (60, 220, 255)   # яркий насыщенный → проходит маску S>=150,V>=220

    def draw_plus(img):
        # глиф «+» ~18x18, попадает в anchor_w/h [9..26]x[7..26]
        cv2.rectangle(img, (ax - 9, ay - 2), (ax + 9, ay + 2), COL, -1)
        cv2.rectangle(img, (ax - 2, ay - 9), (ax + 2, ay + 9), COL, -1)

    def draw_digit1(img, dx):
        # цифра «1»: вертикальная палка 10x20 → w[3..34] h[8..32] area>=22
        cv2.rectangle(img, (dx, ay - 10), (dx + 10, ay + 10), COL, -1)

    def draw_digit2(img, dx):
        # цифра «2»: три горизонтальные + две вертикальные перекладины,
        # форма ОТЛИЧАЕТСЯ от «1» → растровый fingerprint сменится сильно
        cv2.rectangle(img, (dx, ay - 10), (dx + 12, ay - 6), COL, -1)   # верх
        cv2.rectangle(img, (dx + 8, ay - 10), (dx + 12, ay + 2), COL, -1)  # право-верх
        cv2.rectangle(img, (dx, ay - 2), (dx + 12, ay + 2), COL, -1)     # середина
        cv2.rectangle(img, (dx, ay - 2), (dx + 4, ay + 10), COL, -1)     # лево-низ
        cv2.rectangle(img, (dx, ay + 6), (dx + 12, ay + 10), COL, -1)    # низ

    plus = base.copy()
    draw_plus(plus)
    draw_digit1(plus, ax + 20)          # «+1»

    plus2 = base.copy()
    draw_plus(plus2)
    draw_digit2(plus2, ax + 20)          # «+2» — другая форма цифры

    no_plus = p.check(base, t)
    check(not no_plus, "PlusDetector: пустой кадр → фраг не засчитан")

    # Прогон: несколько кадров «+1» (запоминаем baseline), затем «+2» (смена).
    # Шаг по времени = plus_hold_sec, чтобы условие удержания точно прошло.
    frag = False
    seq = [plus] * 5 + [plus2] * 8
    for i, fr in enumerate(seq):
        if p.check(fr, t + i * cfg.plus_hold_sec):
            frag = True
            break
    check(frag, "PlusDetector: смена числа «+N» → фраг засчитан")

    # ── 4. KillDetector: без шаблона должен молча возвращать False ──────────
    hud.kill.reset()
    k = hud.kill.check(alive_frame, True, True)
    if hud.kill.tpl_mask is None:
        check(not k, "KillDetector: без шаблона возвращает False (не падает)")
    else:
        check(isinstance(k, bool), "KillDetector: вернул bool", str(k))

    # ── 5. hud.update() контракт (killed, died) ─────────────────────────────
    try:
        res = hud.update(alive_frame, time.monotonic(), 9999, True, True)
        ok = (isinstance(res, tuple) and len(res) == 2
              and all(isinstance(v, bool) for v in res))
        check(ok, "HudDetectors.update() → (killed, died) оба bool", str(res))
    except Exception as e:
        check(False, "HudDetectors.update()", f"{type(e).__name__}: {e}")

    try:
        hud.reset()
        check(True, "HudDetectors.reset() не падает")
    except Exception as e:
        check(False, "HudDetectors.reset()", f"{type(e).__name__}: {e}")

    # ── 6. YOLO на GPU/CPU ──────────────────────────────────────────────────
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        from ultralytics import YOLO
        m = YOLO("yolo26n.pt")   # скачается один раз (~6 МБ)
        t0 = time.monotonic()
        res = m.predict(alive_frame, verbose=False, conf=0.25, imgsz=320, device=dev)
        dt = time.monotonic() - t0
        nb = 0 if res[0].boxes is None else len(res[0].boxes)
        check(True, f"YOLO предсказание на {dev.upper()}",
              f"{dt * 1000:.0f} мс, боксов: {nb} (на синтетике 0 — норма)")
        if dev == "cuda":
            t0 = time.monotonic()
            for _ in range(5):
                m.predict(alive_frame, verbose=False, conf=0.25, imgsz=320, device=dev)
            dt5 = (time.monotonic() - t0) / 5
            check(None, "Скорость YOLO на GPU",
                  f"~{dt5 * 1000:.0f} мс/кадр. Для target_fps=15 нужно ≤66 мс "
                  f"(и бот ещё ждёт шаг PPO)")
    except Exception as e:
        check(False, "YOLO предсказание", f"{type(e).__name__}: {e}")
        traceback.print_exc()

    # ── 7. Захват экрана (dxcam) ────────────────────────────────────────────
    try:
        import dxcam
        cam = dxcam.create(output_idx=0, output_color="BGR")
        f = cam.grab()
        try:
            cam.release()
        except Exception:
            pass
        check(f is not None, "dxcam: кадр с экрана получен",
              "ок" if f is None else f"{f.shape[1]}×{f.shape[0]}")
    except Exception as e:
        check(None, "dxcam", f"недоступен ({type(e).__name__}) — blockade_ai.py "
                             f"использует только dxcam, без него бот не запустится")

    # ── 8. pydirectinput (без реальных нажатий) ─────────────────────────────
    try:
        import pydirectinput
        check(hasattr(pydirectinput, "keyDown") and hasattr(pydirectinput, "moveRel"),
              "pydirectinput: keyDown/moveRel доступны")
    except Exception as e:
        check(False, "pydirectinput", f"{type(e).__name__}: {e}")

    # ── 9. Веса YOLO и модель PPO ───────────────────────────────────────────
    have_weights = os.path.exists(cfg.yolo_path) or any(
        os.path.exists(p) for p in cfg.yolo_fallbacks)
    check(None if not have_weights else True, "Веса YOLO",
          "есть" if have_weights else
          f"НЕ НАЙДЕНЫ ({cfg.yolo_path}) — blockade_ai.py упадёт с FileNotFoundError. "
          f"Собери датасет и обучи: tools/collect_dataset.py → prepare_dataset.py → train_yolo.py")

    check(None if not os.path.exists(cfg.save_path) else True, "Модель PPO",
          "есть" if os.path.exists(cfg.save_path) else
          f"нет ({cfg.save_path}) — норма для начала: обучи blockade_ai.py --mode train")

    # ── Итог ────────────────────────────────────────────────────────────────
    print("=" * 68)
    if fails:
        print(f"  ИТОГ: {len(fails)} ПРОВАЛ(ОВ): {', '.join(fails)}")
        return 1
    print(f"  ИТОГ: все обязательные проверки пройдены."
          + (f" Предупреждений: {len(warns)}." if warns else ""))
    if warns:
        print("  Что осталось сделать (предупреждения выше):")
        for w in warns:
            print(f"    • {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
