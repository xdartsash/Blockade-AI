# -*- coding: utf-8 -*-
"""
detectors.py — HUD-детекторы для blockade_ai.py
================================================

ВОССТАНОВЛЕННЫЙ МОДУЛЬ. Исходный detectors.py был утерян при переустановке
Windows; этот файл восстановлен по контракту использования в blockade_ai.py и по
подробным комментариям к каждому `plus_*`, `death_*`, `kill_*`-ключу в Config.

Модуль САМОДОСТАТОЧНЫЙ: blockade_ai.py импортирует его на строке 18 — ДО того,
как определён встроенный класс KillDetector (строка ~1108). Поэтому здесь
нельзя делать `from blockade_ai import ...` (циклический импорт). Все три
детектора определены заново, читают настройки из переданного cfg-объекта
(duck-typing: годится любой объект/датакласс с нужными атрибутами).

Что здесь живёт
---------------
KillDetector   — фраг по шаблону своего ника в killfeed (правый верхний угол).
                 Единственный «внешний» источник фрага; бинаризация по V-каналу
                 HSV, чтобы цвет команды (красный/синий/зелёный/жёлтый) не мешал.
PlusDetector   — фраг по счётчику «+N» под прицелом (hitmarker/попадание).
                 Ищет глиф «+» на фиксированной доле ширины окна и цифры справа,
                 засчитывает смену числа с гистерезисом и кулдауном.
DeathDetector  — смерть: и по заливке панели респавна (death_roi), и по
                 «дырке» в HUD здоровья (исчезла правая/центральная часть числа).
                 signals() отдаёт сырые метрики, on_death_screen() — решение.
HudDetectors   — агрегатор: .kill, .plus, .death + update()/reset().

Все пороги/ROI берутся из cfg через getattr(..., default), поэтому модуль
переживёт и старый, и новый config.json.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Тот же логгер, что и в blockade_ai.py — чтобы строки лога шли в общий поток
# и в agent.log с единым форматом.
log = logging.getLogger("blockade_ai")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  KILL-ДЕТЕКТОР (фраг по нику в killfeed)                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class KillDetector:
    """Ищет появление шаблона своего никнейма (вырезанного из killfeed) в
    правом верхнем углу экрана.

    ВАЖНО про цвет: ник в killfeed красится в цвет текущей команды, поэтому
    обычный grayscale здесь не работает — у разных цветов разная яркость по
    формуле люминансности. Решение — бинаризация по V-каналу HSV
    (V = max(R,G,B)): для любого яркого цвета текста V близок к 255, тёмный фон
    killfeed даёт низкий V. Порог по V превращает буквы ЛЮБОГО яркого цвета в
    белый силуэт на чёрном фоне — геометрия сохраняется, цвет команды не важен.
    Шаблон бинаризуется тем же способом при загрузке.

    Фраг засчитывается строго на переходе «силуэта не было → силуэт появился»,
    с кулдауном, чтобы одна и та же строка не давала награду повторно.
    """

    V_THRESH = 150  # выше — «яркий текст», ниже — «тёмный фон killfeed»

    def _binarize(self, bgr: np.ndarray) -> np.ndarray:
        v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, mask = cv2.threshold(v, self.V_THRESH, 255, cv2.THRESH_BINARY)
        return mask

    def _match_masks(self, bgr: np.ndarray) -> tuple[np.ndarray, ...]:
        if not getattr(self.cfg, "kill_use_team_colors", False):
            return (self._binarize(bgr),)

        # Match each text color separately so saturated scenery in another hue
        # does not obscure the nickname (for example blue text on a red wall).
        hue, saturation, value = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV))
        vivid = (saturation >= 160) & (value >= self.V_THRESH)
        team_colors = (
            (hue <= 12) | (hue >= 168),
            (hue >= 98) & (hue <= 138),
            (hue >= 40) & (hue <= 85),
            (hue >= 20) & (hue < 40),
        )
        return tuple(np.where(vivid & color, 255, 0).astype(np.uint8) for color in team_colors)

    def __init__(self, cfg):
        self.cfg = cfg
        self.tpl_mask: Optional[np.ndarray] = None
        path = getattr(cfg, "kill_template_path", "assets/aponima_feed.png")
        if path and os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.tpl_mask = max(self._match_masks(img), key=cv2.countNonZero)
                log.info(f"[KILL] Шаблон ника загружен: {path} "
                         f"({img.shape[1]}×{img.shape[0]})")
        if self.tpl_mask is None:
            log.warning(f"[KILL] Шаблон не найден: {path} — детектор фрагов "
                        f"ОТКЛЮЧЁН (награда за килл начисляться не будет).")

        self._present_prev = False
        self._last_kill_t = -999.0
        self._nkill_saved = 0
        self.last_score = 0.0
        self.last_loc = (0, 0)
        self.last_right_gap = -1

    def check(self, full_frame: np.ndarray,
              recent_shot_ok: bool = True,
              recent_enemy_ok: bool = True) -> bool:
        """Возвращает True строго один раз за реальное killfeed-событие.

        recent_shot_ok/recent_enemy_ok нужны, чтобы не засчитывать твой ник в
        верхней HUD-плашке или в таблице счёта как фраг.
        """
        if self.tpl_mask is None:
            return False

        h, w = full_frame.shape[:2]
        x0, y0, x1, y1 = getattr(self.cfg, "kill_roi", [0.50, 0.07, 1.00, 0.30])
        x0i, y0i = int(w * x0), int(h * y0)
        x1i, y1i = int(w * x1), int(h * y1)
        roi = full_frame[y0i:y1i, x0i:x1i]
        if roi.size == 0:
            return False

        th_, tw_ = self.tpl_mask.shape[:2]
        if roi.shape[0] < th_ or roi.shape[1] < tw_:
            self._present_prev = False
            return False

        matches = []
        for mask in self._match_masks(roi):
            response = cv2.matchTemplate(mask, self.tpl_mask, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(response)
            matches.append((score, loc))
        score, loc = max(matches, key=lambda match: match[0])
        present = score >= float(getattr(self.cfg, "kill_match_thresh", 0.86))

        # Ник появляется в killfeed В ДВУХ РОЛЯХ: слева (ты убил) и справа
        # (убили тебя). Шаблон один, поэтому смерть могла бы засчитаться как
        # фраг. Зазор до правого края ROI их различает.
        roi_w = roi.shape[1]
        right_gap = roi_w - (loc[0] + tw_)
        self.last_score = float(score)
        self.last_loc = (int(loc[0]), int(loc[1]))
        self.last_right_gap = int(right_gap)
        if present and bool(getattr(self.cfg, "kill_side_filter", False)):
            if right_gap <= int(getattr(self.cfg, "kill_victim_margin_px", 30)):
                if getattr(self.cfg, "kill_debug", False):
                    log.info(f"[KILL] отклонено side_filter: зазор_справа={right_gap}px")
                present = False

        now = time.monotonic()
        killed = False
        cooldown = float(getattr(self.cfg, "kill_cooldown_sec", 4.0))

        if present and not self._present_prev and (now - self._last_kill_t) >= cooldown:
            blocked_reason = None
            if bool(getattr(self.cfg, "kill_require_recent_shot", True)) and not recent_shot_ok:
                blocked_reason = "не было недавней стрельбы"
            elif bool(getattr(self.cfg, "kill_require_recent_enemy", True)) and not recent_enemy_ok:
                blocked_reason = "не было недавней цели YOLO"

            if blocked_reason is None:
                killed = True
                self._last_kill_t = now
                log.info(f"[KILL] score={score:.2f} loc={loc} "
                         f"зазор_справа={right_gap}px ROI_w={roi_w}px")
                if bool(getattr(self.cfg, "kill_debug_save", False)) and self._nkill_saved < 60:
                    try:
                        os.makedirs("debug_kills", exist_ok=True)
                        self._nkill_saved += 1
                        _ts = time.strftime("%H%M%S")
                        cv2.imwrite(
                            f"debug_kills/{_ts}_n{self._nkill_saved}"
                            f"_s{int(score * 100)}_gap{right_gap}.png", roi)
                    except Exception as _e:
                        log.warning(f"[KILL] не сохранил кроп: {_e}")
            elif getattr(self.cfg, "kill_debug", False):
                log.info(f"[KILL] Совпадение score={score:.2f} loc={loc}, "
                         f"но отклонено: {blocked_reason}")

        if present and getattr(self.cfg, "kill_debug", False):
            log.debug(f"[KILL] score={score:.2f} loc={loc} "
                      f"recent_shot={recent_shot_ok} recent_enemy={recent_enemy_ok}")

        self._present_prev = present
        return killed

    def reset(self):
        self._present_prev = False
        self._last_kill_t = -999.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  PLUS-ДЕТЕКТОР (фраг по счётчику «+N» под прицелом)                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class PlusDetector:
    """Ловит всплывающий счётчик попаданий/фрагов «+N» под прицелом.

    Это второй, независимый от killfeed источник фрага: в Blockade при
    попадании/убийстве рядом с прицелом коротко горит «+1», «+100» и т.п.
    Алгоритм:
      1. Вырезаем plus_roi, бинаризуем по насыщенным ярким пикселям
         (S >= plus_s_min, V >= plus_v_min); если цветная маска пуста и
         разрешён белый фолбэк — берём белые пиксели (S <= white_s_max,
         V >= white_v_min).
      2. Находим глиф «+» — он стоит на фиксированной доле ширины окна
         (plus_anchor_frac) с допуском plus_anchor_tol_px, и его размеры
         попадают в plus_anchor_w/h_*.
      3. Справа от «+» (в пределах plus_digit_gap_px) ищем цифры — глифы
         подходящего размера, выстроенные в одну строку (plus_line_tol_px).
      4. Считаем «отпечаток» строки цифр (битовая маска). Фраг засчитываем,
         когда отпечаток ОТЛИЧАется от прошлого на >= plus_digit_hamming бит
         (число сменилось) И строка продержалась plus_hold_sec И прошёл
         plus_cooldown_sec с прошлого фрага.
    Дебаунс: plus_off_frames кадров без строки сбрасывают «удержание»,
    plus_change_frames кадров нового отпечатка подтверждают смену.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._prev_fp: Optional[int] = None
        self._cand_fp: Optional[int] = None
        self._cand_since: float = 0.0
        self._cand_frames: int = 0
        self._off_frames: int = 0
        self._last_frag_t: float = -999.0
        self._last_val: Optional[int] = None
        self._n_saved = 0

    # ── вспомогательное ──────────────────────────────────────────────────────
    def _mask(self, roi: np.ndarray) -> np.ndarray:
        C = self.cfg
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        m = ((S >= int(getattr(C, "plus_s_min", 150))) &
             (V >= int(getattr(C, "plus_v_min", 220))))
        if not m.any() and bool(getattr(C, "plus_white_fallback", True)):
            m = ((S <= int(getattr(C, "plus_white_s_max", 60))) &
                 (V >= int(getattr(C, "plus_white_v_min", 230))))
        return (m.astype(np.uint8)) * 255

    @staticmethod
    def _glyphs(mask: np.ndarray) -> List[Tuple[int, int, int, int, int]]:
        """Список (x, y, w, h, area) связанных компонент, кроме фона."""
        n, _lab, st, _cent = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for i in range(1, n):
            x, y, w, h, area = st[i]
            out.append((int(x), int(y), int(w), int(h), int(area)))
        return out

    @staticmethod
    def _fingerprint(mask: np.ndarray, bbox, grid: int = 16) -> int:
        """Растровый отпечаток ФОРМЫ строки «+N».

        Кроп бинарной маски по bbox строки уменьшается до сетки grid×grid и
        превращается в битовую маску: бит установлен, если в ячейке есть
        «чернила». Смена цифры («+1» → «+2», «+5» → «+50», «+1» → «+12») меняет
        форму/ширину строки и даёт hamming >> plus_digit_hamming(=8), а дрожание
        той же строки между кадрами меняет 0-4 бита (ниже порога) — это и есть
        нужный гистерезис.

        (Прежняя реализация «по одному биту на центр глифа» меняла ~1 бит при
        смене цифры и порог 8 не достигался НИКОГДА — детектор не давал фрагов.)
        """
        x0, y0, x1, y1 = bbox
        h, w = mask.shape[:2]
        x0 = max(0, min(int(x0), w - 1)); x1 = max(x0 + 1, min(int(x1), w))
        y0 = max(0, min(int(y0), h - 1)); y1 = max(y0 + 1, min(int(y1), h))
        crop = mask[y0:y1, x0:x1]
        if crop.size == 0:
            return 0
        # INTER_NEAREST сохраняет бинарность «чернила/фон» без размытия.
        small = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_NEAREST)
        cells = small > 127
        fp = 0
        for r in range(grid):
            row = cells[r]
            for c in range(grid):
                if row[c]:
                    fp |= (1 << (r * grid + c))
        return fp

    @staticmethod
    def _hamming(a: int, b: int) -> int:
        return bin(int(a) ^ int(b)).count("1")

    def _read_value(self, digits: List[Tuple[int, int, int, int, int]]) -> Optional[int]:
        """Пытается сложить цифры слева направо в число (по порядку X).
        Это приближение: настоящие глифы не распознаются, но порядок по X
        даёт стабильное «число» для детекта СМЕНЫ."""
        if not digits:
            return None
        ds = sorted(digits, key=lambda g: g[0])
        # Каждый глиф кодируем его шириной/высотой — этого достаточно, чтобы
        # «+5» и «+50» дали разные значения и hamming сработал.
        val = 0
        for i, (x, y, gw, gh, _a) in enumerate(ds):
            val = val * 100 + (gw * 31 + gh * 7 + i)
        return val & 0xFFFFFFFF

    def check(self, full_frame: np.ndarray, now: Optional[float] = None) -> bool:
        """True — один раз за подтверждённую смену числа «+N»."""
        C = self.cfg
        if now is None:
            now = time.monotonic()
        h, w = full_frame.shape[:2]
        x0, y0, x1, y1 = getattr(C, "plus_roi", [0.464, 0.270, 0.559, 0.345])
        xa, ya = int(w * x0), int(h * y0)
        xb, yb = int(w * x1), int(h * y1)
        roi = full_frame[ya:yb, xa:xb]
        if roi.size == 0:
            self._off_frames += 1
            return False
        rh, rw = roi.shape[:2]

        mask = self._mask(roi)
        if int(mask.sum()) == 0:
            self._off_frames += 1
            self._cand_frames = 0
            self._cand_fp = None
            return False

        glyphs = self._glyphs(mask)

        # 1) Якорь «+» — на доле plus_anchor_frac ширины окна (не ROI!).
        anchor_frac = float(getattr(C, "plus_anchor_frac", 0.4805))
        anchor_x = anchor_frac * w - xa  # в координатах ROI
        tol = int(getattr(C, "plus_anchor_tol_px", 9))
        aw_min = int(getattr(C, "plus_anchor_w_min", 9))
        aw_max = int(getattr(C, "plus_anchor_w_max", 26))
        ah_min = int(getattr(C, "plus_anchor_h_min", 7))
        ah_max = int(getattr(C, "plus_anchor_h_max", 26))
        anchor = None
        for g in glyphs:
            x, y, gw, gh, _a = g
            cx = x + gw * 0.5
            if (aw_min <= gw <= aw_max and ah_min <= gh <= ah_max
                    and abs(cx - anchor_x) <= tol):
                anchor = g
                break

        need_off = int(getattr(C, "plus_off_frames", 5))
        if anchor is None:
            self._off_frames += 1
            if self._off_frames >= need_off:
                # строка реально ушла — сбрасываем удержание, но помним число
                self._cand_frames = 0
                self._cand_fp = None
            return False
        self._off_frames = 0

        # 2) Цифры справа от «+» в той же строке.
        ax, ay, aw, ah, _aa = anchor
        acy = ay + ah * 0.5
        gap = int(getattr(C, "plus_digit_gap_px", 40))
        line_tol = int(getattr(C, "plus_line_tol_px", 7))
        dw_min = int(getattr(C, "plus_glyph_w_min", 3))
        dw_max = int(getattr(C, "plus_glyph_w_max", 34))
        dh_min = int(getattr(C, "plus_glyph_h_min", 8))
        dh_max = int(getattr(C, "plus_glyph_h_max", 32))
        d_area = int(getattr(C, "plus_glyph_area_min", 22))
        digits = []
        for g in glyphs:
            x, y, gw, gh, area = g
            if g is anchor:
                continue
            if not (dw_min <= gw <= dw_max and dh_min <= gh <= dh_max):
                continue
            if area < d_area:
                continue
            gcy = y + gh * 0.5
            # справа от якоря, в пределах gap, на одной высоте
            if x >= ax + aw - 2 and (x - (ax + aw)) <= gap and abs(gcy - acy) <= line_tol:
                digits.append(g)

        # Строка «+N» обязана содержать хотя бы одну цифру: одинокий «+» —
        # это часть HUD/прицела, а не счётчик попаданий.
        if not digits:
            self._off_frames += 1
            if self._off_frames >= need_off:
                self._cand_frames = 0
                self._cand_fp = None
            return False

        # Span измеряем ПО ВСЕЙ СТРОКЕ (якорь «+» + цифры), а не только по
        # цифрам: plus_span_min=16 / max=88 задуманы под «+1»…«+100» целиком
        # (одна цифра «1» сама по себе ~10px и не прошла бы нижний порог).
        span_min = int(getattr(C, "plus_span_min", 16))
        span_max = int(getattr(C, "plus_span_max", 88))
        line_items = [anchor] + digits
        lx0 = min(g[0] for g in line_items)
        lx1 = max(g[0] + g[2] for g in line_items)
        span = lx1 - lx0
        if span < span_min or span > span_max:
            # не похоже на счётчик «+N» (слишком узко/широко)
            self._off_frames += 1
            if self._off_frames >= need_off:
                self._cand_frames = 0
                self._cand_fp = None
            return False
        self._off_frames = 0

        # Отпечаток строки (якорь + цифры): кроп маски по bbox строки.
        # Смена числа меняет форму строки → hamming(fp, prev) большой.
        by0 = min(g[1] for g in line_items)
        by1 = max(g[1] + g[3] for g in line_items)
        fp = self._fingerprint(mask, (lx0, by0, lx1, by1))
        val = self._read_value(digits)

        # 3) Гистерезис смены числа.
        hold = float(getattr(C, "plus_hold_sec", 0.35))
        cooldown = float(getattr(C, "plus_cooldown_sec", 0.9))
        ham_thr = int(getattr(C, "plus_digit_hamming", 8))
        change_frames = int(getattr(C, "plus_change_frames", 2))
        # «Та же строка» = hamming ниже порога смены (растровый отпечаток
        # дрожит на 0-4 бита между кадрами из-за сглаживания/шкалы).
        same_thr = max(1, ham_thr // 2)

        frag = False
        if self._prev_fp is None:
            # первое наблюдение строки — запоминаем, фраг не даём
            self._prev_fp = fp
            self._cand_fp = fp
            self._cand_since = now
            self._cand_frames = 1
            self._last_val = val
        else:
            if self._cand_fp is not None and self._hamming(fp, self._cand_fp) < same_thr:
                # та же строка продолжается (дрожание формы в пределах допуска)
                self._cand_frames += 1
            else:
                # кандидат на НОВОЕ число
                self._cand_fp = fp
                self._cand_since = now
                self._cand_frames = 1

            changed = self._hamming(fp, self._prev_fp) >= ham_thr
            held = (now - self._cand_since) >= hold and self._cand_frames >= change_frames
            ready = (now - self._last_frag_t) >= cooldown
            if changed and held and ready:
                frag = True
                self._last_frag_t = now
                self._prev_fp = fp
                self._last_val = val
                if getattr(C, "plus_debug", True):
                    log.info(f"[PLUS] фраг: +{val} fp={fp:#x} hold={now - self._cand_since:.2f}s")
                if bool(getattr(C, "kill_debug_save", False)) and self._n_saved < 60:
                    try:
                        os.makedirs("debug_plus", exist_ok=True)
                        self._n_saved += 1
                        cv2.imwrite(f"debug_plus/p{self._n_saved:02d}_{val}.png", roi)
                    except Exception:
                        pass
            # Если changed, но hold/кулдаун ещё не выдержаны — _prev_fp НЕ
            # трогаем: именно разница с ним подтверждает смену числа на след.
            # кадрах. Сброс происходит только когда строка реально ушла
            # (_off_frames >= plus_off_frames).

        return frag

    def reset(self):
        self._prev_fp = None
        self._cand_fp = None
        self._cand_since = 0.0
        self._cand_frames = 0
        self._off_frames = 0
        self._last_val = None
        # _last_frag_t НЕ сбрасываем: кулдаун переживает смерть/респавн.


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  DEATH-ДЕТЕКТОР (смерть по панели респавна и по «дырке» в HUD здоровья)    ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class DeathDetector:
    """Два независимых признака смерти:

    A) ЗАЛИВКА ПАНЕЛИ. На экране смерти появляется полупрозрачная панель —
       доля пикселей ярче death_gray_min в death_roi прыгает с ~7% (жив)
       до ~60% (смерть). Порог death_fill_min=0.30 даёт запас x8.

    B) «ДЫРКА» В HUD ЗДОРОВЬЯ. У живого в правом нижнем углу горит число HP;
       правая часть числа даёт ~23% белого в death_hud_right_roi, центр —
       >= death_hud_core_min. У мёртвого число исчезает: right < 1.5%,
       core < 3%. death_require_hole требует ОБА условия (именно «дырка»,
       а не блик).

    КОНТРАКТ С blockade_ai.py (важно — от него зависит защита от
    «эпизодов-призраков»):
      • on_death_screen() — УРОВНЕВЫЙ сигнал: возвращает True ПОКА бот мёртв,
        а не один раз в момент смерти. _death_now() в step() вызывает её каждый
        шаг и сам считает death_confirm_steps подтверждений подряд.
      • Переход «жив → мёртв» требует death_hold_frames кадров raw_dead подряд
        (дебаунс одиночных бликов).
      • Обратный переход «мёртв → жив» (re-arm, чтобы следующая смерть
        засчиталась заново) требует death_min_hud_alive_frames кадров ЖИВОГО
        HUD подряд. Ровно это не даёт досчитать одну смерть дважды.
      • between edges выдерживается death_cooldown_sec.
      • Идемпотентность по step_counter: в одном шаге step() on_death_screen
        вызывается ДВАЖДЫ (_death_now и hud.update). Второй вызов с тем же
        step_counter возвращает кэш и НЕ продвигает счётчики — иначе hold
        удваивался бы и смерть подтверждалась в 2 раза быстрее задуманного.
        Вызовы без step_counter (цикл reset/play) всегда продвигают состояние.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._raw_hold = 0        # кадров raw_dead подряд
        self._alive_hold = 0      # кадров живого HUD подряд
        self._dead = False        # текущий УРОВЕНЬ (latched)
        self._last_edge_t = -999.0
        self._last_sc: Optional[int] = None   # для идемпотентности
        self._last_result: bool = False

    # ── сырые метрики ────────────────────────────────────────────────────────
    def _white_mask(self, roi: np.ndarray) -> np.ndarray:
        C = self.cfg
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        m = ((S <= int(getattr(C, "death_hud_white_s", 70))) &
             (V >= int(getattr(C, "death_hud_white_v", 200))))
        return m.astype(np.uint8)

    def _roi(self, full: np.ndarray, frac) -> np.ndarray:
        h, w = full.shape[:2]
        x0, y0, x1, y1 = frac
        return full[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]

    def _pct(self, mask: np.ndarray) -> float:
        if mask.size == 0:
            return 0.0
        return 100.0 * float(np.count_nonzero(mask)) / float(mask.size)

    def signals(self, full: np.ndarray) -> Tuple[float, float, float]:
        """(right_pct, core_pct, fill_frac). НЕ продвигает состояние — чистое
        чтение, безопасно вызывать сколько угодно раз.

        right_pct — % белого в правой части числа HP (жив ~23, смерть <1.5).
        core_pct  — % белого в центре числа HP.
        fill_frac — доля ярких пикселей панели респавна (0..1).
        """
        C = self.cfg
        right_roi = self._roi(full, getattr(C, "death_hud_right_roi", [0.50, 0.925, 0.60, 1.0]))
        core_roi = self._roi(full, getattr(C, "death_hud_core_roi", [0.43, 0.925, 0.50, 1.0]))
        right = self._pct(self._white_mask(right_roi)) if right_roi.size else 0.0
        core = self._pct(self._white_mask(core_roi)) if core_roi.size else 0.0

        death_roi = self._roi(full, getattr(C, "death_roi", [0.597, 0.691, 0.729, 0.833]))
        if death_roi.size:
            gray = cv2.cvtColor(death_roi, cv2.COLOR_BGR2GRAY)
            fill = float(np.count_nonzero(gray >= int(getattr(C, "death_gray_min", 45)))) / float(gray.size)
        else:
            fill = 0.0
        return right, core, fill

    # ── решение ──────────────────────────────────────────────────────────────
    def on_death_screen(self, full: np.ndarray, now: Optional[float] = None,
                        step_counter: Optional[int] = None) -> bool:
        """True ПОКА бот мёртв (уровневый сигнал, см. docstring класса)."""
        # Идемпотентность: повторный вызов в том же шаге — вернуть кэш.
        if step_counter is not None and step_counter == self._last_sc:
            return self._last_result
        if now is None:
            now = time.monotonic()

        result = self._advance(full, now)
        if step_counter is not None:
            self._last_sc = step_counter
            self._last_result = result
        return result

    def _advance(self, full: np.ndarray, now: float) -> bool:
        C = self.cfg
        right, core, fill = self.signals(full)

        fill_dead = fill >= float(getattr(C, "death_fill_min", 0.30))
        right_gone = right < float(getattr(C, "death_hud_right_max", 1.5))
        if bool(getattr(C, "death_require_hole", True)):
            hud_dead = right_gone and (core < float(getattr(C, "death_hud_core_min", 3.0)))
        else:
            hud_dead = right_gone
        raw_dead = bool(fill_dead or hud_dead)
        alive_now = right >= float(getattr(C, "death_hud_right_max", 1.5))

        if raw_dead:
            self._raw_hold += 1
            self._alive_hold = 0
        else:
            self._raw_hold = 0
            self._alive_hold += 1

        hold_need = max(1, int(getattr(C, "death_hold_frames", 2)))
        alive_need = max(1, int(getattr(C, "death_min_hud_alive_frames", 3)))
        cooldown = float(getattr(C, "death_cooldown_sec", 5.0))

        if not self._dead:
            # жив → мёртв: нужен дебаунс raw_dead И прошедший кулдаун
            if self._raw_hold >= hold_need and (now - self._last_edge_t) >= cooldown:
                self._dead = True
                self._last_edge_t = now
                self._raw_hold = 0
                if getattr(C, "death_debug", True):
                    log.info(f"[СМЕРТЬ] right={right:.1f}% core={core:.1f}% "
                             f"fill={fill:.2f} (fill_dead={fill_dead} hud_dead={hud_dead})")
        else:
            # мёртв → жив: re-arm только после alive_need кадров ЖИВОГО HUD.
            # Именно это запрещает досчитать одну смерть дважды.
            if self._alive_hold >= alive_need and alive_now:
                self._dead = False
                self._alive_hold = 0
                if getattr(C, "death_debug", True):
                    log.info(f"[СМЕРТЬ] живой HUD восстановлен (right={right:.1f}%) — re-arm.")
        return self._dead

    def reset(self):
        """Мягкий сброс (новый эпизод): чистим гистерезис и уровень."""
        self._raw_hold = 0
        self._alive_hold = 0
        self._dead = False
        self._last_sc = None
        self._last_result = False

    def hard_reset(self):
        """Полный сброс (после подтверждённого спавна / в грации): то же, что
        reset(), плюс сбрасываем таймер ребра, чтобы не ждать кулдаун заново."""
        self.reset()
        self._last_edge_t = -999.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  АГРЕГАТОР                                                                 ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class HudDetectors:
    """Композит трёх HUD-детекторов. Именно его создаёт Blockade3DEnv:

        self.hud = HudDetectors(C)
        self.kill_detector = self.hud.kill        # ._last_kill_t, .reset()
        self.hud.reset()                          # новый эпизод
        self.hud.death.hard_reset()               # после спавна
        self.hud.death.signals(full)[0]           # «жив ли HUD»
        self.hud.on_death_screen(full, now, sc)   # смерть?
        killed, died = self.hud.update(full, now, sc)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.kill = KillDetector(cfg)
        self.plus = PlusDetector(cfg)
        self.death = DeathDetector(cfg)
        # Счётчик фрагов «+N» можно выключить, поставив plus_hold_sec < 0;
        # по умолчанию он ВКЛЮЧЁН как второй, независимый от killfeed источник.
        self._plus_enabled = float(getattr(cfg, "plus_hold_sec", 0.35)) >= 0.0
        log.info("[HUD] Детекторы инициализированы: kill=%s plus=%s death=ok"
                 % ("ok" if self.kill.tpl_mask is not None else "НЕТ ШАБЛОНА",
                    "on" if self._plus_enabled else "off"))

    def on_death_screen(self, full: np.ndarray, now: Optional[float] = None,
                        step_counter: Optional[int] = None) -> bool:
        return self.death.on_death_screen(full, now, step_counter)

    def update(self, full: np.ndarray, now: Optional[float] = None,
               step_counter: Optional[int] = None,
               recent_shot_ok: bool = True,
               recent_enemy_ok: bool = True) -> Tuple[bool, bool]:
        """Один прогон всех детекторов. Возвращает (killed, died).

        killed — фраг засчитан (killfeed ИЛИ «+N» под прицелом).
        died   — подтверждена смерть на этом кадре.
        Необязательные recent_shot_ok/recent_enemy_ok пробрасываются в
        KillDetector; blockade_ai.py вызывает update() без них, поэтому по
        умолчанию гейт «недавней стрельбы/цели» не сужает детект.
        """
        if now is None:
            now = time.monotonic()

        killed = False
        try:
            killed = self.kill.check(full, recent_shot_ok, recent_enemy_ok)
        except Exception:
            if not getattr(self, "_kill_err", False):
                self._kill_err = True
                log.error("[HUD] Ошибка в KillDetector:")
                traceback.print_exc()

        if not killed and self._plus_enabled:
            try:
                killed = self.plus.check(full, now)
            except Exception:
                if not getattr(self, "_plus_err", False):
                    self._plus_err = True
                    log.error("[HUD] Ошибка в PlusDetector:")
                    traceback.print_exc()

        died = False
        try:
            died = self.death.on_death_screen(full, now, step_counter)
        except Exception:
            if not getattr(self, "_death_err", False):
                self._death_err = True
                log.error("[HUD] Ошибка в DeathDetector:")
                traceback.print_exc()

        return bool(killed), bool(died)

    def reset(self):
        """Новый эпизод/спавн: мягко сбрасываем все три детектора."""
        try:
            self.kill.reset()
        except Exception:
            pass
        try:
            self.plus.reset()
        except Exception:
            pass
        try:
            self.death.reset()
        except Exception:
            pass
