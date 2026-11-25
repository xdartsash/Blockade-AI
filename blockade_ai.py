import os
# Эти переменные должны быть установлены ДО импорта OpenCV и PyTorch.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import argparse
import json
import logging
import queue
import re
import sys
import threading
import time
import traceback
import faulthandler
from dataclasses import asdict, dataclass, field

from detectors import HudDetectors
from recovery import RecoveryPlanner
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pydirectinput
import win32api
import win32con
import win32gui
import dxcam
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, VecFrameStack, VecNormalize,
)
import torch as th
from ultralytics import YOLO
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
#  Crash-лог (записывает трассировку при падении C-кода)
# ──────────────────────────────────────────────────────────────────────────────
_CRASH_LOG = open("crash_report.txt", "a", encoding="utf-8", buffering=1)
faulthandler.enable(file=_CRASH_LOG, all_threads=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Логирование
# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("agent.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("blockade_ai")

# ──────────────────────────────────────────────────────────────────────────────
#  Оптимизации
# ──────────────────────────────────────────────────────────────────────────────
pydirectinput.PAUSE    = 0
pydirectinput.FAILSAFE = False
th.backends.cudnn.benchmark = True
th.set_num_threads(3)  # v13: Athlon X4 840 = 4 ядра
cv2.setNumThreads(1)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  КОНФИГУРАЦИЯ                                                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝
@dataclass
class Config:
    # ── Устройства ──────────────────────────────────────────────────────────
    yolo_device: str = "cuda"       # CPU — безопасно; "cuda" — быстрее
    ppo_device:  str = "cuda"      # GPU для нейросети
    ppo_hybrid: bool = True         # Сбор опыта на CPU, градиенты на GPU
    ppo_play_device: str = "cpu"    # Устройство в режиме игры (градиентов нет)

    # ── Захват экрана ────────────────────────────────────────────────────────
    target_fps: int  = 15          # FPS захвата кадров
    crop_size:  int  = 640         # Вырезка вокруг прицела (px)
    crop_y_shift: int = 0          # Сдвиг crop по Y: 0 = ровно центр прицела
    obs_size:   int  = 84          # Размер кадра для нейросети
    frame_stack: int = 4           # Стек кадров (чувство движения)

    # ── Параметры сглаживания мыши ──────────────────────────────────────────
    mouse_smooth_alpha: float = 0.55  # Меньше = инертнее/плавнее, 1.0 = выкл
    step_min_interval: float  = 0.0   # 0.0 = автоматически подстраивать под 1/target_fps

    # ── YOLO ─────────────────────────────────────────────────────────────────
    yolo_conf:  float = 0.25       # Порог уверенности (снижен, чтобы рамки на дистанции не мигали)
    yolo_half: bool    = False      # FP16. На Pascal (GTX 10xx) ЗАМЕДЛЯЕТ — держать False!
    yolo_every: int    = 2          # Прогонять YOLO раз в N шагов
    kill_check_every: int = 2       # Проверять kill-feed раз в N шагов
    fps_report_every: int = 150     # Лог реального FPS раз в N шагов
    profile_enabled: bool = True    # Разбивка времени шага по секциям в лог
    tpl_match_scale: int = 2        # Во сколько раз уменьшать картинку перед matchTemplate
    yolo_imgsz: int    = 320        # Размер входа YOLO (меньше = быстрее)
    yolo_min_box_w: int = 10
    yolo_min_box_h: int = 24
    yolo_min_box_area: int = 250
    # ── v18: геометрия цели ────────────────────────────────────────────
    # В датасете (1226 боксов) человек всегда вытянут по вертикали:
    # h/w p10=2.05 мед=2.36 p90=2.54. Столбы/двери/лестницы дают 3.5..6.3,
    # вывески и ящики — около 1.0. Фильтр режет и то, и другое.
    target_shape_filter: bool = True
    # v18.1: живые боксы YOLO приземистее разметки: в логе живого
    # сервера медиана 27x47 = h/w 1.74, а 17x25 = 1.47. Нижнюю границу
    # 1.5 надо было брать не из разметки, а из живых детекций.
    target_hw_min: float = 1.25
    target_hw_max: float = 3.6
    # Цель должна прожить два прогона YOLO подряд (1 = выключено).
    # v18.1: ПО УМОЛЧАНИЮ ВЫКЛ. При yolo_every=3 между прогонами ~0.2 с,
    # за это время при повороте мыши бокс 27x47 уезжает на больше
    # своей ширины — IoU становится 0 даже у настоящего игрока.
    target_persist_frames: int = 1
    target_persist_iou: float = 0.2
    # Дополнительный допуск по центрам боксов, пиксели.
    target_persist_tol_px: float = 60.0
    # Каждые N прогонов YOLO печатать воронку фильтров (0 = не печатать).
    yolo_funnel_every: int = 100
    # v18.2: дамп кадров, которые реально ушли в YOLO (0 = выкл).
    # Нужен, чтобы сравнить "что видит бот" с debug_yolo_win.py.
    yolo_dump_every: int = 0
    yolo_dump_max: int = 80
    yolo_dump_dir: str = "yolo_dump"
    # Крупный бокс = враг вплотную. Таких кадров в датасете мало (10%),
    # поэтому conf там падает до 0.2-0.4, а ложных боксов такого размера
    # после переобучения почти нет — для них порог отдельный.
    shoot_big_box_h: int = 110
    shoot_min_conf_big: float = 0.30
    yolo_path:  str    = "runs/detect/train-10/weights/best.pt"
    yolo_fallbacks: list = field(default_factory=lambda: [
        "runs/detect/train-2/weights/best.pt",
        "runs/detect/train/weights/best.pt",
        "best.pt",
    ])
    # Фильтр классов YOLO. Работает корректно только если модель обучена так:
    # 0 = enemy, 1 = teammate. Для старой одноклассовой модели оставь False
    # и используй teammate_nameplate_filter ниже.
    filter_enemy_class: bool = False
    enemy_class_id: int = 0

    # ── Фильтр тиммейтов по нику/nameplate над головой ──────────────────────
    # В Blockade у союзников над головой часто есть ник/уровень. OCR не нужен:
    # ищем маленькую яркую строку над bbox игрока. Если нашли — игнорируем цель.
    teammate_nameplate_filter: bool = True
    nameplate_debug: bool = False
    # patch10: геометрические ворота против ложных ников на текстуре стен.
    # Настоящий ник: 3..10 глифов, строка до ~95px, сидит близко над головой
    # и по горизонтали центрирована относительно самого игрока.
    nameplate_glyph_s_max: int = 70        # было только в getattr, теперь настраивается
    nameplate_glyph_max: int = 10          # больше — это швы кладки, не ник
    nameplate_line_w_max: int = 95         # предел ширины строки ника, px
    nameplate_up_ratio_max: float = 1.5    # выше_головы <= 1.5 * высота бокса
    nameplate_center_tol_ratio: float = 0.8  # |центр строки - центр бокса| <= 0.8 * ширина
    # patch10: выбор слота оружия на спавне (1 блок, 2 бита, 3 винторез, 4 пистолет)
    weapon_select_on_spawn: bool = True
    weapon_slot_key: str = "3"
    nameplate_x_expand: float = 0.65       # насколько расширять ROI по X относительно bbox
    nameplate_y_top: float = 0.70          # верх ROI: y1 - bbox_h * top
    nameplate_y_bottom: float = 0.03       # низ ROI:  y1 - bbox_h * bottom
    nameplate_min_ratio: float = 0.006     # минимум ярких пикселей текста в ROI
    nameplate_max_ratio: float = 0.18      # защита от зелёной карты/яркого фона вместо текста
    nameplate_min_pixels: int = 10         # минимум пикселей текста
    nameplate_min_width_frac: float = 0.35 # строка ника должна быть достаточно широкой
    # Белый/серый текст ника. Обычно это самая надёжная часть nameplate.
    nameplate_white_s_max: int = 95
    nameplate_white_v_min: int = 135
    # Жёлтый/салатовый текст уровня/ника. Можно выключить, если даёт ложняки.
    nameplate_use_yellow: bool = True
    nameplate_yellow_h_min: int = 15
    nameplate_yellow_h_max: int = 45
    nameplate_yellow_s_min: int = 70
    nameplate_yellow_v_min: int = 120
    # Зелёный текст ника. По умолчанию выключен, потому что сама карта зелёная
    # и широкий green-filter может принять фон за ник. Включай только если имена
    # союзников реально зелёные и белый/жёлтый фильтр их не ловит.
    nameplate_use_green: bool = False
    nameplate_green_h_min: int = 35
    nameplate_green_h_max: int = 90
    nameplate_green_s_min: int = 80
    nameplate_green_v_min: int = 140

    # ── Глифовый детектор ника (v7) ─────────────────────────────────────────────
    # Ищет не «яркие пиксели», а строку из буквоподобных клякс на одной высоте.
    nameplate_mode: str = "glyph"        # "glyph" | "ratio" (старое поведение)
    nameplate_up_px: int = 75             # на сколько пикселей вверх смотреть от верха bbox
    nameplate_down_px: int = 2            # отступ от верха bbox вниз
    nameplate_pad_px: int = 30            # расширение ROI по X в пикселях
    nameplate_glyph_min: int = 3          # минимум букв в строке
    nameplate_glyph_w_min: int = 2
    nameplate_glyph_w_max: int = 22
    nameplate_glyph_h_min: int = 4
    nameplate_glyph_h_max: int = 20
    nameplate_glyph_area_min: int = 6
    nameplate_line_tol_px: int = 6        # разброс букв по вертикали
    nameplate_glyph_s_max: int = 70       # замерено на кадрах: S ника = 38..58
    nameplate_glyph_v_min: int = 175      # замерено: ник светлый, B178 G193 R220

    # ══ v13: НОВЫЕ ПАРАМЕТРЫ ══════════════════════════════════════════════
    # Наблюдение
    obs_gray: bool = True              # 1 канал вместо 3 -> CNN втрое дешевле
    # Аим-ассист
    aim_assist_weight: float = 0.85    # 1.0 = ассист решает всё, 0 = только PPO
    aim_target_y_frac: float = 0.35    # куда целиться внутри бокса (0=голова, 0.5=центр)
    # Защита от стрельбы по своим
    shoot_min_conf: float = 0.45       # порог уверенности YOLO для ОГНЯ
    teammate_iou_veto: float = 0.25    # бокс врага перекрыт боксом своего -> это свой
    mate_block_enabled: bool = True    # не стрелять, если под прицелом союзник
    mate_block_pad_px: int = 24
    # Защита от ложных смертей
    death_spawn_grace_sec: float = 3.0  # после спавна HUD ещё не нарисован
    death_confirm_steps: int = 3        # подтверждений подряд
    death_min_hud_alive_frames: int = 3
    # Награда
    r_aimed: float = 0.03               # цель внутри гейта стрельбы
    # Размер сети (меньше = быстрее на слабом CPU)
    ppo_features_dim: int = 128
    ppo_net_arch: int = 64

    # ══ v14: НОВЫЕ ПАРАМЕТРЫ ══════════════════════════════════════════════
    # Как часто реально пересчитывать шаблоны UI (сек). Раньше троттлинг был
    # по номеру шага, а внутри reset() номер шага не меняется — детекторы
    # навсегда залипали на старом значении.
    ui_check_interval: float = 0.12
    # Возрождение: надпись "нажмите ЛКМ" появляется через несколько секунд
    respawn_blind_after: float = 6.0
    # Конец матча / загрузка новой карты
    blind_auto_after: float = 20.0     # экран статичен столько сек -> клик вслепую
    blind_auto_retry: float = 6.0
    static_ratio: float = 0.0015       # ниже этого движения кадр считаем статичным
    capture_restart_after: float = 5.0 # dxcam молчит столько сек -> перезапуск

    # ── v15: мягкая остановка по горячей клавише ────────────────────────
    # Ctrl+C прерывает процесс в случайной точке (иногда прямо во время
    # градиентного апдейта или записи файла). F2 останавливает штатно.
    # Код клавиши Windows: F2=0x71, F3=0x72, F4=0x73, F8=0x77, End=0x23.
    stop_hotkey_vk: int = 0x71
    stop_hotkey_name: str = "F2"

    # ── v15: подтверждение спавна по HUD ────────────────────────────────
    # Выходить из ожидания только когда в HUD видна правая часть числа
    # HP. Иначе бот "играет" на экране смерти (эпизод-призрак).
    alive_hud_confirm: bool = True
    alive_hud_timeout: float = 25.0    # HUD не найден столько сек -> начать всё равно
    # Память о цели
    target_memory_sec: float = 1.5
    # Затухание аим-ассиста: PPO постепенно учится целиться сам
    aim_assist_min: float = 0.35
    aim_assist_decay_steps: int = 400_000
    # Награда
    # v15: было -0.02 — этот штраф давал 64% ВСЕЙ награды и учил бота
    # не стрелять вообще. YOLO видит врага в 0-2% кадров, значит почти
    # любой выстрел формально "в пустоту" — наказывать за это сильно
    # нельзя, иначе политика схлопнется в "никогда не жать огонь".
    r_shoot_no_target: float = -0.005  # огонь, когда цели нет вообще

    # ── PPO гиперпараметры ────────────────────────────────────────────────────
    learning_rate: float = 1e-4
    n_steps:       int   = 256     # Уменьшено — короче пауза на градиентный апдейт
    batch_size:    int   = 128
    n_epochs:      int   = 3       # Уменьшено — real-time мир не убегает вперёд
    gamma:         float = 0.99
    gae_lambda:    float = 0.95
    clip_range:    float = 0.2
    ent_coef:      float = 0.008   # Коэф. исследования
    vf_coef:       float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_500_000

    # ── Лимит длины эпизода (шагов) ────────────────────────────────────────────
    max_episode_steps: int = 20000  # ~3 мин при 10 FPS — короче эпизоды = лучше credit assignment

    # ── Пути сохранений ───────────────────────────────────────────────────────
    save_path:        str = "ppo_models/blockade_final.zip"
    norm_path:        str = "ppo_models/vecnormalize.pkl"
    checkpoint_path:  str = "./ppo_models/"
    tb_log:           str = "./tb_logs/"
    checkpoint_freq:  int = 25_000

    # ── Награды ──────────────────────────────────────────────────────────────
    r_kill:       float =  15.0     # Убийство врага (только по killfeed!)
    r_death:      float = -5.0     # Смерть / зависание
    r_approach:   float =  0.006   # Движение к врагу
    r_proximity:  float =  0.004    # Нахождение рядом с врагом
    r_shoot_near: float =  0.05     # Стрельба вплотную
    r_forward:    float =  0.002   # Награда за РЕАЛЬНОЕ движение вперёд (масштабируется motion ratio)
    r_miss_shot:  float = -0.03    # Штраф за стрельбу в молоко
    r_step:       float = -0.004    # Штраф за каждый шаг
    r_jump:       float = -0.03    # Штраф за каждый прыжок (без него — бесконечный спам прыжков)
    r_enemy_seen: float =  0.01   # Небольшой бонус за то, что враг вообще в кадре

    # ── Автоматическая перезарядка по OCR HUD (скрипт, НЕ часть action space) ──
    # В Blockade 3D разное оружие имеет РАЗНЫЙ размер магазина (единицы —
    # ��есятки — вплоть до ~300 патронов) и разное время перезарядки, поэтому
    # фиксированные числа в конфиге в принципе не могут быть верными для всех
    # стволов сразу. Вместо угадывания констант бот читает РЕАЛЬНУЮ цифру
    # патронов прямо с HUD через OCR (Tesseract) — тогда неважно, какое
    # оружие в руках: пусто по факту = пусто, перезарядка кончилась по факту
    # (цифра снова не ноль) = кончилась. Никаких mag_size/reload_time.
    auto_reload:      bool  = True
    ammo_ocr_enabled: bool  = False  # False = без Tesseract: считаем патроны по выстрелам
    # ROI для цифры патронов — доли ширины/высоты экрана (x0,y0,x1,y1).
    # ПОДСТРОЙ под расположение HUD патронов в Blockade 3D (обычно низ-право).
    ammo_roi: list = field(default_factory=lambda: [0.899, 0.935, 0.986, 1.00])
    ammo_ocr_every:     int   = 5     # раз в сколько шагов гонять OCR (дорогая операция)
    ammo_zero_confirm:  int   = 2     # сколько раз подряд увидеть "0"/пусто, прежде чем поверить
    # Фолбэк, если OCR ничего не распознал (плохой ракурс/перекрыт эффектами):
    # ведём приблизительный счётчик по выстрелам, как раньше, но только пока
    # OCR "молчит" — как только OCR снова считает цифру, он берёт приоритет.
    fallback_mag_size:    int   = 10
    # Темп стрельбы (выстрелов в секунду) для оценки расхода патронов,
    # пока курок удерживается. Раньше считались только нажатия, поэтому
    # магазин пустел в игре, а счётчик этого не видел и R не нажимался.
    weapon_rof:       float = 6.0
    # Перезарядка в простое: цели нет — значит R бесплатный.
    reload_when_idle: bool  = True
    reload_idle_sec:  float = 1.5
    fallback_reload_time: float = 2.5
    # Максимальное разумное число для OCR, чтобы случайный мусор вроде "2111"
    # от иконок патронов не принимался за реальный магазин.
    ammo_max_reasonable: int = 300

    # ── Порог "полного" движения для нормировки r_forward ──────────────────────
    move_full_ratio: float = 0.012  # ratio motion-diff, при котором движение = 100% награды

    # ── Killfeed-детектор (единственный источник фрагов) ──────────────────────
    kill_template_path: str   = "assets/aponima_feed.png"  # шаблон своего ника/строки из killfeed
    kill_use_team_colors: bool = False
    kill_match_thresh:  float = 0.86  # было 0.75; выше = меньше ложных совпадений
    kill_cooldown_sec:  float = 4.0
    # ROI killfeed. ВАЖНО: y0=0.07 отсекает верхнюю HUD-плашку с твоим ником
    # ([21]aponima MIDGARD...), из-за которой раньше были ложные киллы на спавне.
    kill_roi: list = field(default_factory=lambda: [0.50, 0.07, 1.00, 0.30])
    # Защита от ложных киллов: засчитывать killfeed только если бот недавно
    # стрелял и недавно видел цель через YOLO.
    kill_require_recent_shot: bool = True
    kill_recent_shot_sec: float = 2.0
    kill_require_recent_enemy: bool = True
    kill_recent_enemy_sec: float = 2.0
    kill_debug: bool = False
    # --- v6: borba s lozhnymi fragami ---
    kill_debug_save: bool = True
    kill_side_filter: bool = False
    kill_victim_margin_px: int = 30
    kill_death_veto_sec: float = 7.0

    # ── Детекция смерти / респавн ─────────────────────────────────────────────
    respawn_key:         str   = "space"
    respawn_sat_max:     int   = 30
    respawn_val_min:     int   = 180
    respawn_threshold:   float = 0.82
    respawn_seen_needed: int   = 2
    respawn_retry_after: float = 1.5

    # ── Меню выбора команды ───────────────────────────────────────────────────
    team_hue_min:      int   = 20
    team_hue_max:      int   = 40
    team_sat_min:      int   = 100
    team_val_min:      int   = 80
    team_threshold:    float = 0.65
    team_retry_after:  float = 5.0
    auto_x_frac:       float = 0.50
    auto_y_frac:       float = 0.105
    team_auto_offset_y: Optional[int] = None

    # ── Кнопка "Бой" (опционально; см. assets/battle_button.png) ───────────────
    battle_template_path: str   = "assets/battle_button.png"
    battle_threshold:     float = 0.80
    battle_retry_after:   float = 3.0
    battle_x_frac:        float = 0.50   # координаты клика, если шаблон найден
    battle_y_frac:        float = 0.90

    # ── Экран загрузки между картами ────────────────────────────────────────
    loading_black_mean: float = 15.0  # среднее по кадру ниже этого = считаем "чёрный экран загрузки"

    # ── Антизастревание ───────────────────────────────────────────────────────
    stuck_frames_needed: int   = 8
    stuck_ratio:         float = 0.006
    stuck_cooldown:      float = 3.5
    stuck_turn_px:       int   = 260
    stuck_hold_sec:      float = 1.2
    escape_drive_turn_px: int = 0

    # ── Прыжок ───────────────────────────────────────────────────────────────
    jump_cooldown:       float = 5.0   # Мин. интервал между прыжками (сек) —
                                        # увеличен: при почти случайной политике
                                        # на старте обучения (entropy ~ln(N_actions))
                                        # шанс на jump=1 на каждом шаге ~50%, и без
                                        # длинного кулдауна бот прыгает буквально
                                        # непрерывно, пока штраф r_jump не "дойдёт"
                                        # до политики за много тысяч апдейтов.

    # ── Логика ───────────────────────────────────────────────────────────────

    # -- Early training safety -------------------------------------------------
    # Keep jump disabled until the policy learns basic movement and shooting.
    # This prevents random PPO actions from bunny-hopping on spawn.
    jump_enabled: bool = False

    # Random vertical mouse movement often makes the agent stare at the floor.
    # While no enemy is detected, vertical look can be locked to 0.
    vertical_look_when_no_enemy: bool = False

    # Simple YOLO aim assist. PPO still controls movement and shooting, but the
    # camera is pulled toward the detected target to make early learning easier.
    aim_assist: bool = True
    # Вертикалью управляет ТОЛЬКО аим-ассист; PPO к оси питча не допускается
    # (action[3] глушится в step). Питч сам не центрируется, поэтому работает
    # интегратор наклона + авто-возврат к горизонту (см. pitch_* ниже).
    lock_vertical_look: bool = False
    aim_vertical_enabled: bool = True
    aim_gain_x: float = 0.18
    aim_gain_y: float = 0.20
    aim_deadzone: int = 8
    aim_max_x: int = 40
    aim_max_y: int = 25

    # ── Возврат камеры к горизонту ─────────────────────────────────────────
    # pitch_accum — сумма РЕАЛЬНО отправленных в мышь counts по вертикали.
    # Чувствительность мыши знать не нужно: возврат идёт теми же counts,
    # которыми камеру задрали.
    pitch_max_counts: int = 250      # жёсткий предел наклона камеры
    pitch_recenter_after: int = 6    # шагов без цели до начала возврата
    pitch_recenter_step: int = 6     # counts за шаг при возврате
    pitch_debug: bool = True

    # Раздельный гейт стрельбы по осям (вместо hypot < dist_close):
    # горизонтально ассист наводит точно, по вертикали допуск шире.
    shoot_gate_x_px: int = 60
    shoot_gate_y_px: int = 90

    # ── Анти-спавн / базовое движение ──────────────────────────────────────
    # PPO в начале обучения выбирает почти случайные действия и часто просто
    # стоит на спавне. Этот guard не даёт ему сидеть на месте: первые секунды
    # после спавна и когда цели нет — держим W, слегка страфимся и сканируем
    # горизонт мышью. Когда цель далеко — тоже подталкиваем вперёд.
    movement_guard_enabled: bool = True
    spawn_push_sec: float = 4.0
    no_target_roam_enabled: bool = True
    no_target_roam_turn_px: int = 18
    roam_strafe_period: float = 2.0
    force_forward_to_target: bool = True
    target_forward_until_dist: float = 120.0
    # Антизастревание больше не жмёт space, если это выключено: прыжки снова
    # могут превращаться в бессмысленную суету на спавне.
    unstuck_jump_enabled: bool = False

    # ── Антистена v3 ──────────────────────────────────────────────────────
    # Старый антизастреватель смотрел только на motion ratio. Если бот упёрся
    # в стену, но продолжал водить камерой, картинка менялась и детектор думал,
    # что движение есть. Этот слой отдельно ловит близкую стену перед прицелом.
    wall_escape_enabled: bool = True
    wall_escape_frames_needed: int = 8
    wall_escape_min_age_sec: float = 1.0
    wall_escape_hold_sec: float = 2.2
    wall_escape_cooldown_sec: float = 4.0
    wall_escape_turn_px: int = 240
    wall_escape_drive_turn_px: int = 0
    wall_roi_w_frac: float = 0.24
    wall_roi_h_frac: float = 0.28
    wall_gray_std_max: float = 22.0
    wall_edge_ratio_max: float = 0.055
    wall_debug: bool = False

    dist_close:       float = 80.0
    dist_near:        float = 160.0
    dist_max:         float = 300.0
    action_log_every: int   = 10
    det_stats_every:  int   = 300   # каждые N шагов логировать % кадров с видимым врагом

    # ─────────────────────────────────────────────────────────────────────────
    # ── Детектор фрага: счётчик «+N» под прицелом (detectors.py) ───────
    plus_roi: list        = field(default_factory=lambda: [0.464, 0.270, 0.559, 0.345])
    plus_v_min: int       = 220     # яркость: счётчик всегда горит на максимуме
    plus_s_min: int       = 150     # насыщенность: отсекает белый текст HUD
    plus_white_fallback: bool = True   # запасная белая маска, если цветная пуста
    plus_white_v_min: int = 230
    plus_white_s_max: int = 60
    plus_anchor_frac: float = 0.4805   # доля ширины окна, где стоит глиф «+»
    plus_anchor_tol_px: int = 9
    plus_anchor_w_min: int  = 9
    plus_anchor_w_max: int  = 26
    plus_anchor_h_min: int  = 7
    plus_anchor_h_max: int  = 26
    plus_glyph_w_min: int   = 3
    plus_glyph_w_max: int   = 34
    plus_glyph_h_min: int   = 8
    plus_glyph_h_max: int   = 32
    plus_glyph_area_min: int = 22
    plus_line_tol_px: int   = 7
    plus_digit_gap_px: int  = 40
    plus_span_min: int      = 16
    plus_span_max: int      = 88
    plus_hold_sec: float    = 0.35   # сколько держаться до засчёта фрага
    plus_cooldown_sec: float = 0.9
    plus_digit_hamming: int = 8      # отличие отпечатков = смена цифры
    plus_off_frames: int    = 5
    plus_change_frames: int = 2
    plus_debug: bool        = True

    # ── Детектор смерти: заливка панели респавна ──────────────────
    death_roi: list       = field(default_factory=lambda: [0.597, 0.691, 0.729, 0.833])
    death_gray_min: int   = 45
    death_fill_min: float = 0.30    # живой макс 7%, смерть мин 60% — запас x8
    death_hud_right_roi: list = field(default_factory=lambda: [0.50, 0.925, 0.60, 1.0])
    death_hud_core_roi: list = field(default_factory=lambda: [0.43, 0.925, 0.50, 1.0])
    death_hud_right_max: float = 1.5
    death_hud_core_min: float = 3.0
    death_hud_white_v: int = 200
    death_hud_white_s: int = 70
    death_require_hole: bool = True
    death_hold_frames: int = 2
    death_cooldown_sec: float = 5.0
    death_debug: bool     = True

    kill_after_death_veto_sec: float = 1.5   # глушим счётчик после смерти

    @classmethod
    def load(cls, path: str = "config.json") -> "Config":
        obj = cls()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for k, v in json.load(f).items():
                    if hasattr(obj, k):
                        setattr(obj, k, v)
            log.info(f"[CONFIG] Загружена: {path}")
        return obj

    def save(self, path: str = "config.json"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        d = asdict(self)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        log.info(f"[CONFIG] Сохранена: {path}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  YOLO-ВОРКЕР                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class YOLOWorker:
    def __init__(self, model: YOLO, cfg: Config):
        self._model    = model
        self._cfg      = cfg
        # FP16 на Pascal (GTX 1050/1060) идёт в 1/64 скорости FP32 — включать
        # только осознанно через config: "yolo_half": true (для RTX/Turing+).
        self._half     = (bool(getattr(cfg, "yolo_half", False))
                          and cfg.yolo_device.lower().startswith("cuda"))

        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._result:  Tuple[Optional[float], Optional[Tuple[float, float]]] = (None, None)
        # v13: доп. сведения о цели — уверенность и "под прицелом союзник"
        self._info: Dict[str, Any] = {"conf": 0.0, "mate_block": False, "mates": 0}
        self._lock     = threading.Lock()
        self._err_logged = False

        t = threading.Thread(target=self._loop, daemon=True, name="YOLOWorker")
        t.start()
        log.info(f"[YOLO] Воркер запущен | {cfg.yolo_device.upper()}")

    @staticmethod
    def _iou(a, b) -> float:
        """Пересечение боксов. Нужно, чтобы отсеять врага, который на
        самом деле тот же игрок, что уже размечен как союзник."""
        x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
        x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
        iw = max(0.0, float(x1 - x0)); ih = max(0.0, float(y1 - y0))
        inter = iw * ih
        if inter <= 0:
            return 0.0
        aa = max(1.0, float((a[2] - a[0]) * (a[3] - a[1])))
        bb = max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))
        return inter / (aa + bb - inter)

    def _nameplate_glyph_line(self, frame: np.ndarray, box: np.ndarray):
        """Ищет горизонтальную строку из букв над bbox.

        Возвращает (n_букв, ширина, dy_над_bbox) или None.
        Калибровка взята с реальных кадров debug_frames.
        """
        C = self._cfg
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        pad = int(getattr(C, "nameplate_pad_px", 30))
        ry1 = max(0, y1 - int(getattr(C, "nameplate_up_px", 75)))
        ry2 = max(1, y1 - int(getattr(C, "nameplate_down_px", 2)))
        rx1 = max(0, x1 - pad)
        rx2 = min(w, x2 + pad)
        if rx2 - rx1 < 8 or ry2 - ry1 < 6:
            return None

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]
        m = ((S <= int(getattr(C, "nameplate_glyph_s_max", 70))) &
             (V >= int(getattr(C, "nameplate_glyph_v_min", 175)))).astype(np.uint8)
        if int(m.sum()) < 6:
            return None

        n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
        wmin = int(getattr(C, "nameplate_glyph_w_min", 2))
        wmax = int(getattr(C, "nameplate_glyph_w_max", 22))
        hmin = int(getattr(C, "nameplate_glyph_h_min", 4))
        hmax = int(getattr(C, "nameplate_glyph_h_max", 20))
        amin = int(getattr(C, "nameplate_glyph_area_min", 6))
        idx = [i for i in range(1, n)
               if wmin <= st[i][2] <= wmax and hmin <= st[i][3] <= hmax and st[i][4] >= amin]
        need = int(getattr(C, "nameplate_glyph_min", 3))
        if len(idx) < need:
            return None

        tol = int(getattr(C, "nameplate_line_tol_px", 6))
        ys = [st[i][1] + st[i][3] * 0.5 for i in idx]
        best = []
        for cy in ys:
            sel = [i for i, yy in zip(idx, ys) if abs(yy - cy) <= tol]
            if len(sel) > len(best):
                best = sel
        if len(best) < need:
            return None

        gx0 = min(st[i][0] for i in best)
        gx1 = max(st[i][0] + st[i][2] for i in best)
        gy0 = min(st[i][1] for i in best)
        nglyph = len(best)
        line_w = int(gx1 - gx0)
        up_px  = int(y1 - (ry1 + gy0))

        # patch10: геометрическая проверка правдоподобия — отсекает кладку
        bw = float(box[2]) - float(box[0])
        bh = float(box[3]) - float(box[1])
        if nglyph > int(getattr(C, "nameplate_glyph_max", 10)):
            return None
        if line_w > int(getattr(C, "nameplate_line_w_max", 95)):
            return None
        up_lim = max(14.0, bh * float(getattr(C, "nameplate_up_ratio_max", 1.5)))
        if up_px > up_lim:
            return None
        line_cx = float(rx1) + (gx0 + gx1) * 0.5
        box_cx  = (float(box[0]) + float(box[2])) * 0.5
        c_lim = max(14.0, bw * float(getattr(C, "nameplate_center_tol_ratio", 0.8)))
        if abs(line_cx - box_cx) > c_lim:
            return None

        return (nglyph, line_w, up_px)

    def _box_has_teammate_nameplate(self, frame: np.ndarray, box: np.ndarray) -> bool:
        """True, если над bbox видна строка ника/уровня союзника.

        Это не OCR: мы не читаем буквы, а ищем компактную яркую текстовую
        полоску над головой. Хорошо работает для одноклассовой YOLO, когда
        модель видит и врагов, и союзников как `player`, но у союзников есть
        nameplate над головой.
        """
        C = self._cfg
        if not getattr(C, "teammate_nameplate_filter", True):
            return False

        if str(getattr(C, "nameplate_mode", "glyph")).lower() == "glyph":
            line = self._nameplate_glyph_line(frame, box)
            if line is not None and getattr(C, "nameplate_debug", False):
                log.info(f"[NICK] свой: букв={line[0]} ширина={line[1]}px выше_головы={line[2]}px "
                         f"box={int(box[2]-box[0])}x{int(box[3]-box[1])}")
            return line is not None

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)

        # ROI над головой. Расширяем по X, потому что ник обычно шире тела.
        rx1 = max(0, int(x1 - bw * float(getattr(C, "nameplate_x_expand", 0.65))))
        rx2 = min(w, int(x2 + bw * float(getattr(C, "nameplate_x_expand", 0.65))))
        ry1 = max(0, int(y1 - bh * float(getattr(C, "nameplate_y_top", 0.70))))
        ry2 = max(0, int(y1 - bh * float(getattr(C, "nameplate_y_bottom", 0.03))))

        if rx2 <= rx1 or ry2 <= ry1:
            return False

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Белый/серый текст ника/уровня.
        mask = ((S <= int(getattr(C, "nameplate_white_s_max", 95))) &
                (V >= int(getattr(C, "nameplate_white_v_min", 135))))

        # Жёлтый/салатовый текст уровня или ника.
        if getattr(C, "nameplate_use_yellow", True):
            yellow = ((H >= int(getattr(C, "nameplate_yellow_h_min", 15))) &
                      (H <= int(getattr(C, "nameplate_yellow_h_max", 45))) &
                      (S >= int(getattr(C, "nameplate_yellow_s_min", 70))) &
                      (V >= int(getattr(C, "nameplate_yellow_v_min", 120))))
            mask = mask | yellow

        # Зелёный текст ника. Выключен по умолчанию: карта часто зелёная.
        if getattr(C, "nameplate_use_green", False):
            green = ((H >= int(getattr(C, "nameplate_green_h_min", 35))) &
                     (H <= int(getattr(C, "nameplate_green_h_max", 90))) &
                     (S >= int(getattr(C, "nameplate_green_s_min", 80))) &
                     (V >= int(getattr(C, "nameplate_green_v_min", 140))))
            mask = mask | green

        mask_u8 = (mask.astype(np.uint8) * 255)
        # Лёгкое соединение букв в строку, чтобы измерить ширину ника.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)

        pixels = int(np.count_nonzero(mask_u8))
        total = int(mask_u8.size)
        if total <= 0:
            return False

        ratio = pixels / total
        min_ratio = float(getattr(C, "nameplate_min_ratio", 0.006))
        max_ratio = float(getattr(C, "nameplate_max_ratio", 0.18))
        min_pixels = int(getattr(C, "nameplate_min_pixels", 10))

        if pixels < min_pixels or ratio < min_ratio or ratio > max_ratio:
            return False

        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            return False
        text_w = int(xs.max() - xs.min() + 1)
        text_h = int(ys.max() - ys.min() + 1)

        # Ник — это горизонтальная строка, а не случайная точка/блик.
        min_text_w = max(6, int(bw * float(getattr(C, "nameplate_min_width_frac", 0.35))))
        ok = text_w >= min_text_w and text_h >= 2

        if ok and getattr(C, "nameplate_debug", False):
            log.info(f"[YOLO] teammate nameplate: ratio={ratio:.3f} pixels={pixels} "
                     f"span={text_w}x{text_h} box=({int(x1)},{int(y1)},{int(x2)},{int(y2)})")

        return bool(ok)

    def _remove_teammates_by_nameplate(self, frame: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
        """Возвращает только те bbox, над которыми НЕ найден ник тиммейта."""
        if not getattr(self._cfg, "teammate_nameplate_filter", True):
            return boxes_xyxy
        if boxes_xyxy.size == 0:
            return boxes_xyxy

        kept = []
        skipped = 0
        for box in boxes_xyxy:
            if self._box_has_teammate_nameplate(frame, box):
                skipped += 1
                continue
            kept.append(box)

        self._np_skipped = getattr(self, "_np_skipped", 0) + skipped
        self._np_total = getattr(self, "_np_total", 0) + len(boxes_xyxy)
        self._np_calls = getattr(self, "_np_calls", 0) + 1
        if self._np_calls % 150 == 0:
            log.info(f"[NICK] за 150 кадров: игроков={self._np_total} из них своих={self._np_skipped} "
                     f"({100.0*self._np_skipped/max(1,self._np_total):.0f}%) → врагов={self._np_total-self._np_skipped}")
            self._np_skipped = 0
            self._np_total = 0

        if not kept:
            return np.empty((0, 4), dtype=np.float32)
        return np.asarray(kept, dtype=np.float32)

    # v18.2: диагностика самого кадра, который ушёл в YOLO. Если яркость
    # близка к 0 или доля дублей высокая — воркер смотрит в чёрный/замерший
    # кадр, и тогда виноват захват, а не фильтры.
    def _fdiag(self, frame):
        try:
            D = getattr(self, "_fd", None)
            if D is None:
                D = {"n": 0, "w": 0, "h": 0, "bright": 0.0, "dup": 0}
                self._fd = D
            sm = frame[::16, ::16]
            D["n"] += 1
            D["h"] = int(frame.shape[0])
            D["w"] = int(frame.shape[1])
            D["bright"] += float(sm.mean())
            prev = getattr(self, "_fd_prev", None)
            if (prev is not None and prev.shape == sm.shape
                    and np.array_equal(prev, sm)):
                D["dup"] += 1
            self._fd_prev = sm.copy()
        except Exception:
            pass

    # v18.2: сохраняем кадры, которые реально ушли в YOLO, чтобы прогнать по ним
    # ту же модель офлайн (debug_folder.py) и сравнить с debug_yolo_win.py.
    def _dump(self, frame, nraw=0, maxconf=0.0):
        try:
            every = int(getattr(self._cfg, "yolo_dump_every", 0))
            if every <= 0:
                return
            i = int(getattr(self, "_dmp_i", 0)) + 1
            self._dmp_i = i
            if i % every:
                return
            n = int(getattr(self, "_dmp_n", 0))
            if n >= int(getattr(self._cfg, "yolo_dump_max", 80)):
                return
            d = str(getattr(self._cfg, "yolo_dump_dir", "yolo_dump"))
            os.makedirs(d, exist_ok=True)
            name = "f%05d_raw%d_conf%02d.jpg" % (
                i, int(nraw), int(round(float(maxconf) * 100)))
            cv2.imwrite(os.path.join(d, name), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            self._dmp_n = n + 1
            if self._dmp_n == 1:
                log.info("[ДАМП YOLO] пишу кадры в %s/ (каждый %d-й, максимум %d)"
                         % (d, every, int(getattr(self._cfg, "yolo_dump_max", 80))))
        except Exception:
            pass

    # v18.1: воронка фильтров. Показывает, на каком шаге теряются
    # боксы: сырые -> размер -> форма -> устойчивость -> вето -> ник.
    def _funnel(self, calls=0, raw=0, size=0, shape=0, persist=0,
                veto=0, nick=0, hits=0):
        F = getattr(self, "_fn", None)
        if F is None:
            F = {"calls": 0, "raw": 0, "size": 0, "shape": 0,
                 "persist": 0, "veto": 0, "nick": 0, "hits": 0}
            self._fn = F
        F["calls"] += int(calls)
        F["raw"] += int(raw)
        F["size"] += int(size)
        F["shape"] += int(shape)
        F["persist"] += int(persist)
        F["veto"] += int(veto)
        F["nick"] += int(nick)
        F["hits"] += int(hits)
        every = int(getattr(self._cfg, "yolo_funnel_every", 100))
        if every > 0 and F["calls"] >= every:
            D = getattr(self, "_fd", None) or {}
            tail = ""
            if int(D.get("n", 0)) > 0:
                tail = (" | кадр=%dx%d ярк=%.0f дублей=%.0f%%"
                        % (D.get("w", 0), D.get("h", 0),
                           float(D.get("bright", 0.0)) / max(1, int(D["n"])),
                           100.0 * int(D.get("dup", 0)) / max(1, int(D["n"]))))
            log.info(
                "[ВОРОНКА YOLO] за %d прогонов: сырых=%d -> размер=%d"
                " -> форма=%d -> устойчивость=%d -> вето=%d -> ник=%d"
                " | кадров с целью=%d%s"
                % (F["calls"], F["raw"], F["size"], F["shape"],
                   F["persist"], F["veto"], F["nick"], F["hits"], tail))
            self._fn = None
            self._fd = None

    def _loop(self):
        while True:
            frame = self._q.get()
            if frame is None:
                break
            try:
                h, w = frame.shape[:2]
                cx, cy = w / 2, h / 2
                self._fdiag(frame)

                with th.inference_mode():
                    res = self._model.predict(
                        frame,
                        verbose=False,
                        conf=self._cfg.yolo_conf,
                        imgsz=self._cfg.yolo_imgsz,
                        device=self._cfg.yolo_device,
                        half=self._half,
                        max_det=10,
                    )

                boxes = res[0].boxes
                if boxes is None or len(boxes) == 0:
                    self._dump(frame, 0, 0.0)
                    self._funnel(calls=1)
                    with self._lock:
                        self._result = (None, None)
                        self._info = {"conf": 0.0, "mate_block": False, "mates": 0}
                    continue

                C = self._cfg
                xyxy_all = boxes.xyxy.detach().float().cpu().numpy()
                try:
                    conf_all = boxes.conf.detach().float().cpu().numpy()
                except Exception:
                    conf_all = np.ones(len(xyxy_all), np.float32)
                self._dump(frame, len(xyxy_all),
                           float(conf_all.max()) if len(conf_all) else 0.0)
                try:
                    cls_all = boxes.cls.detach().int().cpu().numpy()
                except Exception:
                    cls_all = np.zeros(len(xyxy_all), np.int32)

                # 1) РАЗДЕЛЯЕМ врагов и союзников по классам YOLO.
                # Модель train-13 обучена на data.yaml: 0=enemy, 1=teammate.
                # Раньше filter_enemy_class=False -> союзники шли в цели как есть,
                # это и была главная причина стрельбы по своим.
                enemy_id = int(getattr(C, "enemy_class_id", 0))
                if bool(getattr(C, "filter_enemy_class", True)):
                    e_mask = (cls_all == enemy_id)
                else:
                    e_mask = np.ones(len(xyxy_all), bool)
                mates = xyxy_all[~e_mask] if bool(getattr(C, "filter_enemy_class", True)) \
                    else np.empty((0, 4), np.float32)
                xyxy = xyxy_all[e_mask]
                conf = conf_all[e_mask]

                # 2) Мелкие боксы — шум/далёкая мелочь, по ним всё равно не попасть.
                if xyxy.size:
                    bw = xyxy[:, 2] - xyxy[:, 0]
                    bh = xyxy[:, 3] - xyxy[:, 1]
                    keep = ((bw >= int(getattr(C, 'yolo_min_box_w', 10))) &
                            (bh >= int(getattr(C, 'yolo_min_box_h', 24))) &
                            ((bw * bh) >= int(getattr(C, 'yolo_min_box_area', 250))))
                    xyxy, conf = xyxy[keep], conf[keep]

                _n_raw = int(len(xyxy_all))
                _n_size = int(len(xyxy))
                _n_shape = _n_persist = _n_veto = _n_nick = _n_size

                # 2b) v18: ФИЛЬТР ФОРМЫ. Человек всегда вытянут по вертикали.
                # На живом сервере в логе были ложные 28x176 (h/w=6.3),
                # 30x138 (4.6), 30x111 (3.7), 11x66 (6.0) и плоские 56x60 (1.07).
                if xyxy.size and bool(getattr(C, "target_shape_filter", True)):
                    _bw = np.maximum(xyxy[:, 2] - xyxy[:, 0], 1.0)
                    _bh = xyxy[:, 3] - xyxy[:, 1]
                    _hw = _bh / _bw
                    _lo = float(getattr(C, "target_hw_min", 1.5))
                    _hi = float(getattr(C, "target_hw_max", 3.2))
                    keep = (_hw >= _lo) & (_hw <= _hi)
                    self._shape_drop = getattr(self, "_shape_drop", 0) + int((~keep).sum())
                    xyxy, conf = xyxy[keep], conf[keep]

                _n_shape = int(len(xyxy))

                # 2c) v18: УСТОЙЧИВОСТЬ. Настоящий игрок виден и на прошлом
                # прогоне YOLO; одиночный всплеск conf на текстуре — нет.
                _need_p = int(getattr(C, "target_persist_frames", 2))
                if xyxy.size and _need_p > 1:
                    _prev = getattr(self, "_prev_boxes", None)
                    _piou = float(getattr(C, "target_persist_iou", 0.2))
                    keep = np.zeros(len(xyxy), bool)
                    _tol = float(getattr(C, "target_persist_tol_px", 60.0))
                    if _prev is not None and len(_prev):
                        for i in range(len(xyxy)):
                            _b = xyxy[i]
                            _bcx = (float(_b[0]) + float(_b[2])) * 0.5
                            _bcy = (float(_b[1]) + float(_b[3])) * 0.5
                            _sz = max(float(_b[2] - _b[0]), float(_b[3] - _b[1]))
                            for p in _prev:
                                # v18.1: только IoU было слишком жёстко — при
                                # повороте камеры бокс успевает уехать целиком.
                                _d = (((_bcx - (float(p[0]) + float(p[2])) * 0.5) ** 2) +
                                      ((_bcy - (float(p[1]) + float(p[3])) * 0.5) ** 2)) ** 0.5
                                if (self._iou(_b, p) >= _piou
                                        or _d <= (_tol + 0.5 * _sz)):
                                    keep[i] = True
                                    break
                    self._prev_boxes = xyxy.copy()
                    self._persist_drop = getattr(self, "_persist_drop", 0) + int((~keep).sum())
                    xyxy, conf = xyxy[keep], conf[keep]
                else:
                    self._prev_boxes = xyxy.copy() if xyxy.size else np.empty((0, 4), np.float32)

                _n_persist = int(len(xyxy))
                _n_veto = _n_nick = _n_persist

                # 3) Спорный бокс: "враг" сильно перекрыт боксом "союзник" —
                # значит это один и тот же игрок и модель сомневается. В таких
                # случаях всегда считаем, что это свой, и цель выбрасываем.
                veto = float(getattr(C, "teammate_iou_veto", 0.25))
                if xyxy.size and mates.size and veto > 0:
                    keep = np.ones(len(xyxy), bool)
                    for i in range(len(xyxy)):
                        for m in mates:
                            if self._iou(xyxy[i], m) >= veto:
                                keep[i] = False
                                break
                    xyxy, conf = xyxy[keep], conf[keep]

                _n_veto = int(len(xyxy))

                # 4) Вторая линия защиты — ник/nameplate над головой.
                # Нужна только для старой одноклассовой YOLO; при рабочем
                # фильтре классов её лучше держать выключенной (лишний CPU).
                if xyxy.size and bool(getattr(C, "teammate_nameplate_filter", False)):
                    keep = np.ones(len(xyxy), bool)
                    for i in range(len(xyxy)):
                        if self._box_has_teammate_nameplate(frame, xyxy[i]):
                            keep[i] = False
                    self._np_skipped = getattr(self, "_np_skipped", 0) + int((~keep).sum())
                    self._np_total = getattr(self, "_np_total", 0) + len(keep)
                    self._np_calls = getattr(self, "_np_calls", 0) + 1
                    if self._np_calls % 50 == 0:
                        log.info("[NICK] за 50 прогонов: боксов=%d признано своими=%d (%.0f%%)"
                                 % (self._np_total, self._np_skipped,
                                    100.0 * self._np_skipped / max(1, self._np_total)))
                        self._np_skipped = 0
                        self._np_total = 0
                    xyxy, conf = xyxy[keep], conf[keep]
                _n_nick = int(len(xyxy))

                # 5) Союзник ПОД ПРИЦЕЛОМ — жёсткий запрет на огонь в step().
                # Даже если рядом виден враг: попасть в своего важнее не дать.
                pad = int(getattr(C, "mate_block_pad_px", 24))
                mate_block = False
                for m in mates:
                    if (m[0] - pad) <= cx <= (m[2] + pad) and (m[1] - pad) <= cy <= (m[3] + pad):
                        mate_block = True
                        break

                self._funnel(calls=1, raw=_n_raw, size=_n_size, shape=_n_shape,
                             persist=_n_persist, veto=_n_veto, nick=_n_nick,
                             hits=1 if xyxy.size else 0)

                if xyxy.size == 0:
                    with self._lock:
                        self._result = (None, None)
                        self._info = {"conf": 0.0, "mate_block": mate_block,
                                      "mates": int(len(mates))}
                    continue

                centers = (xyxy[:, :2] + xyxy[:, 2:]) * 0.5
                dists = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
                best = int(np.argmin(dists))

                # Точка прицеливания: не центр бокса (это живот), а верхняя
                # треть — корпус/шея. Попаданий с тем же ассистом заметно больше.
                yf = float(getattr(C, "aim_target_y_frac", 0.35))
                bx = xyxy[best]
                aim_pt = (float(centers[best, 0]),
                          float(bx[1] + (bx[3] - bx[1]) * yf))

                with self._lock:
                    self._result = (float(dists[best]), aim_pt)
                    self._info = {
                        "conf": float(conf[best]) if len(conf) > best else 1.0,
                        "mate_block": mate_block,
                        "mates": int(len(mates)),
                        # v18: размер выбранной цели нужен step() для порога conf
                        "box_h": float(bx[3] - bx[1]),
                        "box_w": float(bx[2] - bx[0]),
                    }

            except Exception:
                with self._lock:
                    self._result = (None, None)
                if not self._err_logged:
                    self._err_logged = True
                    log.error("[YOLO] Ошибка в воркере:")
                    traceback.print_exc()

    def submit(self, frame: np.ndarray):
        try:
            self._q.put_nowait(frame.copy())
        except queue.Full:
            pass

    def get(self) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
        with self._lock:
            return self._result

    def get_info(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._info)

    def stop(self):
        for _ in range(3):
            try:
                self._q.put_nowait(None)
                break
            except queue.Full:
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  KILL-ДЕТЕКТОР (единственный источник сигнала о фраге)                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class KillDetector:
    """
    Ищет появление шаблона своего никнейма (вырезанного из killfeed) в правом
    верхнем углу экрана.

    ВАЖНО про цвет: ник в killfeed красится в цвет текущей команды (красный /
    синий / зелёный / жёлтый), поэтому обычный grayscale (cv2.COLOR_BGR2GRAY)
    здесь не работает — у разных цветов совершенно разная яркость по формуле
    люминансности (Y = 0.299R + 0.587G + 0.114B). Например, чистый красный
    (0,0,255 BGR) даёт в grayscale яркость ~76, зелёный (0,255,0) — ~150,
    жёлтый — ~226, а синий — всего ~29. Для matchTemplate это четыре визуально
    разные картинки, и шаблон, снятый с одного цвета, не совпадёт с другим.

    Решение — бинаризация по V-каналу HSV (Value = max(R,G,B)/255). Для всех
    перечисленных ярких цветов текста V близок к максимуму (255) независимо
    от Hue, а тёмный фон killfeed даёт низкий V. Порог по V превращает буквы
    Л��БОГО яркого цвета в чисто белый силуэт (255) на чёрном фоне (0) —
    геометрия букв сохраняется, а цвет команды перестаёт иметь значение.
    Шаблон бинаризуется тем же способом при загрузке.

    Засчитывает фраг строго на переходе «силуэта не было -> силуэт появился»,
    с кулдауном, чтобы одна и та же строка в killfeed не давала награду
    повторно, пока она висит на экране.
    """

    # Порог по V-каналу HSV: выше — «яркий текст», ниже — «тёмный фон killfeed».
    V_THRESH = 150

    def _binarize(self, bgr: np.ndarray) -> np.ndarray:
        v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, mask = cv2.threshold(v, self.V_THRESH, 255, cv2.THRESH_BINARY)
        return mask

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tpl_mask: Optional[np.ndarray] = None
        path = cfg.kill_template_path
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                self.tpl_mask = self._binarize(img)
                log.info(f"[KILL] Шаблон ника загружен: {path} "
                         f"({img.shape[1]}×{img.shape[0]})")
        if self.tpl_mask is None:
            log.warning(f"[KILL] Шаблон не найден: {path} — детектор фрагов ОТКЛЮЧЁН "
                        f"(награда за килл начисляться не будет).")

        self._present_prev = False
        self._last_kill_t  = -999.0
        self._nkill_saved  = 0
        self.last_score = 0.0
        self.last_loc = (0, 0)
        self.last_right_gap = -1

    def check(self, full_frame: np.ndarray, recent_shot_ok: bool = True, recent_enemy_ok: bool = True) -> bool:
        """Возвращает True строго один раз за реальный killfeed-событие.

        Дополнительные проверки recent_shot_ok/recent_enemy_ok нужны, чтобы не
        засчитывать твой ник в верхней HUD-плашке или в таблице счёта как фраг.
        """
        if self.tpl_mask is None:
            return False

        h, w = full_frame.shape[:2]

        # ROI берём из конфига. По умолчанию y0=0.07, чтобы НЕ захватывать
        # верхнюю постоянную строку с ником игрока ([21]aponima MIDGARD ...).
        x0, y0, x1, y1 = getattr(self.cfg, "kill_roi", [0.50, 0.07, 1.00, 0.30])
        x0i, y0i = int(w * x0), int(h * y0)
        x1i, y1i = int(w * x1), int(h * y1)
        roi = full_frame[y0i:y1i, x0i:x1i]
        if roi.size == 0:
            return False

        mask = self._binarize(roi)

        th_, tw_ = self.tpl_mask.shape[:2]
        if mask.shape[0] < th_ or mask.shape[1] < tw_:
            self._present_prev = False
            return False

        res = cv2.matchTemplate(mask, self.tpl_mask, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        present = score >= self.cfg.kill_match_thresh

        # --- v6 -------------------------------------------------------
        # Tvoy nik poyavlyaetsya v killfeed V DVUH ROLYAH: sleva (ty ubil)
        # i sprava (ubili tebya). Shablon odin i tot zhe, poetomu SMERT
        # zaschityvalas kak FRAG. Zazor do pravogo kraya ROI ih razlichaet.
        roi_w = mask.shape[1]
        right_gap = roi_w - (loc[0] + tw_)
        self.last_score = float(score)
        self.last_loc = (int(loc[0]), int(loc[1]))
        self.last_right_gap = int(right_gap)
        if present and getattr(self.cfg, "kill_side_filter", False):
            if right_gap <= int(getattr(self.cfg, "kill_victim_margin_px", 30)):
                if getattr(self.cfg, "kill_debug", False):
                    log.info(f"[KILL] otkloneno side_filter: zazor_sprava={right_gap}px")
                present = False

        now = time.monotonic()
        killed = False

        if present and not self._present_prev and (now - self._last_kill_t) >= self.cfg.kill_cooldown_sec:
            blocked_reason = None
            if getattr(self.cfg, "kill_require_recent_shot", True) and not recent_shot_ok:
                blocked_reason = "не было недавней стрельбы"
            elif getattr(self.cfg, "kill_require_recent_enemy", True) and not recent_enemy_ok:
                blocked_reason = "не было недавней цели YOLO"

            if blocked_reason is None:
                killed = True
                self._last_kill_t = now
                log.info(f"[KILL] score={score:.2f} loc={loc} "
                         f"zazor_sprava={right_gap}px ROI_w={roi_w}px")
                if getattr(self.cfg, "kill_debug_save", False) and self._nkill_saved < 60:
                    try:
                        os.makedirs("debug_kills", exist_ok=True)
                        self._nkill_saved += 1
                        _ts = time.strftime("%H%M%S")
                        cv2.imwrite(
                            f"debug_kills/{_ts}_n{self._nkill_saved}"
                            f"_s{int(score*100)}_gap{right_gap}.png", roi)
                    except Exception as _e:
                        log.warning(f"[KILL] ne sohranil krop: {_e}")
            elif getattr(self.cfg, "kill_debug", False):
                log.info(f"[KILL] Совпадение score={score:.2f} loc={loc}, но отклонено: {blocked_reason}")

        if present and getattr(self.cfg, "kill_debug", False):
            log.debug(f"[KILL] score={score:.2f} loc={loc} recent_shot={recent_shot_ok} recent_enemy={recent_enemy_ok}")

        self._present_prev = present
        return killed

    def reset(self):
        self._present_prev = False
        self._last_kill_t  = -999.0


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  AMMO READER — читает число патронов прямо с HUD через OCR               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class AmmoReader:
    """
    Разное оружие в Blockade 3D имеет разный размер магазина (от нескольких
    штук до ~300) и разное время перезарядки — зашить эти числа константами
    в конфиг невозможно, они всегда будут верны только для одного ствола.

    Вместо этого читаем РЕАЛЬНУЮ цифру патронов с HUD через OCR (Tesseract,
    режим "только цифры"). OCR — операция небольшая по площади (маленький
    ROI в углу экрана), но всё же дороже обычного template matching, поэтому
    гоняем её не каждый кадр (см. cfg.ammo_ocr_every), а результат
    кэшируется между вызовами.

    Если OCR не смог ничего распознать (плохой контраст, перекрыто
    HUD-эффектом урона/сообщением) — возвращаем None, и вызывающий код сам
    решает, что делать (см. Blockade3DEnv — используется резервный счётчик
    по выстрелам, пока OCR временно "молчит").
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._last_value: Optional[int] = None
        self._zero_streak = 0
        self._err_logged  = False

    def _binarize_digits(self, bgr: np.ndarray) -> np.ndarray:
        # HUD-цифры в большинстве мобильных/веб-шутеров — яркий текст
        # (белый/жёлтый) с обводкой на полупрозрачном тёмном фоне. Порог по
        # V-каналу HSV (как в KillDetector) даёт устойчивый бинарный силуэт
        # независимо от конкретного цвета текста.
        v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
        _, mask = cv2.threshold(v, 150, 255, cv2.THRESH_BINARY)
        return mask

    def read(self, full_frame: np.ndarray) -> Optional[int]:
        """Возвращает распознанное число патронов, либо None, если не удалось."""
        C = self.cfg
        h, w = full_frame.shape[:2]
        x0, y0, x1, y1 = C.ammo_roi
        roi = full_frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
        if roi.size == 0:
            return None

        try:
            mask = self._binarize_digits(roi)
            # Апскейл мелкого HUD-текста заметно повышает точность Tesseract.
            # Важно: не добавляем лишнюю рамку вокруг ROI, потому что на HUD
            # Blockade рядом с цифрами есть белые иконки патронов — с рамкой
            # Tesseract чаще принимает их за лишние единицы.
            mask = cv2.resize(mask, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

            # На HUD Blockade цифры маленькие, а рядом могут быть иконки
            # патронов. Поэтому пробуем несколько psm-режимов и берём первый
            # разумный числовой блок. Если HUD пишет "12/30", число до слэша
            # всё равно окажется первым.
            ocr_configs = (
                "--psm 7 -c tessedit_char_whitelist=0123456789/",
                "--psm 6 -c tessedit_char_whitelist=0123456789/",
                "--psm 8 -c tessedit_char_whitelist=0123456789/",
                "--psm 13 -c tessedit_char_whitelist=0123456789/",
            )
            for cfg in ocr_configs:
                text = pytesseract.image_to_string(mask, config=cfg).strip()
                if not text:
                    continue
                text = text.replace("\n", " ").replace("|", " ")
                first_part = text.split("/")[0]
                matches = re.findall(r"\d+", first_part)
                for digits in matches:
                    value = int(digits)
                    if 0 <= value <= C.ammo_max_reasonable:
                        self._last_value = value
                        return value
            return None
        except pytesseract.pytesseract.TesseractNotFoundError:
            # Tesseract не установлен или не в PATH. Не спамим ошибками —
            # выключаем OCR и отдаём управление fallback-счётчику патронов.
            if not self._err_logged:
                self._err_logged = True
                log.warning("[AMMO-OCR] Tesseract не найден — переключаюсь на счётчик патронов без OCR.")
            self.cfg.ammo_ocr_enabled = False
            return None
        except Exception:
            if not self._err_logged:
                self._err_logged = True
                log.warning("[AMMO-OCR] Ошибка распознавания (проверь ammo_roi / tesseract):")
                traceback.print_exc()
            return None

    def reset(self):
        self._last_value  = None
        self._zero_streak = 0


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  CALLBACKS                                                                 ║
# ╚═════════���══════════════════════════════════════════════════════════════════╝
class SaveNormCallback(BaseCallback):
    def __init__(self, save_freq: int, norm_path: str):
        super().__init__()
        self.save_freq = save_freq
        self.norm_path = norm_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.training_env.save(self.norm_path)
            log.info(f"[NORM] VecNormalize → {self.norm_path}")
        return True


# ── v15: МЯГКАЯ ОСТАНОВКА ПО ГОРЯЧЕЙ КЛАВИШЕ ────────────────────────────────
# Ctrl+C бросает KeyboardInterrupt в произвольной точке: он может прийти
# в середине градиентного апдейта или записи файла. F2 отрабатывает
# штатно: PPO доводит текущий шаг до конца, learn() выходит нормально,
# и блок finally спокойно сохраняет модель и VecNormalize.
# Клавиша читается глобально (GetAsyncKeyState), фокус окна не нужен.
_STOP = {"on": False}


def stop_hotkey_pressed(vk: int = 0x71) -> bool:
    """True, если клавиша остановки нажата. Флаг залипает навсегда."""
    if _STOP["on"]:
        return True
    try:
        if win32api.GetAsyncKeyState(int(vk)) & 0x8000:
            _STOP["on"] = True
    except Exception:
        return False
    return _STOP["on"]


class StopHotkeyCallback(BaseCallback):
    """Останавливает model.learn() по нажатию клавиши (по умолчанию F2).

    Возврат False из _on_step — штатный способ прервать обучение в SB3:
    learn() возвращает управление, и модель сохраняется как обычно.
    """

    def __init__(self, cfg):
        super().__init__()
        self.vk   = int(getattr(cfg, "stop_hotkey_vk", 0x71))
        self.name = str(getattr(cfg, "stop_hotkey_name", "F2"))

    def _on_step(self) -> bool:
        if stop_hotkey_pressed(self.vk):
            log.info(f"[СТОП] {self.name} нажата — останавливаю обучение "
                     f"и сохраняю модель.")
            return False
        return True


class StatsCallback(BaseCallback):
    # v14: по этим графикам можно объективно сравнивать версии модели.
    # tensorboard --logdir tb_logs -> раздел game/
    _KEYS = ("kills", "deaths", "kd", "shots", "accuracy",
             "visible_pct", "ff_blocked", "aim_w",
             # v16
             "trigger_pct", "gate_pct", "stuck_events", "alive_sec")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for k in self._KEYS:
                if k in info:
                    self.logger.record("game/%s" % k, info[k])
        return True


class ReleaseOnUpdateCallback(BaseCallback):
    """
    PPO останавливает сбор опыта на несколько секунд каждые n_steps, чтобы
    сделать градиентный апдейт (n_epochs проходов по батчам). Всё это время
    среда не вызывается — а значит зажатые клавиши (W/A/S/D, ЛКМ) остаются
    зажатыми, и бот продолжает "бежать вслепую" в стену на время апдейта.
    Отпускаем все клавиши перед апдейтом и сразу после него.
    """
    def __init__(self, raw_env: "Blockade3DEnv"):
        super().__init__()
        self.raw_env = raw_env

    def _on_rollout_end(self) -> None:
        self.raw_env._release_all()
        # v13: во время апдейта среда стоит секунды, картинка не меняется.
        # Без сброса базы движения антистак сразу после апдейта решает,
        # что бот застрял, и крутит его на месте.
        self.raw_env.motion_gray = None
        self.raw_env.last_gray = None
        self.raw_env.stuck_obs_f = 0
        self.raw_env.stuck_frz_f = 0
        # v14: среда не знает, сколько шагов уже обучено (step_counter
        # обнуляется при каждом запуске). Отдаём ей счётчик модели —
        # по нему затухает вес аим-ассиста.
        self.raw_env.global_steps = int(getattr(self.model, "num_timesteps", 0))

    def _on_training_start(self) -> None:
        self.raw_env._release_all()

    def _on_step(self) -> bool:
        return True


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ГЛАВНОЕ ОКРУЖЕНИЕ                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  HYBRID PPO — сбор опыта на CPU, градиенты на GPU                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝
class HybridPPO(PPO):
    """PPO, который держит политику на CPU и уезжает на GPU только на train().

    Замеры на GTX 1050 + Athlon X4 840 (bench_ppo2.py):

        forward batch=1   : CPU  16.5 мс   CUDA  28.0 мс  (под нагрузкой YOLO)
        backward b=128    : CPU 1292 мс   CUDA   117 мс

    Прямой проход крошечной сети на GPU упирается не в счёт, а в оверхед запуска
    ядер и в очередь за видеокартой, которую уже занял детектор. Обратный
    проход по минибатчу 128 — наоборот, на GPU быстрее в 11 раз.
    """

    rollout_device = "cpu"
    grad_device    = "cuda"

    def _switch(self, device: str) -> None:
        if device == "cuda" and not th.cuda.is_available():
            return
        dev = th.device(device)
        if str(self.device).split(":")[0] == str(dev).split(":")[0]:
            return
        self.policy.to(dev)
        self.device = dev
        # Adam хранит exp_avg/exp_avg_sq там, где последний раз делался step().
        # Без переноса состояния оптимизатор упадёт на device mismatch.
        opt = getattr(self.policy, "optimizer", None)
        if opt is not None:
            for st in opt.state.values():
                for k, v in list(st.items()):
                    if isinstance(v, th.Tensor):
                        st[k] = v.to(dev)
        # Буфер сам решает, куда класть тензоры в to_torch().
        buf = getattr(self, "rollout_buffer", None)
        if buf is not None:
            buf.device = dev

    def train(self) -> None:
        self._switch(self.grad_device)
        # v16: SB3 при создании модели печатает "Using cpu device" — это
        # НЕ потерянная видеокарта, а замысел гибрида: forward batch=1
        # на CPU быстрее, GPU нужен только на градиентном апдейте.
        # Один раз показываем, что переключение реально сработало.
        if not getattr(self, "_dev_logged", False):
            self._dev_logged = True
            log.info(f"[PPO] Апдейт на {str(self.device).upper()} | сбор опыта на "
                     f"{str(self.rollout_device).upper()} — гибрид работает.")
        try:
            super().train()
        finally:
            self._switch(self.rollout_device)


class Blockade3DEnv(gym.Env):
    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(self, cfg: Optional[Config] = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.recovery_planner = RecoveryPlanner()
        C = self.cfg

        # ── Захват окна ──────────────────────────────────────────────────────
        log.info("=" * 60)
        log.info("[СИСТЕМА] Разверни BLOCKADE на весь screen и кликни по нему.")
        for i in range(5, 0, -1):
            log.info(f"[ФОКУС]  Захват через {i}...")
            time.sleep(1)

        self.hwnd = win32gui.GetForegroundWindow()
        rect     = win32gui.GetClientRect(self.hwnd)
        self.x1, self.y1 = win32gui.ClientToScreen(self.hwnd, (0, 0))
        self.x2, self.y2 = win32gui.ClientToScreen(self.hwnd, (rect[2], rect[3]))
        self.game_w = self.x2 - self.x1
        self.game_h = self.y2 - self.y1
        log.info(f"[СИСТЕМА] Игровая зона: {self.game_w}×{self.game_h}px")
        log.info("=" * 60)

        # ── DXCam ─────────────────────────────────────────────────────────────
        self.camera = dxcam.create(output_idx=0, output_color="BGR")
        self.camera.start(
            region=(self.x1, self.y1, self.x2, self.y2),
            target_fps=C.target_fps,
        )
        time.sleep(0.5)

        # ── YOLO ─────────────────────────────────────────────────────────────
        yolo_path = C.yolo_path
        for fb in C.yolo_fallbacks:
            if not os.path.exists(yolo_path):
                yolo_path = fb

        yolo_dev = C.yolo_device
        if yolo_dev.startswith("cuda") and not th.cuda.is_available():
            log.warning("[YOLO] CUDA недоступна → переключаю на CPU")
            yolo_dev = "cpu"

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"[YOLO] Веса не найдены: {yolo_path}")

        log.info(f"[YOLO] Загрузка: {yolo_path} | {yolo_dev.upper()}")
        self.yolo_model = YOLO(yolo_path)
        self.yolo_model.to(yolo_dev)
        for p in self.yolo_model.model.parameters():
            p.requires_grad = False
        self.yolo_model.model.eval()
        self.yolo_worker = YOLOWorker(self.yolo_model, C)

        # ── Killfeed-детектор фрагов ─────────────────────────────────────────
        self.hud = HudDetectors(C)
        self.kill_detector = self.hud.kill   # совместимость со старым кодом
        self.last_shot_t = -999.0
        self.last_enemy_seen_t = -999.0

        # ── OCR-читалка патронов с HUD (не зависит от того, какое оружие) ────
        self.ammo_reader = AmmoReader(C)

        # ── Шаблон экрана смерти (template matching) ─────────────────────────
        self._rs_sat_max = C.respawn_sat_max
        self._rs_val_min = C.respawn_val_min
        self._rs_thresh  = C.respawn_threshold
        self.respawn_tpl = self._load_tpl("assets/respawn_lmb.png", self._white_mask, "ЭКРАН СМЕРТИ")

        # ── Шаблон меню выбора команды ─────────────────────────────────────
        self._tm_hue_min = C.team_hue_min
        self._tm_hue_max = C.team_hue_max
        self._tm_sat_min = C.team_sat_min
        self._tm_val_min = C.team_val_min
        self._tm_thresh  = C.team_threshold
        self.team_tpl    = self._load_tpl("assets/team_select.png", self._yellow_mask, "МЕНЮ КОМАНДЫ")

        # ── Шаблон кнопки "Бой" (опционально) ────────────────────────────────
        self._bt_thresh  = C.battle_threshold
        self.battle_tpl  = self._load_tpl(C.battle_template_path, self._white_mask, "КНОПКА БОЙ")

        # ── Пространства ─────────────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "frame":      spaces.Box(0, 255, (C.obs_size, C.obs_size,
                                              1 if getattr(C, "obs_gray", True) else 3),
                                     np.uint8),
            # v14: [дальность, dx, dy, скорость_x, скорость_y, давность]
            # Раньше было 3 значения и только "здесь и сейчас": если враг
            # уходил за угол, бот моментально забывал о нём.
            "enemy_info": spaces.Box(-1.0, 1.0, (6,), np.float32),
        })
        # Action space БЕЗ отдельного reload (r жать почти не нужно — оружие
        # само перезаряжается по клику/пусто, а держать R зажатым мешало
        # стрельбе) и с урезанной вертикалью мыши (5 вместо 7 значений —
        # бот реже задирает камеру в потолок/пол на старте обучения).
        # [fb, lr, mouse_x, mouse_y, shoot, jump] = 3*3*7*5*2*2 = 1260 комбинаций
        # (было 3528 с reload и полной вертикалью — сходимость быстрее).
        self.action_space = spaces.MultiDiscrete([3, 3, 7, 5, 2, 2])

        self._mx = [-50, -20, -6, 0, 6, 20, 50]
        self._my = [-8, -3, 0, 3, 8]

        # ── Глобальное состояние ─────────────────────────────────────────────
        self.total_kills  = 0
        self.total_deaths = 0
        self.step_counter = 0
        self.is_first_reset = True
        self.key_states:  Dict[str, bool] = {}
        self.mouse_left   = False

        self._last_rs_vis = False
        self._last_tm_vis = False
        self._last_tm_center: Optional[Tuple[int, int]] = None
        self._last_bt_vis = False
        # v14: троттлинг шаблонов по ВРЕМЕНИ, а не по номеру шага
        self._rs_t        = 0.0
        self._tm_t        = 0.0
        self._last_rs_xy  = (0, 0)
        self._cap_fail_t  = 0.0
        self.global_steps = 0
        self.aim_w_now    = float(getattr(C, "aim_assist_weight", 0.85))
        self.total_shots  = 0
        # v16: метрики стрельбы. total_shots теперь считает реально
        # выпущенные патроны (включая удержание курка), а не только
        # моменты нажатия ЛКМ.
        self._rounds_f     = 0.0
        self.total_trigger = 0
        self.total_gate    = 0
        self.stuck_events  = 0

        # Статистика видимости врага (для диагностики датасета YOLO)
        self._det_hits = 0

        # Сглаживание мыши
        self.last_step_time    = time.monotonic()
        self.last_smoothed_mx  = 0.0
        self.last_smoothed_my  = 0.0
        self.mouse_remainder_x = 0.0
        self.mouse_remainder_y = 0.0
        # Накопленный наклон камеры в counts (сумма отправленных дельт по Y).
        # Живёт СКВОЗЬ смерти: игра сохраняет наклон камеры после респавна.
        self.pitch_accum       = 0
        self.no_target_steps   = 0
        self.last_aimed_ok     = False
        self.last_jump_t       = -999.0  # Для кулдауна прыжка (см. jump_cooldown)

        # ── Автоматическая перезарядка (см. Config.auto_reload) ────────────
        # Основной источник истины — OCR-чтение реальной цифры патронов с
        # HUD (AmmoReader): работает для ЛЮБОГО оружия без подстройки чисел
        # под конкретный магазин. Пока OCR ещё не дал первое показание (None)
        # или временно "молчит" (плохой ракурс/перекрыт эффектом) — используем
        # запасной счётчик по числу реальных выстрелов с cfg.fallback_mag_size,
        # который OCR перезатирает своим значением, как только снова считает.
        self.ammo_left        = None   # None = ещё не знаем, ждём OCR/выстрел
        self.ammo_shots_fired = 0      # для запасного счётчика (fallback)
        self.reloading_until  = 0.0
        self._ammo_ocr_zero_streak = 0

        self.dead = False

        self._reset_episode()

    def _reset_episode(self):
        self.last_frame:  Optional[np.ndarray] = None
        self.last_gray:   Optional[np.ndarray] = None
        self.motion_gray: Optional[np.ndarray] = None
        self.escape_until = 0.0
        self.escape_sign  = 1
        self.escape_drive_lr = 1
        self.escape_drive_mx = 0
        self.last_esc_t   = -999.0
        self.stuck_obs_f  = 0
        self.front_wall_f = 0
        self.stuck_frz_f  = 0
        self.screen_frozen = False
        self.last_motion_ratio = 0.0

        self.spawn_t    = time.monotonic()
        self.prev_dist: Optional[float] = None

        self.dead = False

        self.ep_kills  = 0
        self.ep_deaths = 0

        # Сброс памяти сглаживания мыши на каждый новый эпизод/спавн
        self.last_smoothed_mx  = 0.0
        self.last_smoothed_my  = 0.0
        self.mouse_remainder_x = 0.0
        self.mouse_remainder_y = 0.0
        # ВАЖНО: pitch_accum здесь НЕ сбрасываем. Игра после смерти сохраняет
        # наклон камеры, поэтому накопленный счётчик остаётся валидным; сброс
        # заставил бы возврат отматывать уже отмотанное и увёл камеру в пол.
        self.no_target_steps   = 0
        self.last_aimed_ok     = False
        # v13
        self.last_target_conf  = 0.0
        self.last_target_h     = 0.0   # v18: высота бокса цели в px
        self.mate_block        = False
        self._death_confirm    = 0
        # v14: память о цели и счётчики эпизода для метрик
        self._tgt_prev         = None
        self._tgt_mem          = None
        self.ep_shots          = 0
        self.ep_steps          = 0
        self.ep_seen           = 0
        # v16: сколько раз политика жала огонь и сколько из этого
        # срезал гейт (нет цели / не наведён / низкая conf / свой)
        self.ep_trigger        = 0
        self.ep_gate           = 0

        # На новом спавне оружие обычно полное — но мы всё равно не гадаем
        # число, а сразу спрашиваем OCR на следующем шаге (см. step()).
        self.ammo_left        = None
        self.ammo_shots_fired = 0
        self.ammo_used_f      = 0.0
        self.reloading_until  = 0.0
        self._ammo_ocr_zero_streak = 0

        self.hud.reset()
        self.ammo_reader.reset()
        self.last_shot_t = -999.0
        self.last_enemy_seen_t = -999.0

    def _load_tpl(self, path, mask_fn, label):
        if not os.path.exists(path):
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        m = mask_fn(img)
        log.info(f"[{label}] Шаблон загружен ({img.shape[1]}×{img.shape[0]})")
        return m

    def _white_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv,
            np.array([0,   0,              self._rs_val_min], np.uint8),
            np.array([180, self._rs_sat_max, 255],            np.uint8))

    def _yellow_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        return cv2.inRange(hsv,
            np.array([self._tm_hue_min, self._tm_sat_min, self._tm_val_min], np.uint8),
            np.array([self._tm_hue_max, 255,              255],              np.uint8))

    def _match_mask(self, img_m, tpl_m):
        """matchTemplate стоит O(W*H*w*h) — на ROI 614x384 с шаблоном 545x77
        это ~900 млн операций и ~14 мс на слабом CPU. Уменьшение обоих
        в 2 раза режет стоимость в 16 раз. Крупные UI-плашки от этого не теряются.
        """
        if tpl_m is None:
            return 0.0, (0, 0)
        tmpl_h, tmpl_w = tpl_m.shape[:2]
        h, w    = img_m.shape[:2]
        if h < tmpl_h or w < tmpl_w:
            return 0.0, (0, 0)
        sc = max(1, int(getattr(self.cfg, "tpl_match_scale", 2)))
        if sc > 1 and tmpl_h >= sc * 10 and tmpl_w >= sc * 10:
            img_s = cv2.resize(img_m, (w // sc, h // sc), interpolation=cv2.INTER_AREA)
            tpl_s = cv2.resize(tpl_m, (tmpl_w // sc, tmpl_h // sc), interpolation=cv2.INTER_AREA)
            if img_s.shape[0] >= tpl_s.shape[0] and img_s.shape[1] >= tpl_s.shape[1]:
                r = cv2.matchTemplate(img_s, tpl_s, cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(r)
                return float(score), (loc[0] * sc, loc[1] * sc)
        r = cv2.matchTemplate(img_m, tpl_m, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(r)
        return float(score), loc

    def _restart_camera(self):
        """v14: после конца матча игра фризится и пересоздаёт D3D-устройство.
        dxcam при этом часто перестаёт отдавать новые кадры НАВСЕГДА: бот
        "видит" последний кадр старой карты и не реагирует ни на меню выбора
        команды, ни на смерть. Помогал только перезапуск скрипта руками —
        отсюда и "нажимает АВТО только когда я запущу код". Теперь
        перезапускаем сам захват, не трогая обучение.
        """
        C = self.cfg
        for _fn in ("stop", "release"):
            try:
                getattr(self.camera, _fn)()
            except Exception:
                pass
        self.camera = None
        try:
            dxcam.reset()      # без этого create() вернёт тот же мёртвый объект
        except Exception:
            pass
        time.sleep(0.5)
        # Окно могло быть пересоздано движком — заново берём геометрию.
        try:
            _rect = win32gui.GetClientRect(self.hwnd)
            _x1, _y1 = win32gui.ClientToScreen(self.hwnd, (0, 0))
            _x2, _y2 = win32gui.ClientToScreen(self.hwnd, (_rect[2], _rect[3]))
            if _x2 > _x1 and _y2 > _y1:
                if (_x2 - _x1, _y2 - _y1) != (self.game_w, self.game_h):
                    log.warning(f"[ЗАХВАТ] размер окна изменился: "
                                f"{self.game_w}x{self.game_h} -> "
                                f"{_x2 - _x1}x{_y2 - _y1}")
                self.x1, self.y1, self.x2, self.y2 = _x1, _y1, _x2, _y2
                self.game_w, self.game_h = _x2 - _x1, _y2 - _y1
        except Exception as _ge:
            log.warning(f"[ЗАХВАТ] геометрия окна: {_ge}")
        try:
            self.camera = dxcam.create(output_idx=0, output_color="BGR")
            self.camera.start(region=(self.x1, self.y1, self.x2, self.y2),
                              target_fps=C.target_fps)
            time.sleep(0.5)
            log.warning("[ЗАХВАТ] dxcam перезапущен.")
        except Exception as _ce:
            log.error(f"[ЗАХВАТ] не смог перезапустить dxcam: {_ce}")

    def _get_frame(self) -> np.ndarray:
        frame = self.camera.get_latest_frame() if self.camera is not None else None
        for _ in range(5):
            if frame is not None:
                break
            time.sleep(0.002)
            frame = self.camera.get_latest_frame() if self.camera is not None else None
        if frame is None:
            # v14: считаем, сколько времени захват молчит. Раньше молча
            # отдавался последний кадр — бот играл в стоп-кадр.
            _lim = float(getattr(self.cfg, "capture_restart_after", 5.0))
            if self._cap_fail_t <= 0.0:
                self._cap_fail_t = time.monotonic()
            elif time.monotonic() - self._cap_fail_t >= _lim:
                log.warning("[ЗАХВАТ] нет новых кадров "
                            f"{time.monotonic() - self._cap_fail_t:.0f}с — перезапуск.")
                self._cap_fail_t = 0.0
                self._restart_camera()
            frame = self.last_frame
        else:
            self._cap_fail_t = 0.0
        if frame is None:
            return np.zeros((self.game_h, self.game_w, 3), np.uint8)
        self.last_frame = frame
        return frame

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        s = self.cfg.obs_size
        # INTER_AREA с 640x640 сразу в 84x84 считает все исходные пиксели.
        # Сначала режем срезом до ~2s, потом честный INTER_AREA — картинка та же.
        h, w = frame.shape[:2]
        st = max(1, min(h // (s * 2), w // (s * 2)))
        if st > 1:
            frame = np.ascontiguousarray(frame[::st, ::st])
        small = cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA)
        if getattr(self.cfg, "obs_gray", True):
            # v13: 1 канал вместо 3. При frame_stack=4 вход CNN становится
            # 4 канала вместо 12 — первый свёрточный слой считает втрое
            # меньше, а цвет для навигации почти бесполезен (цель ищет YOLO).
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)[:, :, None]
        return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    def _center_crop(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cs   = min(self.cfg.crop_size, h, w)
        # Важно: crop должен быть центрирован вокруг реального прицела.
        # Если сдвинуть crop вверх, YOLO-центр и прицел перестают совпадать,
        # и aim-assist начинает систематически тянуть камеру в небо/пол.
        base_y = (h - cs) // 2
        base_x = (w - cs) // 2
        y = int(np.clip(base_y + self.cfg.crop_y_shift, 0, max(0, h - cs)))
        x = int(np.clip(base_x, 0, max(0, w - cs)))
        return frame[y:y+cs, x:x+cs]

    def _is_loading_screen(self, full: np.ndarray) -> bool:
        """Чёрный/почти чёрный экран между картами — просто ждём, не кликаем."""
        return float(full.mean()) < self.cfg.loading_black_mean

    def _respawn_locate(self, full: np.ndarray):
        """Панель "нажмите ЛКМ для возрождения" + её центр в координатах экрана.

        v14, ДВА исправления:
        1) Троттлинг был `self.step_counter % 3`, но step_counter НЕ растёт
           внутри reset(). В цикле ожидания метод навсегда возвращал старое
           значение кэша, и панель возрождения не находилась вообще.
        2) Возвращаем координаты панели, чтобы кликать по ней, а не в центр
           экрана наугад.
        """
        if self.respawn_tpl is None:
            self._last_rs_vis = False
            return False, 0, 0
        _now = time.monotonic()
        if _now - self._rs_t < float(getattr(self.cfg, "ui_check_interval", 0.12)):
            return self._last_rs_vis, self._last_rs_xy[0], self._last_rs_xy[1]
        self._rs_t = _now
        h, w  = full.shape[:2]
        y0, x0 = int(h * 0.10), int(w * 0.20)
        roi   = full[y0:int(h*0.60), x0:int(w*0.80)]
        score, loc = self._match_mask(self._white_mask(roi), self.respawn_tpl)
        th_, tw_ = self.respawn_tpl.shape[:2]
        self._last_rs_vis = score >= self._rs_thresh
        self._last_rs_xy  = (self.x1 + x0 + int(loc[0]) + tw_ // 2,
                             self.y1 + y0 + int(loc[1]) + th_ // 2)
        return self._last_rs_vis, self._last_rs_xy[0], self._last_rs_xy[1]

    def _respawn_visible(self, full: np.ndarray) -> bool:
        vis, _x, _y = self._respawn_locate(full)
        return vis

    def _team_select_visible(self, full: np.ndarray) -> bool:
        # ═══ ЭТО И БЫЛА ПРИЧИНА "АВТО нажимается только один раз" ═══
        # Здесь стоял троттлинг `self.step_counter % 3 != 0`, а step_counter
        # растёт ТОЛЬКО в step(). Цикл ожидания в reset() шага не делает,
        # поэтому после конца матча метод бесконечно возвращал устаревший
        # кэш (обычно False) и меню выбора команды никогда не находилось.
        # Работало лишь на первом старте, когда step_counter == 0.
        if self.team_tpl is None:
            self._last_tm_vis = False
            self._last_tm_center = None
            return False
        _now = time.monotonic()
        if _now - self._tm_t < float(getattr(self.cfg, "ui_check_interval", 0.12)):
            return self._last_tm_vis
        self._tm_t = _now
        h, w = full.shape[:2]
        left, top = int(w * 0.20), int(h * 0.08)
        roi = full[top:int(h * 0.35), left:int(w * 0.80)]
        score, (match_x, match_y) = self._match_mask(self._yellow_mask(roi), self.team_tpl)
        self._last_tm_vis = score >= self._tm_thresh
        self._last_tm_center = None
        if self._last_tm_vis:
            tpl_h, tpl_w = self.team_tpl.shape[:2]
            self._last_tm_center = (left + match_x + tpl_w // 2, top + match_y + tpl_h // 2)
        return self._last_tm_vis

    def _battle_button_visible(self, full: np.ndarray) -> bool:
        """
        Опциональная кнопка "Бой" (отдельная от меню выбора команды).
        Работает только если создан assets/battle_button.png (скриншот
        кнопки, вырезанный точно по её границам). Если файла нет — метод
        всегда возвращает False и ни на что не влияет.
        """
        if self.battle_tpl is None:
            return False
        h, w = full.shape[:2]
        roi   = full[int(h*0.70):int(h*0.98), int(w*0.20):int(w*0.80)]
        score, _ = self._match_mask(self._white_mask(roi), self.battle_tpl)
        self._last_bt_vis = score >= self._bt_thresh
        return self._last_bt_vis

    def _wait_for_focus(self):
        if win32gui.GetForegroundWindow() != self.hwnd:
            self._release_all()
            log.info("[ПАУЗА] Верни фокус в игру; F2 — остановка.")
        while win32gui.GetForegroundWindow() != self.hwnd:
            if not win32gui.IsWindow(self.hwnd) or stop_hotkey_pressed(self.cfg.stop_hotkey_vk):
                raise KeyboardInterrupt
            time.sleep(0.1)

    def _force_focus(self):
        self._wait_for_focus()

    def _release_all(self):
        for key in ("w", "a", "s", "d", "space", self.cfg.respawn_key):
            if self.key_states.get(key):
                pydirectinput.keyUp(key)
                self.key_states[key] = False
        if self.mouse_left:
            pydirectinput.mouseUp(button="left")
            self.mouse_left = False

    def _manage_key(self, key: str, press: bool):
        if self.key_states.get(key) == press:
            return
        self.key_states[key] = press
        (pydirectinput.keyDown if press else pydirectinput.keyUp)(key)

    def _do_click(self, ax: int, ay: int):
        self._force_focus()
        self._release_all()
        win32api.SetCursorPos((ax, ay))
        time.sleep(0.10)
        pydirectinput.mouseDown(button="left")
        time.sleep(0.08)
        pydirectinput.mouseUp(button="left")
        time.sleep(0.10)

    def _click_respawn(self, cx: int = None, cy: int = None):
        # v14: если панель "нажмите ЛКМ" найдена шаблоном — кликаем ровно по
        # ней. Клик в центр экрана остаётся запасным вариантом.
        if cx is None or cy is None:
            cx = self.x1 + self.game_w // 2
            cy = self.y1 + self.game_h // 2
            _tag = "вслепую в центр"
        else:
            _tag = "по панели"
        self._do_click(int(cx), int(cy))
        log.info(f"[СПАВН] Клик {_tag} ({int(cx)}, {int(cy)})")

    def _click_auto(self):
        C  = self.cfg
        x = int(self.game_w * C.auto_x_frac)
        y = int(self.game_h * C.auto_y_frac)
        if self._last_tm_center is not None and C.team_auto_offset_y is not None:
            heading_x, heading_y = self._last_tm_center
            auto_y = heading_y + C.team_auto_offset_y
            if 0 <= auto_y < self.game_h:
                x, y = heading_x, auto_y
        ax, ay = self.x1 + x, self.y1 + y
        self._do_click(ax, ay)
        log.info(f"[МЕНЮ] Нажата кнопка АВТО ({ax}, {ay})")

    def _click_battle(self):
        C  = self.cfg
        ax = self.x1 + int(self.game_w * C.battle_x_frac)
        ay = self.y1 + int(self.game_h * C.battle_y_frac)
        self._do_click(ax, ay)
        log.info(f"[МЕНЮ] Нажата кнопка БОЙ ({ax}, {ay})")

    def _is_reloading(self) -> bool:
        return time.monotonic() < self.reloading_until

    def _start_reload(self):
        """
        Скриптовая перезарядка — НЕ решение нейросети.
        Если OCR выключен, после нажатия R считаем, что магазин снова полный
        через fallback_reload_time. Это позволяет работать без Tesseract.
        """
        if self.mouse_left:
            pydirectinput.mouseUp(button="left")
            self.mouse_left = False

        pydirectinput.keyDown("r")
        time.sleep(0.05)
        pydirectinput.keyUp("r")

        self.reloading_until = time.monotonic() + self.cfg.fallback_reload_time
        _spent = int(getattr(self, "ammo_shots_fired", 0))
        self.ammo_shots_fired = 0
        self.ammo_used_f      = 0.0
        self.ammo_used_f = 0.0
        log.info(
            f"[ПЕРЕЗАРЯДКА] нажал R (израсходовано≈{_spent} "
            f"из {self.cfg.fallback_mag_size})"
        )

        # В режиме без OCR сразу выставляем примерный полный магазин.
        # _is_reloading() всё равно временно запрещает стрельбу до конца таймера.
        if not self.cfg.ammo_ocr_enabled:
            self.ammo_left = self.cfg.fallback_mag_size

        log.debug("[ПЕРЕЗАРЯДКА] Магазин пуст → авто-R")

    def _update_ammo(self, full: np.ndarray):
        """
        Обновляет self.ammo_left.

        1) Если ammo_ocr_enabled=False — Tesseract вообще не вызывается,
           патроны считаются примерно по количеству выстрелов.
        2) Если OCR включён, но Tesseract не установлен/сломался — код
           автоматически переключается на этот же резервный режим.
        """
        C = self.cfg

        # Режим без OCR/Tesseract: примерный счётчик по выстрелам.
        if not C.ammo_ocr_enabled:
            if self._is_reloading():
                return
            self.ammo_left = max(0, C.fallback_mag_size - self.ammo_shots_fired)
            return

        # OCR дорогой, поэтому читаем HUD не каждый шаг.
        if self.step_counter % C.ammo_ocr_every != 0:
            return

        ocr_value = self.ammo_reader.read(full)

        # Если AmmoReader сам отключил OCR после ошибки Tesseract,
        # сразу переходим на fallback, не дожидаясь следующего шага.
        if not C.ammo_ocr_enabled:
            if self._is_reloading():
                return
            self.ammo_left = max(0, C.fallback_mag_size - self.ammo_shots_fired)
            return

        if ocr_value is not None:
            self.ammo_left = ocr_value

            # Подтверждение конца перезарядки: цифра снова не ноль.
            if ocr_value > 0 and self._is_reloading():
                self.reloading_until = 0.0
                log.debug(f"[ПЕРЕЗАРЯДКА] Подтверждена по OCR, патронов: {ocr_value}")

            if ocr_value <= 0:
                self._ammo_ocr_zero_streak += 1
            else:
                self._ammo_ocr_zero_streak = 0
        else:
            # OCR не смог прочитать — резервная оценка по выстрелам.
            if self.ammo_left is None:
                self.ammo_left = max(0, C.fallback_mag_size - self.ammo_shots_fired)

    def _front_wall_visible(self, full: np.ndarray) -> bool:
        """Эвристика близкой стены прямо перед прицелом.

        Когда бот стоит лицом в стену, центральный участок кадра обычно почти
        одноцветный и с малым числом контуров. Motion-ratio это не ловит, если
        камера в это время крутится, поэтому нужен отдельный wall detector.
        """
        C = self.cfg
        if not getattr(C, "wall_escape_enabled", True):
            return False

        h, w = full.shape[:2]
        rw = int(w * getattr(C, "wall_roi_w_frac", 0.24))
        rh = int(h * getattr(C, "wall_roi_h_frac", 0.28))
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - rw // 2)
        x2 = min(w, cx + rw // 2)
        y1 = max(0, cy - rh // 2)
        y2 = min(h, cy + rh // 2)
        roi = full[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        std = float(gray.std())
        edges = cv2.Canny(gray, 50, 120)
        edge_ratio = float(np.count_nonzero(edges) / edges.size)

        blocked = std < getattr(C, "wall_gray_std_max", 22.0) and edge_ratio < getattr(C, "wall_edge_ratio_max", 0.055)
        if blocked and getattr(C, "wall_debug", False):
            log.info(f"[СТЕНА] center std={std:.1f} edge={edge_ratio:.3f}")
        return bool(blocked)

    def _pt(self, name: str):
        """Замер времени участка step(). Пустое имя = старт отсчёта."""
        if not getattr(self.cfg, "profile_enabled", False):
            return
        now = time.perf_counter()
        if not name:
            # Полный цикл между стартами step(): всё, что не попало в секции,
            # съедает PPO (forward + градиенты) и обёртки stable-baselines3.
            _c0 = getattr(self, "_cycle_t0", None)
            if _c0 is not None:
                d0 = getattr(self, "_prof", None)
                if d0 is None:
                    d0 = {}
                    self._prof = d0
                d0["_cycle"] = d0.get("_cycle", 0.0) + (now - _c0) * 1000.0
            self._cycle_t0 = now
        prev = getattr(self, "_pt_last", None)
        if prev is not None and name:
            d = getattr(self, "_prof", None)
            if d is None:
                d = {}
                self._prof = d
            d[name] = d.get(name, 0.0) + (now - prev) * 1000.0
        self._pt_last = now

    def _gray_small(self, full: np.ndarray) -> np.ndarray:
        """Даунскейл кадра 128x72 в оттенках серого, ОДИН раз за step().

        Раньше resize+cvtColor полного кадра 1080p делались дважды за шаг
        (_anti_stuck и _check_freeze) — это до 8 мс/кадр впустую на слабом CPU.
        """
        if (getattr(self, "_sg_key", -1) == self.step_counter
                and getattr(self, "_sg_cache", None) is not None):
            return self._sg_cache
        # cv2.resize(INTER_AREA) по полному кадру стоит ~7 мс на Athlon.
        # Прореживание срезом почти бесплатно, а для детекта движения точности хватает.
        sy = max(1, full.shape[0] // 216)
        sx = max(1, full.shape[1] // 384)
        tiny  = np.ascontiguousarray(full[::sy, ::sx])
        tgray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY)
        gray  = cv2.resize(tgray, (128, 72), interpolation=cv2.INTER_AREA)
        self._sg_key   = self.step_counter
        self._sg_cache = gray
        return gray

    def _anti_stuck(self, full: np.ndarray, forward: bool, front_blocked: bool = False) -> bool:
        now = time.monotonic()
        C   = self.cfg

        gray  = self._gray_small(full)

        if self.motion_gray is None or now - self.spawn_t < 1.0:
            self.motion_gray = gray
            self.stuck_obs_f = 0
            self.front_wall_f = 0
            return False

        diff     = cv2.absdiff(self.motion_gray, gray)
        _, thr   = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        ratio    = np.count_nonzero(thr) / thr.size
        self.motion_gray = gray
        # Сохраняем ratio для награды за реальное движение (см. _get_reward) —
        # раньше r_forward платился просто за зажатый W, даже если бот упирался
        # в стену и физически стоял на месте.
        self.last_motion_ratio = float(ratio)

        # 1) старый детектор: мало изменений картинки при зажатом W.
        if forward and ratio < C.stuck_ratio:
            self.stuck_obs_f += 1
        else:
            self.stuck_obs_f = 0

        # 2) новый детектор: прямо перед прицелом близкая стена. Он важнее,
        # потому что при вращении камеры ratio может быть большим даже без
        # реального движения по карте.
        wall_active = (
            forward
            and getattr(C, "wall_escape_enabled", True)
            and now - self.spawn_t >= getattr(C, "wall_escape_min_age_sec", 1.0)
            and front_blocked
        )
        if wall_active:
            self.front_wall_f += 1
        else:
            self.front_wall_f = 0

        motion_stuck = self.stuck_obs_f >= C.stuck_frames_needed
        wall_stuck = self.front_wall_f >= getattr(C, "wall_escape_frames_needed", 4)
        if not (motion_stuck or wall_stuck):
            return False

        # v3: wall escape should not spam turns. One calm turn + back/strafe is
        # enough; repeated 850px turns made the agent spin in place.
        cooldown = (
            getattr(C, "wall_escape_cooldown_sec", C.stuck_cooldown)
            if wall_stuck else C.stuck_cooldown
        )
        if now - self.last_esc_t < float(cooldown):
            return False

        self.stuck_obs_f = 0
        self.front_wall_f = 0
        self.last_esc_t  = now
        hold_sec = getattr(C, "wall_escape_hold_sec", C.stuck_hold_sec) if wall_stuck else C.stuck_hold_sec
        self.escape_until = now + float(hold_sec)
        self._release_all()

        did_jump = False
        if C.unstuck_jump_enabled:
            pydirectinput.keyDown("space")
            time.sleep(0.10)
            pydirectinput.keyUp("space")
            self.key_states["space"] = False
            did_jump = True

        turn_px = getattr(C, "wall_escape_turn_px", C.stuck_turn_px) if wall_stuck else C.stuck_turn_px
        turn, hold_sec = self.recovery_planner.next(now, int(turn_px), float(hold_sec))
        self.escape_until = time.monotonic() + hold_sec
        self.escape_drive_lr = 2 if turn > 0 else 1
        drive_turn_px = getattr(C, "wall_escape_drive_turn_px", getattr(C, 'escape_drive_turn_px', 140)) if wall_stuck else getattr(C, 'escape_drive_turn_px', 140)
        # v3: drive mouse is intentionally small/zero. Big per-frame mouse
        # movement during escape caused endless spinning when near walls.
        self.escape_drive_mx = int(np.clip((1 if turn > 0 else -1) * int(drive_turn_px), -35, 35))
        direction = 1 if turn > 0 else -1
        chunks    = 6
        base_step, rem = divmod(abs(int(turn)), chunks)
        for i in range(chunks):
            dx = base_step + (1 if i < rem else 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, direction * dx, 0, 0, 0)
            time.sleep(0.02)

        # После ручного поворота сбрасываем память EMA-сглаживания и остаток,
        # иначе следующий шаг PPO наследует "чужой" импульс от этого поворота
        # и мышь резко дёргается (эффект "бешеного блендера").
        self.last_smoothed_mx  = 0.0
        self.last_smoothed_my  = 0.0
        self.mouse_remainder_x = 0.0
        self.mouse_remainder_y = 0.0

        reason = "стена" if wall_stuck else "motion"
        action_name = "Прыжок + поворот" if did_jump else "Поворот"
        # v16: частота застреваний — отдельная метрика (game/stuck_events).
        # Если она растёт — бот возится в углах вместо боя.
        self.stuck_events = int(getattr(self, "stuck_events", 0)) + 1
        log.info(f"[ЗАСТРЕВАНИЕ] {reason}: {action_name} {'→' if turn > 0 else '←'}")
        return True

    def _drive_escape(self) -> bool:
        if time.monotonic() >= self.escape_until:
            return False
        self._manage_key('w', False)
        self._manage_key('s', True)
        self._manage_key('a', self.escape_drive_lr == 1)
        self._manage_key('d', self.escape_drive_lr == 2)
        self._execute_mouse(self.escape_drive_mx, 0)
        return True

    def _check_freeze(self, full: np.ndarray, mx: int, my: int):
        gray  = self._gray_small(full)

        if time.monotonic() - self.spawn_t < 2.0 or self.last_gray is None:
            self.last_gray     = gray
            self.stuck_frz_f   = 0
            self.screen_frozen = False
            return

        diff   = cv2.absdiff(self.last_gray, gray)
        _, thr = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        ratio  = np.count_nonzero(thr) / thr.size

        if (mx != 0 or my != 0) and ratio < 0.002:
            self.stuck_frz_f += 1
        else:
            self.stuck_frz_f = 0

        self.last_gray     = gray
        self.screen_frozen = self.stuck_frz_f >= 45

    def _get_reward(self, dist: Optional[float], action: np.ndarray, killed: bool, jumped: bool) -> float:
        """
        Единственный источник награды за фраг — killed, вычисленный отдельно
        через KillDetector.check() по killfeed. Старая эвристика «цель была
        и пропала во время стрельбы» полностью удалена: она кормила модель
        ложноположительными фрагами из-за мигания рамок YOLO и учила бота
        просто стрелять в воздух рядом с целью (reward hacking).

        r_forward теперь масштабируется реальным motion ratio (см. _anti_stuck),
        а не фактом зажатого W — бот больше не получает награду за упирание
        в стену. r_jump — явный штраф за каждый прыжок, иначе необученная
        политика с ~50% шансом на jump=1 в каждом кадре спамит прыжки почти
        непрерывно (entropy на старте обучения близка к максимуму).
        """
        C = self.cfg

        # Копим вклад каждого слагаемого, чтобы видеть, что реально движет
        # ботом. Плотная награда, начисляемая каждый шаг, при 21 шаге/сек
        # легко перевешивает разовый бонус за фраг — глазами это не поймать,
        # только замером (см. лог [НАГРАДА]).
        rw = getattr(self, '_rw', None)
        if rw is None:
            rw = self._rw = {}

        def add(key: str, val: float) -> float:
            rw[key] = rw.get(key, 0.0) + val
            return val

        reward = add('шаг', C.r_step)
        shoot   = action[4] == 1
        forward = action[0] == 1

        if killed:
            self.total_kills += 1
            self.ep_kills    += 1
            reward           += add('ФРАГ', C.r_kill)
            kd = self.total_kills / max(1, self.total_deaths)
            log.info(f"[🎯 ФРАГ #{self.total_kills}] Раунд: {self.ep_kills} | K/D: {kd:.2f}")

        # v15: два штрафа за один выстрел больше НЕ суммируются.
        #   'мимо'            = цель есть, но далеко (dist > dist_near);
        #   'огонь_без_цели'  = цели нет вообще.
        # Раньше выстрел в пустоту получал оба штрафа сразу, и вместе они
        # давали ~80% всей награды: бот учился только одному — не жать
        # кнопку огня. Стрелять он при этом всё равно не мог: гейт
        # стрельбы блокирует выстрел без цели, патроны не тратились.
        if shoot and dist is not None and dist > C.dist_near:
            reward += add('мимо', C.r_miss_shot * 0.5)

        if shoot and dist is None:
            # v15: если врага видели меньше target_memory_sec назад — это
            # не огонь в пустоту, а добивание по последней позиции
            # (именно так играет человек). Штрафуем только стрельбу
            # в воздух, когда цели не было давно.
            _mem_age = time.monotonic() - float(
                getattr(self, "last_enemy_seen_t", -999.0))
            if _mem_age >= float(getattr(C, "target_memory_sec", 1.5)):
                reward += add('огонь_без_цели',
                              float(getattr(C, 'r_shoot_no_target', -0.005)))

        if dist is not None:
            if self.prev_dist is not None:
                delta = self.prev_dist - dist
                reward += add('сближение', float(np.clip(delta, -20, 20)) * C.r_approach)
            reward += add('близость', (1.0 - min(dist, C.dist_max) / C.dist_max) * C.r_proximity)
            # Условие ровно то же, что у гейта стрельбы в step(): иначе награда
            # капает за нажатия, которые гейт не пропустил (или наоборот).
            if shoot and getattr(self, "last_aimed_ok", dist < C.dist_close):
                reward += add('огонь_вплотную', C.r_shoot_near)
            reward += add('враг_виден', C.r_enemy_seen)
            # v13: платим за то, что цель РЕАЛЬНО в гейте стрельбы.
            # Без этого у политики не было ни одного плотного сигнала
            # "поверни камеру на врага", а фраги за 100k шагов = 0.
            if getattr(self, 'last_aimed_ok', False):
                reward += add('прицел', float(getattr(C, 'r_aimed', 0.0)))
            self.prev_dist = dist
        else:
            self.prev_dist = None

        if forward and time.monotonic() >= self.escape_until and abs(self.last_smoothed_mx) < 2 and abs(self.last_smoothed_my) < 2:
            # Награда пропорциональна тому, насколько сильно реально сдвинулась
            # картинка (0..1), а не бинарному факту "клавиша зажата".
            move_frac = min(self.last_motion_ratio / C.move_full_ratio, 1.0)
            reward += add('вперёд', C.r_forward * move_frac * 3.0)

        if jumped:
            reward += add('прыжок', C.r_jump)

        return reward

    def _execute_mouse(self, target_mx: float, target_my: float):
        """
        Один чистый вызов win32api.mouse_event за шаг — БЕЗ дробления на чанки.
        Раньше здесь был цикл на N микро-шагов с time.sleep(0.002) между ними.
        Проблема: стандартное разрешение системного таймера в Windows — ~15.6мс,
        а не 2мс, поэтому каждый такой sleep() реально блокировал поток заметно
        дольше заявленного и просаживал FPS всего скрипта. Игра получает дельты
        как относительное перемещение (Relative Mouse Look) и сама суммирует
        события между своими кадрами рендера — дробить их на чанки не даёт
        визуальной плавности, только тратит бюджет шага.

        EMA-сглаживание и накопление дробного остатка (mouse_remainder)
        сохранены, чтобы мелкие команды (дельта 2-3px) не терялись при
        округлении int().

        Метод вызывается БЕЗУСЛОВНО на каждом шаге (см. step()), в том числе
        при target_mx == target_my == 0, чтобы EMA-фильтр честно затухал к
        нулю, а не застревал на последнем ненулевом значении.
        """
        C = self.cfg
        sm_x = C.mouse_smooth_alpha * target_mx + (1.0 - C.mouse_smooth_alpha) * self.last_smoothed_mx
        sm_y = C.mouse_smooth_alpha * target_my + (1.0 - C.mouse_smooth_alpha) * self.last_smoothed_my

        self.last_smoothed_mx = sm_x
        self.last_smoothed_my = sm_y

        total_x = sm_x + self.mouse_remainder_x
        total_y = sm_y + self.mouse_remainder_y

        int_x, int_y = int(total_x), int(total_y)
        self.mouse_remainder_x = total_x - int_x
        self.mouse_remainder_y = total_y - int_y

        # Ограничитель наклона: pitch_accum + int_y всегда внутри ±pitch_max_counts.
        lim = int(getattr(C, "pitch_max_counts", 250))
        if lim > 0:
            allowed_y = max(-lim - self.pitch_accum,
                            min(lim - self.pitch_accum, int_y))
            if allowed_y != int_y:
                # Упёрлись в предел: гасим память фильтра, иначе остаток копится
                # и мышь дёрнется рывком, как только предел отпустит.
                self.last_smoothed_my  = 0.0
                self.mouse_remainder_y = 0.0
                int_y = allowed_y

        if int_x != 0 or int_y != 0:
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int_x, int_y, 0, 0)
            # Считаем именно ОТПРАВЛЕННЫЕ counts: EMA и дробный остаток учтены.
            self.pitch_accum += int_y

    def _make_obs(self, full: np.ndarray, dist: Optional[float] = None, target: Optional[Tuple[float, float]] = None) -> Tuple[Dict, np.ndarray]:
        C    = self.cfg
        crop = self._center_crop(full)

        # v14: ПАМЯТЬ О ПРОТИВНИКЕ. Раньше вектор описывал только текущий
        # кадр: враг зашёл за угол — бот мгновенно о нём забыл. Теперь
        # запоминаем последнюю позицию, считаем скорость и экстраполируем,
        # где враг вероятно находится сейчас, плюс отдаём "давность" данных.
        _now_t  = time.monotonic()
        _mem_s  = max(0.2, float(getattr(C, "target_memory_sec", 1.5)))
        if dist is not None and target is not None:
            tx, ty = target
            # FIX: target в координатах полного кадра -> нормируем по центру экрана
            cx = self.game_w / 2.0
            cy = self.game_h / 2.0
            ndx = float(np.clip((tx - cx) / cx, -1.0, 1.0))
            ndy = float(np.clip((ty - cy) / cy, -1.0, 1.0))
            nd  = 1.0 - min(dist, C.dist_max) / C.dist_max
            _pv = getattr(self, "_tgt_prev", None)
            if _pv is None:
                vx = vy = 0.0
            else:
                _dt = max(1e-3, _now_t - _pv[2])
                vx  = float(np.clip((ndx - _pv[0]) / _dt * 0.5, -1.0, 1.0))
                vy  = float(np.clip((ndy - _pv[1]) / _dt * 0.5, -1.0, 1.0))
            self._tgt_prev = (ndx, ndy, _now_t)
            self._tgt_mem  = (nd, ndx, ndy, vx, vy, _now_t)
            ei = np.array([nd, ndx, ndy, vx, vy, 0.0], np.float32)
        else:
            self._tgt_prev = None
            _mem = getattr(self, "_tgt_mem", None)
            if _mem is None:
                ei = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32)
            else:
                _el  = _now_t - _mem[5]
                _age = float(np.clip(_el / _mem_s, 0.0, 1.0))
                if _age >= 1.0:
                    ei = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.float32)
                else:
                    ei = np.array([
                        _mem[0],
                        float(np.clip(_mem[1] + _mem[3] * _el, -1.0, 1.0)),
                        float(np.clip(_mem[2] + _mem[4] * _el, -1.0, 1.0)),
                        _mem[3], _mem[4], _age,
                    ], np.float32)

        return {"frame": self._preprocess(crop), "enemy_info": ei}, crop

    def _movement_guard(self, fb: int, lr: int, mx: int, dist: Optional[float], target: Optional[Tuple[float, float]]) -> Tuple[int, int, int]:
        """Простой rule-based слой, чтобы PPO не сидел на спавне.

        Он НЕ заменяет обучение полностью: PPO всё ещё решает стрелять и
        управляет камерой при цели. Guard только принудительно даёт базовое
        перемещение, пока политика ещё почти случайная.
        """
        C = self.cfg
        if not C.movement_guard_enabled:
            return fb, lr, mx

        now = time.monotonic()
        age = now - self.spawn_t
        forced = False

        # Чередуем A/D, чтобы выйти с линии спавна и не идти строго в стену.
        period = max(0.25, float(C.roam_strafe_period))
        phase = int(now / period)
        roam_lr = 1 if (phase % 2 == 0) else 2
        scan_sign = 1 if (int(now / (period * 2.0)) % 2 == 0) else -1

        if age < C.spawn_push_sec:
            fb = 1
            lr = roam_lr
            forced = True

        elif C.no_target_roam_enabled and target is None:
            fb = 1
            lr = roam_lr
            scan = int(C.no_target_roam_turn_px) * scan_sign
            if abs(mx) < abs(scan):
                mx = scan
            forced = True

        elif C.force_forward_to_target and dist is not None and dist > C.target_forward_until_dist:
            fb = 1
            forced = True

        if forced and self.step_counter % max(1, C.action_log_every * 10) == 0:
            log.debug(f"[MOVE] guard fb={fb} lr={lr} mx={mx} dist={'none' if dist is None else int(dist)}")

        return fb, lr, mx

    def _death_now(self, full: np.ndarray, now: float) -> bool:
        """Смерть с тройной защитой от ложняка.

        1) ГРАЦИЯ ПОСЛЕ СПАВНА. Первые death_spawn_grace_sec секунд HUD
           может быть ещё не отрисован, а "правая часть числа HP пустая" —
           это ровно то, что детектор считает смертью. Отсюда и брались
           смерти #2/#3 через 0-9 секунд после респавна.
        2) ПОДТВЕРЖДЕНИЕ death_confirm_steps шагов подряд.
        3) Если виден шаблон панели респавна — верим сразу, без ожидания.
        """
        C = self.cfg
        if now - self.spawn_t < float(getattr(C, "death_spawn_grace_sec", 3.0)):
            try:
                self.hud.death.hard_reset()
            except Exception:
                pass
            self._death_confirm = 0
            return False

        hud_dead = self.hud.on_death_screen(full, now, self.step_counter)
        panel = self._respawn_visible(full) if self.respawn_tpl is not None else False
        if not (hud_dead or panel):
            self._death_confirm = 0
            return False

        self._death_confirm = getattr(self, "_death_confirm", 0) + 1
        need = 1 if panel else max(1, int(getattr(C, "death_confirm_steps", 3)))
        if self._death_confirm < need:
            return False
        return True

    # ────────────────────────────────────────────────────────────────────────
    #  STEP — работает ТОЛЬКО пока агент жив и играет на карте.
    #  Вся логика ожидания меню/респавна перенесена в reset().
    # ────────────────────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        C = self.cfg
        self._wait_for_focus()

        now = time.monotonic()
        min_interval = C.step_min_interval if C.step_min_interval > 0 else (1.0 / C.target_fps)
        time_passed = now - self.last_step_time
        if time_passed < min_interval:
            time.sleep(min_interval - time_passed)
        self.last_step_time = time.monotonic()

        self.step_counter += 1
        self._pt("")
        full = self._get_frame()
        self._pt("захват")

        # ── МАТЧ ЗАКОНЧИЛСЯ / МЕНЮ ВЫБОРА КОМАНДЫ ─────────────────────────────
        # Важно: меню может появиться прямо во время step(), а не только в reset().
        # Если его не обработать здесь, PPO будет продолжать "играть" в меню до
        # max_episode_steps. Поэтому сразу кликаем АВТО/БОЙ и завершаем эпизод.
        if self._team_select_visible(full):
            self._release_all()
            self._click_auto()

            obs, _ = self._make_obs(full)
            info = {
                "kills": self.total_kills,
                "deaths": self.total_deaths,
                "kd": self.total_kills / max(1, self.total_deaths),
            }
            log.info("[МЕНЮ] Обнаружено меню выбора команды в step() → нажал АВТО")
            return obs, 0.0, True, False, info

        if self._battle_button_visible(full):
            self._release_all()
            self._click_battle()

            obs, _ = self._make_obs(full)
            info = {
                "kills": self.total_kills,
                "deaths": self.total_deaths,
                "kd": self.total_kills / max(1, self.total_deaths),
            }
            log.info("[МЕНЮ] Обнаружена кнопка БОЙ в step() → нажал БОЙ")
            return obs, 0.0, True, False, info

        # ── ЖЁСТКАЯ ГРАНИЦА ЭПИЗОДА: смерть детектирована -> terminated=True ──
        # Раньше смерть обрабатывалась бесконечно внутри step() (агент мог
        # "степать" в состоянии смерти много кадров подряд), а done срабатывал
        # только при зависании картинки. Это ломало математику GAE/Advantage,
        # потому что PPO никогда не видел настоящей терминальной границы на
        # смерти. Теперь как только виден экран смерти — эпизод заканчивается
        # немедленно, а вся логика респавна/ожидания уходит в reset().
        if self._death_now(full, now):
            self._release_all()
            self.dead = True
            self.total_deaths += 1
            self.ep_deaths    += 1
            obs, _ = self._make_obs(full)
            # --- v6: OTMENA LOZHNOGO FRAGA ------------------------------
            _dr = C.r_death
            _veto = float(getattr(C, "kill_death_veto_sec", 0.0))
            _since = time.monotonic() - getattr(self.kill_detector, "_last_kill_t", -999.0)
            if _veto > 0 and 0.0 <= _since <= _veto and self.total_kills > 0:
                self.total_kills -= 1
                self.ep_kills = max(0, self.ep_kills - 1)
                _dr -= C.r_kill
                self.kill_detector._last_kill_t = -999.0
                log.warning(
                    f"[⚠ ОТМЕНА ФРАГА] "
                    f"Смерть через {_since:.1f}с "
                    f"после фрага — "
                    f"это был твой ник "
                    f"как ЖЕРТВЫ. "
                    f"Награда снята.")
            kd = self.total_kills / max(1, self.total_deaths)
            log.info(f"[💀 СМЕРТЬ #{self.total_deaths}] Раунд: {self.ep_kills} фрагов | K/D: {kd:.2f}")
            info = {"kills": self.total_kills, "deaths": self.total_deaths, "kd": kd}
            return obs, _dr, True, False, info

        self._pt("проверки_UI")
        dist, target = self.yolo_worker.get()
        _yi = self.yolo_worker.get_info()
        self.last_target_conf = float(_yi.get("conf", 0.0))
        self.last_target_h = float(_yi.get("box_h", 0.0))
        self.mate_block = bool(_yi.get("mate_block", False))
        if dist is not None:
            self.last_enemy_seen_t = time.monotonic()
        mx = my = 0
        jumped = False

        # Обновляем реальный счётчик патронов по OCR HUD ДО принятия решения
        # о стрельбе — иначе бот мог бы "стрелять" уже пустым магазином на
        # основе устаревших данных.
        self._update_ammo(full)
        self._pt("патроны")

        guard_wants_forward = False
        if C.movement_guard_enabled:
            age = time.monotonic() - self.spawn_t
            guard_wants_forward = (
                age < C.spawn_push_sec or
                (C.no_target_roam_enabled and target is None) or
                (C.force_forward_to_target and dist is not None and dist > C.target_forward_until_dist)
            )
        wants_forward = (action[0] == 1 or guard_wants_forward)
        front_blocked = self._front_wall_visible(full) if wants_forward else False
        escaping = self._anti_stuck(full, forward=wants_forward, front_blocked=front_blocked)
        self._pt("антистак")

        if escaping:
            self._drive_escape()
        elif self._drive_escape():
            pass
        elif time.monotonic() >= self.escape_until:
            fb   = action[0]
            lr   = action[1]
            jump = C.jump_enabled and action[5] == 1

            self._manage_key("w", fb == 1)
            self._manage_key("s", fb == 2)
            self._manage_key("a", lr == 1)
            self._manage_key("d", lr == 2)

            # Кулдаун прыжка — без него необученная политика может выбирать
            # jump=1 каждый кадр (до 15 раз/сек при 15 FPS), и бот прыгает
            # непрерывно. Минимальный интервал имитирует "нельзя прыгнуть,
            # пока не приземлился".
            if (jump and not self.key_states.get("space", False)
                    and now - self.last_jump_t >= C.jump_cooldown):
                pydirectinput.keyDown("space")
                time.sleep(0.02)
                pydirectinput.keyUp("space")
                self.key_states["space"] = True
                self.last_jump_t = now
                jumped = True
            elif not jump:
                self.key_states["space"] = False

            # v14: ассист — костыль для раннего обучения, а не постоянная
            # подпорка. Вес плавно падает aim_assist_weight -> aim_assist_min
            # за aim_assist_decay_steps шагов, и PPO вынужден учиться
            # наводиться сам (в т.ч. вертикалью).
            _w0   = float(np.clip(getattr(C, "aim_assist_weight", 0.85), 0.0, 1.0))
            _wmin = float(np.clip(getattr(C, "aim_assist_min", _w0), 0.0, 1.0))
            _dec  = float(getattr(C, "aim_assist_decay_steps", 0) or 0)
            if _dec > 0 and _wmin < _w0:
                _done  = float(max(int(getattr(self, "global_steps", 0)),
                                   self.step_counter))
                _w_aim = _w0 + (_wmin - _w0) * min(1.0, _done / _dec)
            else:
                _w_aim = _w0
            self.aim_w_now = _w_aim
            pol_mx = self._mx[action[2]]
            pol_my = self._my[action[3]]

            # v13: раньше ассист ЦЕЛИКОМ затирал mx, а my жёстко ставился в 0.
            # Итог: две из шести осей действия были мертвы, PPO не мог
            # научиться наводиться сам и получал шум в градиенте. Теперь
            # ассист и политика смешиваются с весом aim_assist_weight.
            mx = pol_mx
            my = int(round(pol_my * (1.0 - _w_aim)))
            if C.lock_vertical_look:
                my = 0
                self.last_smoothed_my = 0.0
                self.mouse_remainder_y = 0.0

            # YOLO aim assist: pull the crosshair horizontally toward the target.
            # Vertical aim is disabled by default until movement/shooting becomes stable.
            if C.aim_assist and target is not None:
                # FIX: target теперь в координатах ПОЛНОГО кадра, центр = центр экрана
                cx = self.game_w / 2.0
                cy = self.game_h / 2.0
                err_x = target[0] - cx
                err_y = target[1] - cy

                if abs(err_x) < C.aim_deadzone:
                    aim_x = 0
                else:
                    aim_x = int(np.clip(err_x * C.aim_gain_x, -C.aim_max_x, C.aim_max_x))

                if (not C.aim_vertical_enabled) or C.lock_vertical_look:
                    aim_y = 0
                elif abs(err_y) < C.aim_deadzone:
                    aim_y = 0
                else:
                    aim_y = int(np.clip(err_y * C.aim_gain_y, -C.aim_max_y, C.aim_max_y))

                mx = int(round(aim_x * _w_aim + pol_mx * (1.0 - _w_aim)))
                my = int(round(aim_y * _w_aim + pol_my * (1.0 - _w_aim)))

            if C.lock_vertical_look:
                my = 0
                self.last_smoothed_my = 0.0
                self.mouse_remainder_y = 0.0

            # ── Возврат камеры к горизонту ─────────────────────────────────
            # Питч не самоцентрируется. Без этого блока схема самозапирается:
            # ассист задрал камеру на врага сверху -> врага убили/он ушёл ->
            # my=0 -> бот навсегда смотрит в небо, YOLO не видит целей.
            if target is None:
                self.no_target_steps += 1
            else:
                self.no_target_steps = 0

            if (C.aim_vertical_enabled and not C.lock_vertical_look
                    and target is None
                    and self.no_target_steps >= C.pitch_recenter_after
                    and self.pitch_accum != 0):
                back = min(abs(self.pitch_accum), int(C.pitch_recenter_step))
                my = -back if self.pitch_accum > 0 else back
                if C.pitch_debug and self.step_counter % C.action_log_every == 0:
                    log.info(f"[ПИТЧ] возврат к горизонту: наклон={self.pitch_accum:+d} counts, шаг {my:+d}")

            # Не даём ранней PPO-политике сидеть на спавне: если цели нет или
            # мы только появились, guard принудительно включает базовое движение.
            fb, lr, mx = self._movement_guard(fb, lr, mx, dist, target)
            self._manage_key("w", fb == 1)
            self._manage_key("s", fb == 2)
            self._manage_key("a", lr == 1)
            self._manage_key("d", lr == 2)

            # Стрельба разрешена только когда цель реально видна, не слишком
            # далеко, магазин не пуст (по OCR/резервной оценке) и мы н�� в
            # процессе перезарядки. ammo_left может быть None только в самые
            # первые кадры эпизода, пока OCR/резерв ещё не дали ни одного
            # значения — в этом случае разрешаем стрелять (оптимистично).
            # Стреляем только когда цель почти в центре прицела. Это важно,
            # пока YOLO может путать врагов/тиммейтов или видеть дальние цели.
            # Раньше здесь было hypot(err_x, err_y) < dist_close: гейт требовал
            # совпадения по двум осям сразу и физически запрещал стрелять, если
            # враг выше/ниже центра больше чем на dist_close пикселей.
            if target is not None:
                err_x_s = abs(target[0] - self.game_w / 2.0)
                err_y_s = abs(target[1] - self.game_h / 2.0)
                aimed_ok = (err_x_s < C.shoot_gate_x_px and err_y_s < C.shoot_gate_y_px)
            else:
                aimed_ok = False

            # v13: два предохранителя против стрельбы по своим.
            # conf_ok — низкая уверенность YOLO чаще всего значит
            # "игрок, но непонятно чей"; по таким не стреляем вообще.
            # v18: порог зависит от размера бокса. Врага вплотную YOLO часто
            # даёт с conf 0.2-0.4 (в датасете мало близких кадров), и старый
            # единый порог 0.45 запрещал огонь в самой важной ситуации.
            _need_conf = float(getattr(C, "shoot_min_conf", 0.45))
            if float(getattr(self, "last_target_h", 0.0)) >= float(getattr(C, "shoot_big_box_h", 110)):
                _need_conf = min(_need_conf, float(getattr(C, "shoot_min_conf_big", 0.30)))
            conf_ok = self.last_target_conf >= _need_conf
            mate_ok = not (bool(getattr(C, "mate_block_enabled", True)) and self.mate_block)
            if aimed_ok and not (conf_ok and mate_ok):
                self._ff_blocked = getattr(self, '_ff_blocked', 0) + 1
                if self._ff_blocked % 25 == 1:
                    log.info(f"[СВОИ] огонь заблокирован #{self._ff_blocked}: "
                             f"conf={self.last_target_conf:.2f} "
                             f"союзник_под_прицелом={self.mate_block}")
            aimed_ok = aimed_ok and conf_ok and mate_ok
            self.last_aimed_ok = aimed_ok
            want_shoot = action[4] == 1 and dist is not None and aimed_ok
            # v16: разделяем "чего хотела политика" и "что разрешил
            # гейт". Без этих двух чисел shots=1 выглядит как баг, а на
            # самом деле это гейт режет почти всё, потому что YOLO не
            # видит цели.
            if int(action[4]) == 1:
                self.ep_trigger    = int(getattr(self, "ep_trigger", 0)) + 1
                self.total_trigger = int(getattr(self, "total_trigger", 0)) + 1
                if not want_shoot:
                    self.ep_gate    = int(getattr(self, "ep_gate", 0)) + 1
                    self.total_gate = int(getattr(self, "total_gate", 0)) + 1
            ammo_known_empty = self.ammo_left is not None and self.ammo_left <= 0

            if self._is_reloading():
                shoot = False
            elif want_shoot and ammo_known_empty:
                self._start_reload()
                shoot = False
            else:
                shoot = want_shoot

            if self.step_counter % C.action_log_every == 0:
                dist_str = f"{dist:.0f}" if dist is not None else "нет цели"
                ammo_str = str(self.ammo_left) if self.ammo_left is not None else "?"
                log.debug(f"[PPO] fb={fb} lr={lr} jp={int(jump)} m=({mx},{my}) sh={int(shoot)} ammo={ammo_str} dist={dist_str}")

            # Вызывается безусловно каждый шаг — см. подробный комментарий
            # внутри _execute_mouse().
            self._execute_mouse(mx, my)

            if shoot:
                self.last_shot_t = time.monotonic()

            # Расход патронов, пока курок УДЕРЖИВАЕТСЯ. Без этого счётчик
            # видел только нажатия, магазин в игре пустел, R не нажимался,
            # и бот до конца жизни щёлкал пустым оружием.
            _t_now = time.monotonic()
            _t_prev = getattr(self, "_ammo_t_prev", _t_now)
            self._ammo_t_prev = _t_now
            if self.mouse_left:
                _dt = min(0.5, max(0.0, _t_now - _t_prev))
                self.ammo_used_f = getattr(self, "ammo_used_f", 0.0) + \
                    _dt * float(getattr(C, "weapon_rof", 6.0))
                self.ammo_shots_fired = int(self.ammo_used_f)
                # v16: выстрелы во время УДЕРЖАНИЯ. Раньше total_shots
                # рос только на переходе "не жал -> нажал", поэтому
                # секундная очередь из 6 патронов считалась одним выстрелом,
                # и accuracy = kills/shots была числом без смысла.
                # Отдельный аккумулятор, чтобы сброс ammo_used_f при
                # перезарядке не терял статистику.
                self._rounds_f = getattr(self, "_rounds_f", 0.0) + \
                    _dt * float(getattr(C, "weapon_rof", 6.0))
                _whole = int(self._rounds_f)
                if _whole > 0:
                    self._rounds_f -= _whole
                    self.total_shots += _whole
                    self.ep_shots    += _whole

            if shoot and not self.mouse_left:
                pydirectinput.mouseDown(button="left")
                self.mouse_left = True
                self.ep_shots    += 1     # v14: для метрики точности
                self.total_shots += 1
                self.ammo_used_f = getattr(self, "ammo_used_f", 0.0) + 1.0
                self.ammo_shots_fired = int(self.ammo_used_f)
                if self.ammo_left is not None:
                    self.ammo_left = max(0, self.ammo_left - 1)
                if ammo_known_empty:
                    pydirectinput.mouseUp(button="left")
                    self.mouse_left = False
            elif not shoot and self.mouse_left:
                pydirectinput.mouseUp(button="left")
                self.mouse_left = False

            # Цели нет и мы не стреляем — перезарядка ничего не стоит.
            if (getattr(C, "reload_when_idle", True)
                    and not self._is_reloading()
                    and not shoot
                    and dist is None
                    and int(getattr(self, "ammo_shots_fired", 0)) >= 1
                    and (time.monotonic() - self.last_shot_t)
                        >= float(getattr(C, "reload_idle_sec", 1.5))):
                self._start_reload()

            if C.auto_reload and ammo_known_empty and not self._is_reloading() and not shoot:
                self._start_reload()

        self._pt("управление")
        self._check_freeze(full, mx, my)
        self._pt("фриз")

        now_for_kill = time.monotonic()
        recent_shot_ok = (now_for_kill - self.last_shot_t) <= C.kill_recent_shot_sec
        recent_enemy_ok = (now_for_kill - self.last_enemy_seen_t) <= C.kill_recent_enemy_sec
        # Kill-feed висит на экране несколько секунд, проверять его каждый
        # кадр не нужно — template matching по ROI стоит заметного CPU.
        # Оба детектора стоят 0.13 мс — троттлинг больше не нужен.
        killed, _died_hud = self.hud.update(
            full, now_for_kill, self.step_counter,
            recent_shot_ok, recent_enemy_ok)
        self._pt("kill_feed")
        reward = self._get_reward(dist, action, killed, jumped)
        if self.screen_frozen:
            reward = C.r_death

        obs, crop = self._make_obs(full, dist, target)
        self._pt("obs")

        if self.step_counter % max(1, int(getattr(C, "yolo_every", 2))) == 0:
            self.yolo_worker.submit(full)  # FIX: детект по ПОЛНОМУ кадру (весь FOV + масштаб как при обучении), а не по центральному кропу

        # Диагностика датасета YOLO: если враг виден в <10% шагов — это,
        # скорее всего, причина того, что бот "не за что зацепиться" и просто
        # бесцельно бродит/прыгает — проверь качество/размер датасета обучения
        # YOLO и не занижен ли yolo_imgsz для дальних целей.
        # ── Замер реального FPS (сколько шагов в секунду реально тянет ПК) ──
        _fps_every = max(0, int(getattr(C, "fps_report_every", 150)))
        if _fps_every:
            if getattr(self, "_fps_t0", None) is None:
                self._fps_t0 = time.monotonic()
                self._fps_n  = 0
            self._fps_n += 1
            if self._fps_n >= _fps_every:
                _el = time.monotonic() - self._fps_t0
                if _el > 0:
                    _target = C.target_fps
                    _real   = self._fps_n / _el
                    _mark   = "OK" if _real >= _target * 0.9 else "НЕ ВЫТЯГИВАЕТ"
                    log.info(f"[FPS] {_real:.1f} шаг/сек (цель {_target}) — {_mark}")
                    _prof = getattr(self, "_prof", None)
                    if _prof:
                        _nn    = max(1, self._fps_n)
                        _cyc   = _prof.pop("_cycle", 0.0) / _nn
                        _inner = sum(_prof.values()) / _nn
                        _srt   = sorted(_prof.items(), key=lambda kv: -kv[1])
                        _txt   = " | ".join(f"{k}={v/_nn:.1f}" for k, v in _srt)
                        _out   = max(0.0, _cyc - _inner)
                        log.info(f"[ПРОФИЛЬ мс/шаг] цикл={_cyc:.1f} = "
                                 f"step({_inner:.1f}) + PPO/SB3({_out:.1f})")
                        log.info(f"[ПРОФИЛЬ внутри step] {_txt}")
                        self._prof = {}
                    _rw = getattr(self, '_rw', None)
                    if _rw:
                        _sum = sum(_rw.values())
                        _abs = sum(abs(v) for v in _rw.values()) or 1.0
                        _rs  = sorted(_rw.items(), key=lambda kv: -abs(kv[1]))
                        _rt  = ' | '.join(
                            f'{k}={v:+.2f}({abs(v) / _abs * 100:.0f}%)' for k, v in _rs
                        )
                        log.info(
                            f'[НАГРАДА за {_nn} шагов] итого={_sum:+.2f} | {_rt}'
                        )
                        self._rw = {}
                self._fps_t0 = time.monotonic()
                self._fps_n  = 0

        # v14: счётчики эпизода для графиков в TensorBoard
        self.ep_steps += 1
        if dist is not None:
            self.ep_seen += 1

        self._det_hits += 1 if dist is not None else 0
        if self.step_counter % C.det_stats_every == 0:
            pct = 100.0 * self._det_hits / C.det_stats_every
            log.info(f"[YOLO] Цель видна в {pct:.0f}% шагов за последние {C.det_stats_every}")
            # v16: рядом с видимостью сразу видно судьбу выстрелов.
            _tp = 100.0 * int(getattr(self, "ep_trigger", 0)) / max(1, self.ep_steps)
            _gp = 100.0 * int(getattr(self, "ep_gate", 0)) / max(
                1, int(getattr(self, "ep_trigger", 0)))
            log.info(f"[ОГОНЬ] курок нажат в {_tp:.0f}% шагов | гейт срезал "
                     f"{_gp:.0f}% нажатий | патронов выпущено {int(self.total_shots)}")
            self._det_hits = 0

        info = {
            "kills":  self.total_kills,
            "deaths": self.total_deaths,
            "kd":     self.total_kills / max(1, self.total_deaths),
            # v14: метрики качества — по ним видно прогресс задолго до
            # первых фрагов (см. раздел game/ в TensorBoard).
            "shots":       int(self.total_shots),
            "accuracy":    self.total_kills / max(1, self.total_shots),
            "visible_pct": 100.0 * self.ep_seen / max(1, self.ep_steps),
            # v16: метрики, по которым виден прогресс до первых фрагов:
            #   trigger_pct  — в каком % шагов политика жмёт огонь;
            #   gate_pct     — какой % этих нажатий срезал гейт;
            #   stuck_events — сколько раз сработал антистак;
            #   alive_sec    — сколько секунд бот жив в текущем эпизоде.
            "trigger_pct":  100.0 * int(getattr(self, "ep_trigger", 0)) / max(1, self.ep_steps),
            "gate_pct":     100.0 * int(getattr(self, "ep_gate", 0)) / max(
                1, int(getattr(self, "ep_trigger", 0))),
            "stuck_events": int(getattr(self, "stuck_events", 0)),
            "alive_sec":    float(time.monotonic() - self.spawn_t),
            "ff_blocked":  int(getattr(self, "_ff_blocked", 0)),
            "aim_w":       float(getattr(self, "aim_w_now", 0.0)),
        }
        # terminated=False (настоящая смерть уже обработана выше), truncated —
        # зависание картинки (внешняя причина, не "смерть по правилам игры").
        return obs, reward, False, self.screen_frozen, info

    # ────────────────────────────────────────────────────────────────────────
    #  RESET — вся логика ожидания меню выбора команды и экрана смерти теперь
    #  здесь. step() вызывается только тогда, когда агент реально в игре.
    # ────────────────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        C = self.cfg
        self._release_all()
        time.sleep(0.1)

        if not self.is_first_reset:
            kd = self.total_kills / max(1, self.total_deaths)
            log.info(f"[📊 ИТОГ ЭПИЗОДА] Фраги: {self.ep_kills} | K={self.total_kills} D={self.total_deaths} K/D={kd:.2f}")

        self._reset_episode()
        self._force_focus()

        if self.is_first_reset:
            self.is_first_reset = False
            log.info("[СТАРТ] ИИ запущен. Жду игру/меню выбора команды/возрождение...")

        last_team_click_t    = -999.0
        last_respawn_click_t = -999.0
        last_battle_click_t  = -999.0
        # v14: конец матча = игра фризится (спиннер Windows), потом грузит
        # новую карту, и только потом появляется меню выбора команды.
        # Отслеживаем "экран не меняется" и в крайнем случае жмём вслепую.
        last_blind_t         = -999.0
        dead_since           = None
        no_hud_since         = None   # v15: сколько уже нет HUD здоровья
        static_since         = None
        prev_small           = None
        blind_n              = 0
        warn_t               = time.monotonic()
        alive_streak          = 0
        ALIVE_CONFIRM_NEEDED  = 2
        wait_start            = time.monotonic()
        REFOCUS_EVERY         = 10.0
        last_refocus_t        = wait_start

        full = self._get_frame()
        while True:
            self._wait_for_focus()
            full = self._get_frame()
            now  = time.monotonic()

            # v15: F2 обязана работать и здесь. Пока идёт ожидание меню
            # или респавна, step() не вызывается, а значит callback
            # обучения не опрашивается — без этой проверки горячая
            # клавиша молчала бы именно тогда, когда бот застрял.
            if stop_hotkey_pressed(getattr(C, "stop_hotkey_vk", 0x71)):
                raise KeyboardInterrupt("stop hotkey")

            if now - last_refocus_t >= REFOCUS_EVERY:
                self._force_focus()
                last_refocus_t = now

            # v14: насколько кадр изменился с прошлой итерации. Меню, таблица
            # результатов и зависшая игра статичны; живая игра — нет.
            _sm = cv2.cvtColor(cv2.resize(full, (128, 72),
                                          interpolation=cv2.INTER_AREA),
                               cv2.COLOR_BGR2GRAY)
            if prev_small is not None and prev_small.shape == _sm.shape:
                _d = cv2.absdiff(prev_small, _sm)
                _r = float(np.count_nonzero(_d > 25)) / float(_d.size)
                if _r < float(getattr(C, "static_ratio", 0.0015)):
                    if static_since is None:
                        static_since = now
                else:
                    static_since = None
            prev_small = _sm

            # Экран загрузки между картами (почти чёрный кадр) — просто ждём,
            # клики здесь только зря тратят время и могут случайно что-то
            # задеть на UI, который ещё не прогрузился.
            if self._is_loading_screen(full):
                alive_streak = 0
                time.sleep(0.5)
                continue

            if self._team_select_visible(full):
                alive_streak = 0
                if now - last_team_click_t >= C.team_retry_after:
                    self._click_auto()
                    last_team_click_t = now
                time.sleep(0.2)
                continue

            if self._battle_button_visible(full):
                alive_streak = 0
                if now - last_battle_click_t >= C.battle_retry_after:
                    self._click_battle()
                    last_battle_click_t = now
                time.sleep(0.2)
                continue

            _pan_vis, _pan_x, _pan_y = self._respawn_locate(full)
            if _pan_vis or self.hud.on_death_screen(full, now):
                alive_streak = 0
                if dead_since is None:
                    dead_since = now
                if _pan_vis:
                    if now - last_respawn_click_t >= C.respawn_retry_after:
                        self._click_respawn(_pan_x, _pan_y)
                        last_respawn_click_t = now
                else:
                    # v14: надпись "нажмите ЛКМ для возрождения" появляется не
                    # сразу, а через ~5 секунд. Пока её нет, клики уходят в
                    # пустоту и только сбивают фокус окна — спокойно ждём.
                    if (now - dead_since >= float(getattr(C, "respawn_blind_after", 6.0))
                            and now - last_respawn_click_t >= 2.0):
                        self._click_respawn()
                        last_respawn_click_t = now
                time.sleep(0.15)
                continue
            dead_since = None

            # v14: ничего не распознано, но экран стоит на месте — значит это
            # меню/таблица результатов, шаблона которой у нас нет. Пробуем
            # АВТО и центр экрана по очереди, а не висим до бесконечности.
            if (static_since is not None
                    and now - static_since >= float(getattr(C, "blind_auto_after", 20.0))
                    and now - last_blind_t >= float(getattr(C, "blind_auto_retry", 6.0))):
                blind_n += 1
                log.warning(f"[МЕНЮ?] Экран не меняется {now - static_since:.0f}с, "
                            f"ничего не распознано — клик #{blind_n} вслепую.")
                if blind_n % 2 == 1:
                    self._click_auto()
                else:
                    self._click_respawn()
                last_blind_t = now
                alive_streak = 0
                time.sleep(0.3)
                continue

            if now - warn_t >= 30.0:
                warn_t = now
                log.warning(f"[ОЖИДАНИЕ] {now - wait_start:.0f}с не понимаю, что на "
                            f"экране. Проверь assets/team_select.png и "
                            f"assets/respawn_lmb.png (пересними, если UI изменился).")

            # ═══ v15: ГЛАВНЫЙ ФИКС ЭПИЗОДОВ-ПРИЗРАКОВ ═══
            # "Ничего не распознано" — это НЕ "я жив и играю". После
            # смерти HUD исчезает, надпись "нажмите ЛКМ" появляется лишь
            # через ~5 секунд, а гистерезис детектора к этому моменту
            # уже отпустил смерть (в логе "[СМЕРТЬ] панель ушла").
            # Старый код ровно тут выходил из ожидания и запускал
            # эпизод-призрак: бот 3-4 секунды "играл" на экране смерти,
            # антистак крутил его на месте, а появившаяся панель
            # возрождения засчитывалась как НОВАЯ смерть. Отсюда были
            # смерти #2 и #4 сразу после респавна и ep_len_mean 392.
            # Теперь выходим только по положительному признаку жизни:
            # в HUD снова видна правая часть числа HP (~23% белого).
            _hud_alive = True
            if bool(getattr(C, "alive_hud_confirm", True)):
                try:
                    _right = self.hud.death.signals(full)[0]
                    _hud_alive = _right >= float(
                        getattr(C, "death_hud_right_max", 1.5))
                except Exception:
                    _hud_alive = True   # нет доступа к HUD — не блокируем

            if _hud_alive:
                no_hud_since = None
                alive_streak += 1
            else:
                alive_streak = 0
                if no_hud_since is None:
                    no_hud_since = now
                # HUD нет и шаблон панели не сработал — почти наверняка
                # это экран смерти. Терпеливо ждём, а через
                # respawn_blind_after кликаем вслепую.
                if (now - no_hud_since >= float(getattr(C, "respawn_blind_after", 6.0))
                        and now - last_respawn_click_t >= 2.0):
                    self._click_respawn()
                    last_respawn_click_t = now

            if alive_streak >= ALIVE_CONFIRM_NEEDED:
                break

            # Страховка от вечного ожидания: если ROI здоровья не
            # совпадает с UI этой карты, лучше начать эпизод, чем висеть.
            if now - wait_start >= float(getattr(C, "alive_hud_timeout", 25.0)):
                log.warning("[ОЖИДАНИЕ] HUD здоровья не найден за %.0fс — "
                            "начинаю эпизод без подтверждения. Если это "
                            "повторяется, проверь death_hud_right_roi."
                            % (now - wait_start))
                break
            time.sleep(0.1)

        self.dead      = False
        self.spawn_t   = time.monotonic()
        self.last_gray = None
        self.motion_gray = None
        self._death_confirm = 0
        # v13: гистерезис смерти обнуляем ТОЛЬКО здесь, после
        # подтверждённого спавна. Двойной счёт смертей теперь невозможен:
        # детектор требует min_hud_alive кадров живого HUD.
        try:
            self.hud.death.hard_reset()
        except Exception:
            pass
        log.info("[СПАВН] ✅ В игре, эпизод начат.")
        # patch10: берём в руки огнестрельное оружие. Бита/блок в руках — это
        # и невозможность убить, и главный источник ложных детекций YOLO.
        _wc = getattr(self, "cfg", None) or getattr(self, "_cfg", None)
        if _wc is not None and bool(getattr(_wc, "weapon_select_on_spawn", True)):
            _wk = str(getattr(_wc, "weapon_slot_key", "3"))
            try:
                pydirectinput.keyDown(_wk)
                time.sleep(0.03)
                pydirectinput.keyUp(_wk)
                log.info(f"[ОРУЖИЕ] выбран слот {_wk}")
            except Exception as _we:
                log.warning(f"[ОРУЖИЕ] не удалось выбрать слот: {_we}")

        obs, crop = self._make_obs(full)
        self.yolo_worker.submit(full)  # FIX: детект по полному кадру
        return obs, {}

    def close(self):
        self._release_all()
        self.yolo_worker.stop()
        self.camera.stop()
        try:
            self.camera.release()
        except AttributeError:
            pass
        log.info("[СИСТЕМА] Окружение закрыто.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ОБУЧЕНИЕ                                                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def train(cfg: Config):
    device = cfg.ppo_device if (cfg.ppo_device == "cpu" or th.cuda.is_available()) else "cpu"

    # Гибрид: сбор опыта (forward batch=1) на CPU, градиенты на GPU.
    use_hybrid = cfg.ppo_device != "cpu" and bool(getattr(cfg, "ppo_hybrid", True)) and th.cuda.is_available()
    PPOClass   = HybridPPO if use_hybrid else PPO
    if use_hybrid:
        device = HybridPPO.rollout_device
        log.info("[PPO] ГИБРИД: опыт → CPU, градиенты → GPU "
                 "(строка SB3 'Using cpu device' — так и задумано)")
    else:
        log.info(f"[PPO] Устройство: {device.upper()}")

    raw_env = Blockade3DEnv(cfg)
    limited_env = TimeLimit(raw_env, max_episode_steps=cfg.max_episode_steps)
    env     = Monitor(limited_env)
    env     = DummyVecEnv([lambda: env])
    env     = VecFrameStack(env, n_stack=cfg.frame_stack, channels_order="last")

    hyper = dict(
        learning_rate = cfg.learning_rate,
        n_steps       = cfg.n_steps,
        batch_size    = cfg.batch_size,
        n_epochs      = cfg.n_epochs,
        gamma         = cfg.gamma,
        gae_lambda    = cfg.gae_lambda,
        clip_range    = cfg.clip_range,
        ent_coef      = cfg.ent_coef,
        vf_coef       = cfg.vf_coef,
        max_grad_norm = cfg.max_grad_norm,
    )

    # v14: раньше несовместимая старая модель роняла скрипт с
    # AssertionError "spaces must have the same shape: (84,84,12) != (84,84,4)".
    # Это происходит после любого изменения наблюдения (например перевода
    # кадра в ч/б). Теперь мы это ловим, откладываем старые файлы в архив
    # (не удаляем!) и спокойно начинаем обучение заново.
    base_env = env

    def _archive_old(reason: str) -> None:
        _stamp = time.strftime("%Y%m%d_%H%M%S")
        for _p in (cfg.save_path, cfg.norm_path):
            if os.path.exists(_p):
                _dst = "%s.old_%s" % (_p, _stamp)
                try:
                    os.rename(_p, _dst)
                    log.warning(f"[СБРОС] {_p} → {_dst}")
                except OSError as _oe:
                    log.error(f"[СБРОС] не смог переименовать {_p}: {_oe}")
        log.warning(f"[СБРОС] Причина: {reason}")
        log.warning("[СБРОС] Старая модель училась с другим наблюдением/наградой — "
                    "дообучать её поверх новой бессмысленно. Начинаю с нуля.")

    def _make_fresh(_venv):
        os.makedirs("ppo_models", exist_ok=True)
        _e = VecNormalize(_venv, norm_obs=False, norm_reward=True, clip_reward=10.0)
        # v13: сеть меньше = дешевле и forward на CPU, и backward на GPU.
        pk = dict(
            features_extractor_kwargs=dict(
                cnn_output_dim=int(getattr(cfg, "ppo_features_dim", 128))),
            net_arch=dict(pi=[int(getattr(cfg, "ppo_net_arch", 64))],
                          vf=[int(getattr(cfg, "ppo_net_arch", 64))]),
        )
        _m = PPOClass(
            "MultiInputPolicy", _e, verbose=1, device=device,
            tensorboard_log=cfg.tb_log, policy_kwargs=pk, **hyper,
        )
        return _e, _m

    model = None
    if os.path.exists(cfg.save_path):
        log.info(f"[🧠] Найдена модель: {cfg.save_path} — пробую продолжить обучение.")
        try:
            if os.path.exists(cfg.norm_path):
                env = VecNormalize.load(cfg.norm_path, base_env)
                env.training   = True
                env.norm_reward = True
            else:
                env = VecNormalize(base_env, norm_obs=False, norm_reward=True,
                                   clip_reward=10.0)
            model = PPOClass.load(
                cfg.save_path, env=env, device=device,
                custom_objects=hyper, tensorboard_log=cfg.tb_log,
            )
            log.info("[🧠] Модель подхвачена, обучение продолжается.")
        except (AssertionError, ValueError, RuntimeError, KeyError,
                TypeError, EOFError) as _le:
            log.warning(f"[НЕСОВМЕСТИМО] {type(_le).__name__}: {_le}")
            _archive_old(str(_le) or type(_le).__name__)
            env   = base_env
            model = None

    if model is None:
        log.info("[СТАРТ] Создаю новую PPO-модель.")
        env, model = _make_fresh(base_env)

    # После load() модель могла остаться на чужом устройстве — фиксируем явно.
    if use_hybrid:
        model._switch(HybridPPO.rollout_device)

    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=cfg.checkpoint_freq,
            save_path=cfg.checkpoint_path,
            name_prefix="blockade",
            save_vecnormalize=True,
        ),
        SaveNormCallback(cfg.checkpoint_freq, cfg.norm_path),
        StatsCallback(),
        ReleaseOnUpdateCallback(raw_env),
        StopHotkeyCallback(cfg),
    ])

    log.info(f"[ОБУЧЕНИЕ] Запускаю на {cfg.total_timesteps:,} шагов.")
    log.info("[TIP] tensorboard --logdir tb_logs  — для графиков в браузере")
    log.info(f"[TIP] {getattr(cfg, 'stop_hotkey_name', 'F2')} — мягкая остановка: "
             f"модель и VecNormalize сохранятся штатно (Ctrl+C тоже работает, "
             f"но грубее).")
    try:
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
        )
    except KeyboardInterrupt:
        log.info("[СТОП] Прервано пользователем — сохраняю модель.")
    except Exception:
        log.error("[ОШИБКА] Исключение во время обучения:")
        traceback.print_exc()
        raise
    finally:
        os.makedirs("ppo_models", exist_ok=True)
        try:
            model.save(cfg.save_path)
            env.save(cfg.norm_path)
        finally:
            env.close()
        log.info(f"[✅ OK] Модель → {cfg.save_path}")
        log.info(f"[✅ OK] Норм.   → {cfg.norm_path}")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  РЕЖИМ ИГРЫ (ИНФЕРЕНС без обучения)                                        ║
# ╚════════════════════════════════════════════════════════════════════════════╝
def play(cfg: Config):
    if not os.path.exists(cfg.save_path):
        log.error(f"[ИГРА] Модель не найдена: {cfg.save_path}")
        return

    # В режиме игры градиентов нет, а forward batch=1 на CPU быстрее:
    # 16.5 мс против 28 мс на CUDA, где очередь занимает YOLO.
    device = getattr(cfg, "ppo_play_device", "cpu")
    if device == "cuda" and not th.cuda.is_available():
        device = "cpu"
    log.info(f"[ИГРА] Устройство: {device.upper()}")

    raw_env = Blockade3DEnv(cfg)
    limited_env = TimeLimit(raw_env, max_episode_steps=cfg.max_episode_steps)
    env     = Monitor(limited_env)
    env     = DummyVecEnv([lambda: env])
    env     = VecFrameStack(env, n_stack=cfg.frame_stack, channels_order="last")

    if os.path.exists(cfg.norm_path):
        try:
            env = VecNormalize.load(cfg.norm_path, env)
            env.training    = False
            env.norm_reward = False
            log.info(f"[ИГРА] VecNormalize загружена: {cfg.norm_path}")
        except (AssertionError, ValueError, RuntimeError, KeyError,
                TypeError, EOFError) as _ne:
            log.warning(f"[ИГРА] VecNormalize несовместима ({_ne}) — играю без неё.")

    try:
        model = PPO.load(cfg.save_path, env=env, device=device)
    except (AssertionError, ValueError, RuntimeError, KeyError, TypeError) as _me:
        log.error(f"[ИГРА] Модель несовместима с текущими настройками: {_me}")
        log.error("[ИГРА] Это модель от старой версии наблюдения. "
                  "Обучи новую: python blockade_ai.py --mode train")
        env.close()
        return
    log.info(f"[ИГРА] Модель загружена: {cfg.save_path}")

    log.info(f"[TIP] {getattr(cfg, 'stop_hotkey_name', 'F2')} — выход из режима игры.")
    ep  = 0
    try:
        obs = env.reset()
        while True:
            # v15: мягкий выход по горячей клавише.
            if stop_hotkey_pressed(getattr(cfg, "stop_hotkey_vk", 0x71)):
                log.info(f"[СТОП] {getattr(cfg, 'stop_hotkey_name', 'F2')} "
                         f"нажата — выхожу.")
                break
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if done.any():
                ep += 1
                log.info(f"[ИГРА] Эпизод #{ep} завершён.")
                # DummyVecEnv already resets completed episodes.
    except KeyboardInterrupt:
        log.info("[ИГРА] Остановлено.")
    finally:
        env.close()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  ТОЧКА ВХОДА                                                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝
# ╔══════════════════════════════════════╗
# ║  ПРЕСЕТЫ ЗАПУСКА                                                        ║
# ╚═══════════════════════════════════════╝
# "game"  — боевой сервер: берём config.json как есть, ничего не меняем.
#
# "dummy" — ТРЕНАЖЁР: пустой сервер + второй клиент-мишень (браузер/телефон).
#           На живом сервере враг в кадре 0-4% шагов (visible_pct в логе),
#           поэтому прицелу и стрельбе учиться физически не на чем. Мишень
#           стоит в кадре постоянно -> visible_pct 50-80%, гейт открывается,
#           shots/accuracy наконец начинают расти.
#           ВАЖНО: мишень может быть в ТВОЕЙ команде, поэтому фильтр классов
#           и защита от огня по своим здесь выключены. На живом сервере с этим
#           пресетом запускать НЕЛЬЗЯ — будет огонь по союзникам.
#           Все правки только в памяти, config.json не перезаписывается.
PRESETS = {
    "dummy": {
        # цель — ЛЮБОЙ игрок в кадре (мишень часто оказывается союзником)
        "filter_enemy_class": False,
        "mate_block_enabled": False,
        "teammate_iou_veto": 0.0,
        # порог детекта ниже: в логе у живого игрока было conf=0.43
        "yolo_conf": 0.25,
        "shoot_min_conf": 0.30,
        # окно разрешения выстрела шире, иначе гейт срезает почти всё
        "shoot_gate_x_px": 48,
        "shoot_gate_y_px": 56,
        # мишень видно всегда — штраф за пустой курок не нужен
        "r_shoot_no_target": 0.0,
        "r_miss_shot": -0.01,
        # платим за то, ради чего тренажёр и нужен
        "r_enemy_seen": 0.03,
        "r_aimed": 0.06,
        "r_shoot_near": 0.10,
        # эпизоды короче: мишень не убивает, ждать смерти нечего
        "max_episode_steps": 600,
    },
}

PRESET_NAMES = {
    "game":  "ИГРА (боевой сервер)",
    "dummy": "ТРЕНАЖЁР (мишень)",
}


def _apply_preset(cfg, name: str) -> str:
    """Накладывает пресет на cfg В ПАМЯТИ. config.json остаётся нетронутым."""
    over = PRESETS.get(str(name or "game").lower()) or {}
    for k, v in over.items():
        if not hasattr(cfg, k):
            continue
        old = getattr(cfg, k)
        if old != v:
            setattr(cfg, k, v)
            log.info(f"[ПРЕСЕТ] {k}: {old} → {v}")
    if over:
        log.info("[ПРЕСЕТ] ТРЕНАЖЁР: цель — ЛЮБОЙ игрок в кадре, защита от огня "
                 "по своим ОТКЛЮЧЕНА.")
        log.info("[ПРЕСЕТ] Только пустой сервер со своей мишенью! На живом "
                 "сервере запускай без --preset dummy.")
        log.info("[ПРЕСЕТ] config.json не изменён.")
    return PRESET_NAMES.get(str(name or "game").lower(), str(name))


def main():
    parser = argparse.ArgumentParser(
        description="Blockade 3D AI Agent (PPO + YOLO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",   choices=["train", "play"], default="train",
                        help="train — обучение | play — игра без обучения")
    parser.add_argument("--config", default="config.json",
                        help="Путь к JSON-файлу конфигурации")
    parser.add_argument("--fresh", action="store_true",
                        help="Начать обучение с нуля: старую модель отложить в архив")
    parser.add_argument("--preset", choices=["game", "dummy"], default="game",
                        help="game — боевой сервер | dummy — тренажёр по мишени "
                             "на пустом сервере (config.json не меняется)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    from tools.preflight import check
    missing = check(os.path.abspath(args.config))
    if missing:
        parser.error("; ".join(missing))
    for path in (cfg.save_path, cfg.norm_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Пресет накладывается ПОСЛЕ save: файл конфига остаётся боевым.
    _pname = _apply_preset(cfg, getattr(args, "preset", "game"))

    if getattr(args, 'fresh', False) and args.mode == 'train':
        # Награды изменены — старая политика училась под другую функцию цели,
        # дообучать её поверх новой бессмысленно. Ничего не удаляем: файлы
        # переименовываются с меткой времени, откатиться можно всегда.
        _stamp = time.strftime('%Y%m%d_%H%M%S')
        for _p in (cfg.save_path, cfg.norm_path):
            if os.path.exists(_p):
                _dst = '%s.old_%s' % (_p, _stamp)
                os.rename(_p, _dst)
                log.info(f'[СБРОС] {_p} → {_dst}')
        log.info('[СБРОС] Обучение начнётся с чистого листа.')

    log.info("╔══════════════════════════════════════════════╗")
    log.info(f"║  Режим: {args.mode.upper():<38}║")
    log.info(f"║  Пресет: {_pname:<37}║")
    log.info(f"║  YOLO : {cfg.yolo_device.upper():<38}║")
    log.info(f"║  PPO  : {cfg.ppo_device.upper():<38}║")
    log.info(f"║  CUDA : {'ДОСТУПНА' if th.cuda.is_available() else 'НЕДОСТУПНА':<38}║")
    log.info("╚══════════════════════════════════════════════╝")

    if args.mode == "train":
        train(cfg)
    else:
        play(cfg)


if __name__ == "__main__":
    main()
