# -*- coding: utf-8 -*-
"""
tools/check_env.py — проверка окружения Blockade AI
====================================================

Проверяет, что все пакеты стоят и что CUDA реально доступна на GTX 1050.
Запуск:
    python -I tools/check_env.py
"""
import importlib
import sys

# (имя модуля для import, человекочитаемое имя, обязательный?)
MODULES = [
    ("numpy", "NumPy", True),
    ("cv2", "OpenCV", True),
    ("torch", "PyTorch", True),
    ("torchvision", "TorchVision", True),
    ("ultralytics", "Ultralytics YOLO", True),
    ("gymnasium", "Gymnasium", True),
    ("stable_baselines3", "Stable-Baselines3", True),
    ("dxcam", "DXCam (захват экрана)", True),
    ("pydirectinput", "pydirectinput (клавиши/мышь)", True),
    ("win32gui", "pywin32", True),
    ("win32api", "pywin32 (api)", True),
    ("mss", "mss (фолбэк захвата)", False),
    ("pytesseract", "pytesseract (OCR, опционально)", False),
    ("tensorboard", "TensorBoard (графики)", False),
    ("tqdm", "tqdm", False),
    ("psutil", "psutil", False),
]


def main():
    print("=" * 64)
    print(f"  Python: {sys.version.split()[0]}  ({sys.executable})")
    print("=" * 64)

    missing_required = []
    for mod, label, required in MODULES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [OK]   {label:<34} {ver}")
        except Exception as e:
            mark = "НЕТ!" if required else "нет"
            print(f"  [{mark:^6}] {label:<34} {type(e).__name__}: {e}")
            if required:
                missing_required.append(label)

    print("-" * 64)

    # ── CUDA / GPU ───────────────────────────────────────────────────────────
    try:
        import torch
    except Exception:
        print("  [GPU]  PyTorch не установлен — проверка CUDA невозможна")
        return 1

    print(f"  [GPU]  torch версия      : {torch.__version__}")
    print(f"  [GPU]  собрана с CUDA    : {torch.version.cuda or 'нет (CPU-сборка)'}")
    ok = torch.cuda.is_available()
    print(f"  [GPU]  torch.cuda.is_available() -> {ok}")
    if ok:
        print(f"  [GPU]  устройство        : {torch.cuda.get_device_name(0)}")
        try:
            cap = torch.cuda.get_device_capability(0)
            print(f"  [GPU]  compute capability: sm_{cap[0]}{cap[1]}"
                  + ("  (Pascal — OK для cu118)" if cap[0] == 6 else ""))
        except Exception:
            pass
        try:
            free, total = torch.cuda.mem_get_info()
            print(f"  [GPU]  VRAM свободно     : {free / 1024**2:.0f} МБ из {total / 1024**2:.0f} МБ")
        except Exception:
            pass
        # Реальный тест: перемножить матрицы на GPU
        try:
            a = torch.randn(256, 256, device="cuda")
            b = a @ a
            torch.cuda.synchronize()
            print(f"  [GPU]  тест вычислений   : OK ({b.shape}, device={b.device})")
        except Exception as e:
            print(f"  [GPU]  тест вычислений   : ОШИБКА {type(e).__name__}: {e}")
            ok = False
    else:
        print("  [GPU]  CUDA НЕДОСТУПНА — YOLO и PPO пойдут на CPU (медленно).")
        print("         Лечение: переустановить torch с cu118:")
        print("           python -m pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu118")
        print("         Затем в config.json оставить yolo_device/ppo_device = \"cuda\".")

    print("-" * 64)

    # ── Локальные модули проекта ─────────────────────────────────────────────
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root)
    for mod in ("detectors",):
        try:
            importlib.import_module(mod)
            print(f"  [OK]   локальный модуль {mod}.py импортируется")
        except Exception as e:
            print(f"  [НЕТ!] локальный модуль {mod}.py: {type(e).__name__}: {e}")
            missing_required.append(f"{mod}.py")

    # blockade_ai.py при импорте открывает лог-файлы и требует тяжёлые пакеты,
    # поэтому проверяем только синтаксис, не импортируя.
    import py_compile
    try:
        py_compile.compile(os.path.join(root, "blockade_ai.py"), doraise=True)
        print("  [OK]   blockade_ai.py компилируется")
    except Exception as e:
        print(f"  [НЕТ!] blockade_ai.py: {e}")
        missing_required.append("blockade_ai.py")

    print("=" * 64)
    if missing_required:
        print(f"  ИТОГ: НЕ ХВАТАЕТ -> {', '.join(missing_required)}")
        print("  Запусти: powershell -ExecutionPolicy Bypass -File install.ps1")
        return 1
    print("  ИТОГ: окружение готово."
          + (" GPU работает — можно обучать." if ok
             else " Но CUDA недоступна (будет медленно)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
