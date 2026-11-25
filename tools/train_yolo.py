# -*- coding: utf-8 -*-
"""
tools/train_yolo.py — обучение детектора врагов/союзников под GTX 1050
=======================================================================

Обучает YOLO26n (самая лёгкая) на твоём датасете dataset/data.yaml и кладёт
веса в runs/detect/trainN/weights/best.pt. blockade_ai.py ищет их по списку
Config.yolo_path / yolo_fallbacks — после обучения просто поправь yolo_path в
config.json на свежую папку (или оставь "best.pt" в корне, скрипт сам его
туда скопирует).

Настройки ПОД СЛАБОЕ ЖЕЛЕЗО (GTX 1050 2GB + Athlon X4):
  • model  = yolo26n.pt   — меньше всех, влезает в 2 ГБ VRAM;
  • imgsz  = 320          — как ждёт blockade_ai.py (yolo_imgsz=320), иначе
                           детект в игре будет видеть цели иначе, чем на обучении;
  • batch  = 8            — 16 может не влезть в 2 ГБ; если OOM — ставь 4;
  • device = 0            — GPU. Если CUDA не поднялась: --device cpu (медленно);
  • half   = False        — FP16 на Pascal ЗАМЕДЛЯЕТ, не включаем;
  • workers= 2            — Athlon X4: много воркеров забьёт CPU.

Запуск
------
    python -I tools/train_yolo.py                 # дефолт (yolo26n, 320, 100 эпох)
    python -I tools/train_yolo.py --epochs 150 --batch 4
    python -I tools/train_yolo.py --resume runs/detect/trainN   # дообучить

После обучения
--------------
Скрипт копирует лучший вес в корень как best.pt и печатает путь. Открой
config.json и укажи:
    "yolo_path": "runs/detect/trainN/weights/best.pt"
или просто оставь "best.pt" — он всегда указывает на последнюю обученную.
Проверь качество:  python -I tools/debug_yolo.py dataset/valid/images
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def pick_device(want: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    if want == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    print("[YOLO] CUDA недоступна — обучаю на CPU (это МЕДЛЕННО, часы-дни). "
          "Для GPU поставь torch с cu118 — см. install.ps1 / README.")
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description="Обучение YOLO под GTX 1050")
    ap.add_argument("--data", default=os.path.join(ROOT, "dataset", "data.yaml"))
    ap.add_argument("--model", default="yolo26n.pt",
                    help="базовые веса (yolo26n.pt скачается автоматически)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto | 0 | cpu")
    ap.add_argument("--name", default="train", help="имя прогона (runs/detect/<name>N)")
    ap.add_argument("--resume", default=None, help="папка прогона для дообучения")
    ap.add_argument("--patience", type=int, default=30,
                    help="ранний стоп, если метрика не растёт N эпох")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        print(f"[ОШИБКА] нет {args.data}. Сначала: collect_dataset → разметка → "
              f"prepare_dataset.py")
        sys.exit(1)

    device = args.device if args.device in ("0", "cpu") else pick_device(args.device)
    print(f"[YOLO] устройство: {device} | epochs={args.epochs} imgsz={args.imgsz} "
          f"batch={args.batch}")

    from ultralytics import YOLO

    if args.resume:
        last = os.path.join(args.resume, "weights", "last.pt")
        if not os.path.exists(last):
            print(f"[ОШИБКА] {last} не найден — нечего возобновлять")
            sys.exit(1)
        model = YOLO(last)
        results = model.train(resume=True)
        best = last.replace("last.pt", "best.pt")
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            amp=False,
            cache=False,
            half=False,               # Pascal: FP16 медленнее — не включаем
            name=args.name,
            patience=args.patience,
            project=os.path.join(ROOT, "runs", "detect"),
            exist_ok=False,
            # Аугментации умеренные: сильный mosaic/hsv путает модель на малом
            # датасете и даёт ложные боксы в игре (см. комментарии v18 в config).
            degrees=0.0,
            translate=0.1,
            scale=0.3,
            fliplr=0.5,               # зеркалить можно — симметрия не важна
            flipud=0.0,               # вверх ногами — НЕТ (игроки не ходят так)
            mosaic=0.6,
            mixup=0.0,
            copy_paste=0.0,
            verbose=True,
        )
        save_dir = getattr(results, "save_dir", None)
        best = os.path.join(str(save_dir), "weights", "best.pt") if save_dir \
            else os.path.join(ROOT, "runs", "detect", "train", "weights", "best.pt")

    # Копируем лучший вес в корень как best.pt (его blockade_ai.py берёт фолбэком).
    if os.path.exists(best):
        dst = os.path.join(ROOT, "best.pt")
        shutil.copy2(best, dst)
        print(f"[YOLO] лучшие веса: {best}")
        print(f"[YOLO] скопировано в: {dst}")
        print(f"[YOLO] укажи в config.json: \"yolo_path\": \"{os.path.relpath(best, ROOT)}\"")
        print("[YOLO] проверка качества: python -I tools/debug_yolo.py dataset/valid/images")
    else:
        print(f"[YOLO][ВНИМАНИЕ] не нашёл best.pt по пути {best} — посмотри "
              f"содержимое runs/detect/")


if __name__ == "__main__":
    main()
