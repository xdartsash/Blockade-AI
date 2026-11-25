# -*- coding: utf-8 -*-
"""
tools/debug_yolo.py — проверить обученную YOLO на кадрах/папке
===============================================================

Гоняет веса по картинкам и рисует боксы: зелёный = прошёл игровые фильтры
(размер/форма, как в YOLOWorker), красный = отброшен фильтром, серый = низкая
conf. Помогает ДО обучения RL понять, видит ли модель врагов и не ловит ли
столбы/двери.

Запуск
------
    python -I tools/debug_yolo.py dataset/valid/images
    python -I tools/debug_yolo.py dataset/raw/cap_00042.jpg --weights best.pt
    python -I tools/debug_yolo.py dataset/valid/images --conf 0.25 --save out/

Ключи ровно те, что и в Config blockade_ai.py (yolo_conf, yolo_min_box_w/h,
target_hw_min/max), поэтому «зелёные» боксы здесь = цели, которые бот реально
возьмёт в игре.
"""
import argparse
import os
import sys

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Отладка YOLO на кадрах")
    ap.add_argument("path", help="папка с кадрами или один файл")
    ap.add_argument("--weights", default="best.pt",
                    help="веса YOLO (best.pt или runs/detect/trainN/weights/best.pt)")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cpu", help="cpu | 0 (для скорости — 0, если CUDA)")
    ap.add_argument("--save", default=None, help="папка для кадров с рамками (иначе окно)")
    # игровые фильтры (значения по умолчанию из Config)
    ap.add_argument("--min-w", type=int, default=10)
    ap.add_argument("--min-h", type=int, default=24)
    ap.add_argument("--min-area", type=int, default=250)
    ap.add_argument("--hw-min", type=float, default=1.25)
    ap.add_argument("--hw-max", type=float, default=3.6)
    ap.add_argument("--no-window", action="store_true", help="не показывать окно (только --save)")
    args = ap.parse_args()

    # ищем веса
    weights = args.weights
    if not os.path.isabs(weights):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cand = os.path.join(root, weights)
        weights = cand if os.path.exists(cand) else weights
    if not os.path.exists(weights):
        print(f"[ОШИБКА] веса не найдены: {weights}")
        print("         Обучи модель: python -I tools/train_yolo.py")
        sys.exit(1)

    # собираем файлы
    if os.path.isdir(args.path):
        files = [os.path.join(args.path, f) for f in sorted(os.listdir(args.path))
                 if os.path.splitext(f)[1].lower() in IMG_EXT]
    else:
        files = [args.path]
    if not files:
        print(f"[ОШИБКА] нет картинок в {args.path}")
        sys.exit(1)

    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(weights)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}

    if args.save:
        os.makedirs(args.save, exist_ok=True)

    print(f"[DEBUG] {len(files)} кадров | веса={weights} | conf={args.conf} imgsz={args.imgsz}")
    tot_green = tot_red = tot_low = 0

    for fp in files:
        img = cv2.imread(fp)
        if img is None:
            continue
        res = model.predict(img, verbose=False, conf=args.conf,
                            imgsz=args.imgsz, device=args.device)
        boxes = res[0].boxes
        out = img.copy()
        n_green = n_red = 0
        if boxes is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().int().numpy()
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bw, bh = x2 - x1, y2 - y1
                hw = bh / max(1, bw)
                area = bw * bh
                passed = (bw >= args.min_w and bh >= args.min_h
                          and area >= args.min_area
                          and args.hw_min <= hw <= args.hw_max)
                cname = names.get(int(k), str(int(k)))
                label = f"{cname} {c:.2f} {'OK' if passed else f'h/w{hw:.1f}'}"
                if passed:
                    color = (0, 200, 0)   # зелёный — цель дойдёт до бота
                    n_green += 1
                else:
                    color = (0, 0, 230)   # красный — отброшен фильтром формы/размера
                    n_red += 1
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                cv2.putText(out, label, (x1, max(12, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        tot_green += n_green
        tot_red += n_red
        if n_green == 0 and n_red == 0:
            tot_low += 1

        tag = f"{os.path.basename(fp)}: зелёных={n_green} красных={n_red}"
        print("[DEBUG] " + tag)

        if args.save:
            cv2.imwrite(os.path.join(args.save, os.path.basename(fp)), out)
        elif not args.no_window:
            cv2.imshow("YOLO debug (q=выход, any=далее)", out)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break

    if not args.no_window and not args.save:
        cv2.destroyAllWindows()

    print("─" * 60)
    print(f"[DEBUG] ИТОГО: целей(зелёных)={tot_green} | отброшено(красных)={tot_red} | "
          f"кадров без детекта={tot_low}/{len(files)}")
    if tot_green == 0:
        print("[DEBUG] Модель НИЧЕГО не проводит через фильтры. Проверь: обучена ли она,")
        print("        тот ли imgsz/conf, и есть ли враги вообще на этих кадрах.")
    elif tot_low > len(files) * 0.7:
        print("[DEBUG] >70% кадров без детекта — мало данных с врагами или низкий conf.")
    if args.save:
        print(f"[DEBUG] кадры с рамками: {args.save}")


if __name__ == "__main__":
    main()
