"""Strict YOLO validation and split by recording session. Standard library only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXT = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}

def validate_label(path, count):
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError('нужно ровно 5 чисел')
            cls, x, y, w, h = map(float, parts)
            if not all(math.isfinite(v) for v in (cls, x, y, w, h)):
                raise ValueError('NaN/Inf')
            if cls != int(cls) or not 0 <= cls < count:
                raise ValueError('неверный класс')
            if not (0 < w <= 1 and 0 < h <= 1 and w/2-1e-6 <= x <= 1-w/2+1e-6 and h/2-1e-6 <= y <= 1-h/2+1e-6):
                raise ValueError('рамка вне изображения или нулевого размера')
        except ValueError as e:
            raise ValueError(f'{path}:{number}: {e}') from e

def prepare(raw, images, out, classes, fraction=.2, seed=42):
    if not 0 < fraction < 1:
        raise ValueError('val-split должен быть между 0 и 1')
    if not classes or len(set(classes)) != len(classes):
        raise ValueError('Нужны уникальные имена классов')
    if any((out / name).exists() and any((out/name).rglob('*.*')) for name in ('train','valid')) or (out/'data.yaml').exists():
        raise ValueError('Выходной датасет уже заполнен. Укажи новый --out, старые файлы не удалены.')
    groups, seen, skipped = {}, {}, 0
    for source in dict.fromkeys((raw.resolve(), images.resolve())):
        if not source.is_dir():
            continue
        for image in sorted(source.rglob('*')):
            if image.suffix.lower() not in EXT:
                continue
            relative = image.relative_to(source)
            label = image.with_suffix('.txt')
            if not label.exists():
                label = source.parent/'labels'/relative.with_suffix('.txt')
            if not label.exists():
                skipped += 1
                continue
            validate_label(label, len(classes))
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            label_text = label.read_text(encoding='utf-8-sig').strip()
            if digest in seen:
                if seen[digest] != label_text:
                    raise ValueError(f'У одинакового изображения разные метки: {image}')
                continue
            seen[digest] = label_text
            group = str(source) + '/' + (relative.parts[0] if len(relative.parts)>1 else 'flat')
            groups.setdefault(group, []).append((image, label, digest))
    if len(groups) < 2:
        raise ValueError('Нужны минимум две разные сессии: raw/session_01, raw/session_02. Соседние кадры нельзя честно разделять случайно.')
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = max(1, min(len(keys)-1, round(len(keys)*fraction)))
    valid = set(keys[:n])
    report = {'seed': seed, 'classes': classes, 'skipped_unlabelled': skipped, 'sessions': {}, 'counts': {'train':0,'valid':0}}
    for key in keys:
        split = 'valid' if key in valid else 'train'
        report['sessions'][key] = split
        for image, label, digest in groups[key]:
            for folder in ('images','labels'):
                (out/split/folder).mkdir(parents=True,exist_ok=True)
            shutil.copy2(image, out/split/'images'/(digest+image.suffix.lower()))
            shutil.copy2(label, out/split/'labels'/(digest+'.txt'))
            report['counts'][split] += 1
    yaml = f'path: {json.dumps(out.resolve().as_posix())}\ntrain: train/images\nval: valid/images\nnames: {json.dumps(classes)}\n'
    (out/'data.yaml').write_text(yaml,encoding='utf-8')
    (out/'split_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw',type=Path,default=ROOT/'dataset/raw')
    ap.add_argument('--images',type=Path,default=ROOT/'dataset/images')
    ap.add_argument('--out',type=Path,default=ROOT/'dataset')
    ap.add_argument('--classes',nargs='+',default=['enemy','teammate'])
    ap.add_argument('--val-split',type=float,default=.2)
    ap.add_argument('--seed',type=int,default=42)
    a = ap.parse_args()
    try:
        print(json.dumps(prepare(a.raw,a.images,a.out,a.classes,a.val_split,a.seed),ensure_ascii=False,indent=2))
    except (ValueError,OSError) as e:
        ap.exit(1, str(e)+'\n')

if __name__ == '__main__':
    main()
