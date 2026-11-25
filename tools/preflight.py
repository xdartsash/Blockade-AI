"""Read-only readiness check. Does not import torch or send game input."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check(config, root=ROOT):
    cfg = json.loads((root / config).read_text(encoding='utf-8-sig'))
    errors = []
    for key in ('target_fps', 'n_steps', 'batch_size'):
        value = cfg.get(key, {'target_fps': 15, 'n_steps': 256, 'batch_size': 64}[key])
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f'{key}: требуется положительное целое число')
    if not errors and cfg.get('n_steps', 256) % cfg.get('batch_size', 64):
        errors.append('n_steps должен делиться на batch_size')
    paths = [cfg.get('yolo_path', 'best.pt'), cfg.get('kill_template_path', 'assets/aponima_feed.png'), 'assets/respawn_lmb.png', 'assets/team_select.png']
    for name in paths:
        p = root / name
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f'Отсутствует: {name}')
    if Path(paths[0]).name in ('yolov8n.pt', 'yolo26n.pt'):
        errors.append('Use a BLOCKADE-finetuned best.pt for play, not generic pretrained weights')
    return errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.json')
    args = ap.parse_args()
    errors = check(args.config)
    for e in errors:
        print('[НУЖНО] ' + e)
    print('Файлы готовы; качество детекторов проверь в игре.' if not errors else 'Сначала восстанови данные по START_RU.md. Обучение не запущено.')
    return int(bool(errors))

if __name__ == '__main__':
    raise SystemExit(main())
