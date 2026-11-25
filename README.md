<div align="center">

# BLOCKADE AI

**Experimental computer vision and reinforcement learning project**

![Python](https://img.shields.io/badge/Python-3.11–3.13-3776AB?style=for-the-badge)
![YOLO](https://img.shields.io/badge/YOLO-26-7C3AED?style=for-the-badge)
![PPO](https://img.shields.io/badge/RL-PPO-16A34A?style=for-the-badge)
![Windows](https://img.shields.io/badge/Platform-Windows-0078D4?style=for-the-badge)

</div>

## О проекте

BLOCKADE AI — экспериментальный проект компьютерного зрения и обучения с подкреплением. Использует YOLO для обнаружения объектов на изображениях и PPO (Stable-Baselines3) для экспериментов с агентом.

Проект находится в стадии разработки. Полностью автономная работа и качество моделей не подтверждены. Используйте только в разрешённых тестовых средах, соблюдая правила игры и права на данные.

## Статус публикации

Исходный код, конфигурации, тесты и инструменты опубликованы. Модели, изображения, датасеты и архивы остаются локальными; репозиторий не является готовой сборкой. Тесты после переноса не запускались.

## Структура

| Компонент | Назначение |
| --- | --- |
| `blockade_ai.py` | Основная логика экспериментального агента |
| `detectors.py` | Модули распознавания |
| `preview_yolo.py` | Предварительный просмотр детекций |
| `tools/` | Сбор, разметка, подготовка данных и инструменты обучения |
| `tests/` | Тесты компонентов |
| `profiles/` | Конфигурационные профили |
| `assets/` | Вспомогательные ресурсы |
| `requirements.txt` | Зависимости Python |
| `constraints.txt` | Ограничения версий зависимостей |

## Технологии

- Python, PyTorch, Ultralytics YOLO, OpenCV.
- Gymnasium и Stable-Baselines3 (PPO).
- Локальные конфигурации и экспериментальные модели.

## Документация

- [START_RU.md](START_RU.md) — описание состояния проекта и окружения.
- [OVERLAY_RU.md](OVERLAY_RU.md) — ограничения экспериментальной разметки.
- [assets/README.md](assets/README.md) — вспомогательные ресурсы.

Историческая документация может отставать от текущего состояния файлов. Точность и производительность текущих моделей в этом обновлении не тестировались.

## Данные и модели

Локальные архивы, сырые игровые записи, датасеты, журналы и результаты обучения не включаются в обычный Git-репозиторий. Перед распространением моделей, скриншотов и ресурсов необходимо проверить права на них и отсутствие личных данных.

## License

A project-wide license has not been declared. Third-party libraries, game assets and datasets retain their respective rights.

<div align="center">

Developed by **xdartsash**

</div>
