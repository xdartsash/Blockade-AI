<div align="center">

# BLOCKADE AI

**Computer Vision · Object Detection · Reinforcement Learning**

Экспериментальный проект искусственного интеллекта на базе YOLO26 и PPO.

[![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-26-7C3AED?style=for-the-badge)](https://docs.ultralytics.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
![PPO](https://img.shields.io/badge/RL-PPO-16A34A?style=for-the-badge)
![Windows](https://img.shields.io/badge/Platform-Windows-0078D4?style=for-the-badge)

[![Stars](https://img.shields.io/github/stars/xdartsash/Blockade-AI?style=flat-square)](https://github.com/xdartsash/Blockade-AI/stargazers)
[![Last commit](https://img.shields.io/github/last-commit/xdartsash/Blockade-AI?style=flat-square)](https://github.com/xdartsash/Blockade-AI/commits/main)
[![Issues](https://img.shields.io/github/issues/xdartsash/Blockade-AI?style=flat-square)](https://github.com/xdartsash/Blockade-AI/issues)
[![Repository size](https://img.shields.io/github/repo-size/xdartsash/Blockade-AI?style=flat-square)](https://github.com/xdartsash/Blockade-AI)

[О проекте](#о-проекте) · [Архитектура](#архитектура) · [Модели](#модели) · [Данные](#данные) · [Документация](#документация) · [English](#english)

</div>

---

## О проекте

**BLOCKADE AI** — экспериментальная исследовательская система компьютерного зрения и обучения с подкреплением. Проект объединяет детекцию объектов, инструменты подготовки изображений, конфигурации экспериментов и агент на основе PPO.

Проект предназначен для исследований в разрешённых тестовых средах. Соблюдайте правила игры и условия использования сторонних данных.

> **Статус:** активная разработка. Исходники и веса опубликованы, но качество моделей, автономность агента и работоспособность сборки после переноса не подтверждены.

## Архитектура

| Компонент | Назначение |
| :--- | :--- |
| Computer Vision | YOLO, OpenCV и обработка изображений |
| Reinforcement Learning | PPO, Gymnasium, Stable-Baselines3 |
| Dataset Pipeline | Инструменты подготовки и проверки разметки |
| Configuration | JSON-профили экспериментов |
| Validation | Тесты и вспомогательные диагностические инструменты |

## Модели

В репозитории находятся три файла весов:

| Файл | Размер |
| :--- | ---: |
| `blockade_player_yolo26n_20261123.pt` | 5.42 MB |
| `blockade_player_yolo26n_roboflow_v2i.pt` | 5.42 MB |
| `yolov8n.pt` | 6.55 MB |

Метрики точности (mAP, precision, recall), результаты независимого тестирования и условия распространения весов требуют отдельной проверки. Не загружайте непроверенные файлы моделей в доверенную среду.

## Данные

Опубликован небольшой пример размеченных данных: **10 изображений и 10 файлов разметки одной сессии** в `dataset/raw/`. Он не является полноценной независимой обучающей или проверочной выборкой.

Дополнительные наборы Roboflow опубликованы в [overlay_review/datasetrobo/](overlay_review/datasetrobo/). Сохранены 6 555 файлов, включая исходные документы атрибуции:

| Набор | Изображения по метаданным | Лицензия по метаданным |
| :--- | ---: | :--- |
| [blockada-build v2](https://universe.roboflow.com/blockade/blockada-build/dataset/2) | 200 | CC BY 4.0 |
| [BLOCKADE v4](https://universe.roboflow.com/blockade-lake2/blockade/dataset/4) | 839 | CC BY 4.0 |
| [kouloui v6](https://universe.roboflow.com/test-jonrj/kouloui/dataset/6) | 2 234 | CC BY 4.0 |

Указанные лицензии взяты из локальных экспортов Roboflow и не проверены независимо. Наборы имеют разные схемы классов; их нельзя считать единым датасетом без дополнительной проверки. Сырые записи и резервные архивы не включены в опубликованную сборку.

## Структура репозитория

```text
Blockade-AI/
├── assets/              # Ресурсы и документация
├── dataset/             # Пример размеченных данных
├── overlay_review/     # Опубликованные наборы Roboflow
├── ppo_models/         # Каталог контрольных точек PPO
├── runs/               # Каталог результатов экспериментов
├── tb_logs/            # Каталог журналов TensorBoard
├── profiles/            # Конфигурации экспериментов
├── tests/               # Тесты и фикстуры
├── tools/               # Вспомогательные инструменты
├── blockade_ai.py
├── detectors.py
├── preview_yolo.py
├── recovery.py
├── config.json
├── requirements.txt
├── constraints.txt
├── START_RU.md
└── README.md
```

## Документация

- [START_RU.md](START_RU.md) — история проекта и требования окружения.
- [OVERLAY_RU.md](OVERLAY_RU.md) — описание экспериментальной разметки и её ограничений.
- [assets/README.md](assets/README.md) — вспомогательные ресурсы.
- [Issues](https://github.com/xdartsash/Blockade-AI/issues) — ошибки и предложения.

Часть исторической документации может отставать от текущего состояния репозитория.

## Направления развития

- Воспроизводимые эксперименты на независимых данных.
- Измерение точности детекции и качества разметки.
- Тестирование компонентов и документирование ограничений.
- Контроль совместимости зависимостей и происхождения данных.

## English

**BLOCKADE AI** is an experimental computer vision and reinforcement learning research project using YOLO and PPO. The repository contains source code, model weights, configuration profiles, test fixtures and a small annotated sample dataset.

Development is ongoing. Model accuracy and end-to-end functionality have not been independently verified. Use only in authorized testing environments.

## License & credits

A project-wide license has not been declared. Third-party libraries, model weights, game assets and datasets remain subject to their respective licenses and terms. Dataset attribution is listed above.

---

<div align="center">

**Developed by [xdartsash](https://github.com/xdartsash)**

⭐ If you find the research useful, consider starring the repository.

</div>
