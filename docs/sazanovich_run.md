# PatchCore: установка и запуск

## Окружение

Рекомендуется Python 3.10 или 3.11 и отдельное виртуальное окружение. Не устанавливайте эти зависимости поверх окружения другого участника.

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-patchcore.txt
```

Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements-patchcore.txt
```

Проверьте доступность CUDA командой `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`. Для CPU задайте `--device cpu`; для автоматического выбора используется `device: auto` в конфигурации.

## Данные

Скачайте MVTec AD с официальной страницы https://www.mvtec.com/research-teaching/datasets/mvtec-ad и распакуйте так, чтобы путь содержал каталоги `bottle`, `cable`, `carpet`, а внутри каждого были `train/good`, `test` и `ground_truth`.

Путь передаётся через `--data-root`; абсолютный путь не сохраняется в Git. Обучающий Dataset в режиме train индексирует только `train/good`. В test-режиме для каждого дефектного изображения проверяется существование соответствующей маски.

## Команды

```bash
python -m src.train_patchcore --help
python -m src.evaluate_patchcore --help
python -m src.train_patchcore --config configs/patchcore.yaml --data-root "D:/datasets/mvtec_ad"
python -m src.evaluate_patchcore --config configs/patchcore.yaml --data-root "D:/datasets/mvtec_ad"
```

Можно явно указать устройство: `--device cpu` или `--device cuda`.

## Реализация

Backbone — ImageNet-pretrained ResNet-18. Используются layer2 и layer3; layer3 приводится к размеру layer2, признаки конкатенируются и агрегируются average pooling 3×3. Память строится из патчей train/good; greedy k-center выбирает coreset размера min(max_bank_size, round(N × coreset_ratio)). Расстояния до memory bank считаются чанками через torch.cdist.

Порог бинарной карты для per-image решений калибруется по 99.5-му процентилю image scores на train/good. Он не подбирается по test. AUROC и AUPRO используют непрерывные anomaly scores. AUPRO вычисляется приближённо по 300 равномерным порогам до FPR=0.30; этот параметр фиксирован конфигурацией/кодом и должен быть указан в отчёте.

Артефакты моделей и manifest находятся в `artifacts/patchcore/`; таблицы и изображения — в `results/patchcore_metrics.csv`, `results/patchcore_per_image.csv` и `results/patchcore_examples/`. Большие веса и датасет не коммить.

## Ограничения текущей версии

До запуска на локальном MVTec не известны фактические метрики, скорость, размер memory bank и визуальное качество. Их нельзя заполнять оценками или синтетическими числами. После первого запуска нужно зафиксировать версии пакетов, проверить повторную загрузку модели, smoke-тест и корректность карт.
