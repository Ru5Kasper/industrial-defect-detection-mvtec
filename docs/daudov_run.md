# EDA MVTec AD: запуск (Даудов)

## Установка

Нужны только CPU и четыре пакета (torch не требуется):

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate ; Linux/macOS: source .venv/bin/activate
pip install pillow numpy pandas matplotlib
```

Проверено на Python 3.13.15, Pillow 13.0.0, NumPy 2.5.4, pandas 3.0.6, matplotlib 3.11.2.

## Запуск

Из корня репозитория:

```bash
python scripts/analyze_dataset.py --data-root "<путь к MVTec AD>" --output-dir results/eda
```

`--data-root` — папка, в которой лежат категории (`bottle/`, `cable/`, …). Абсолютный путь в код и конфиги не зашит.

Дополнительные параметры:

- `--tables-dir results` — куда писать сводные таблицы и JSON;
- `--categories ...` — ожидаемые категории (по умолчанию все 15);
- `--pilot bottle cable carpet` — категории для детальных графиков и галерей.

Код выхода: `0` — проблем нет, `1` — найдена критическая проблема (отчёт всё равно записывается), `2` — неверные аргументы. Полный прогон — около 2 минут на CPU. Датасет только читается.

Тесты (синтетический набор создаётся во временной папке и удаляется):

```bash
python -m unittest discover -s tests -v
```

## Выходные файлы

| Файл | Содержимое |
| --- | --- |
| `results/dataset_summary.csv` | одна строка на категорию: `category,train_good,test_good,test_defect,test_total,defect_types,mask_count,min_width,max_width,min_height,max_height`; `defect_types` — число типов дефектов без `good` |
| `results/defect_summary.csv` | `category,defect_type,image_count,mask_count` для каждого типа теста, включая `good` |
| `results/data_quality.json` | статус, итоги, число прочитанных файлов и проверенных пар, `critical_issues`, `warnings`, списки проблем по видам (пустые, если проблем нет), режимы и размеры изображений по категориям, раздел `loader_compatibility`, выбранные для галерей файлы |
| `results/eda/mask_stats.csv` | по каждой маске: путь к изображению и маске, размер, число и доля дефектных пикселей в исходном разрешении, число дефектных пикселей после nearest-resize до 256 × 256 |
| `results/eda/train_test_counts.png` | train/good, test/good и test-дефекты по 15 категориям |
| `results/eda/test_composition_pilot.png` | состав теста по типам для bottle, cable, carpet |
| `results/eda/defect_area_pilot.png` | распределение доли дефектных пикселей по типам (лог. шкала) |
| `results/eda/gallery_{bottle,cable,carpet}.jpg` | исправный объект и первое по имени изображение каждого типа дефекта: изображение, маска, контур маски |

Все пути в таблицах и JSON относительны к `--data-root`. Повторный запуск на тех же данных даёт побайтно те же файлы.

## Ограничения

- Галереи показывают по одному детерминированно выбранному примеру (`000.png`) на тип, а не случайную выборку.
- Детальные графики и галереи строятся только для пилотных категорий; для других — через `--pilot <категории>`.
- Совместимость с загрузчиком `MVTecDataset` проверена без запуска torch: повторяются те же операции Pillow (`convert`, nearest-resize, порог `> 0`).
- Проверяется структура, читаемость и согласованность файлов, но не корректность самой разметки (точность границ масок экспертами MVTec не перепроверялась).
