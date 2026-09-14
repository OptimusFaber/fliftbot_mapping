# Аудит алгоритма `ray_plane_intersection_plot.ipynb`

Дата аудита: 2026-09-14. Исходный notebook не изменялся.

## Краткий вывод

Аналитическая формула коллеги правильная:

```text
t = (H - O_z) / d_z
P_camera = O + t d
P_map_xy = Rz(yaw) P_camera_xy + [x, y]
```

Но notebook не является полноценным алгоритмом построения карты меток. Он
показывает один выбранный кадр, а затем может показать его точки в глобальной
системе. В нём нет автоматической оценки высоты потолка, робастной
триангуляции/выбора track hypothesis, оценки неопределённости, abstention или
контроля соответствия `tracking_frame` и lever arm LiDAR.

Поэтому его хороший сохранённый пример нельзя напрямую сравнивать с V5/V6:
там показаны 5 удачных наблюдений одного кадра, а две другие точки имеют ошибки
по ID около 3–4 м. На текущей записи DePHAN прямой перенос notebook дал хорошую
медианную ошибку, но плохие отдельные track/code подняли p95 примерно до 0.6 м.
После робастного отбора V6 даёт MAE около 5.7 см и p95 около 9.5 см.

## Что фактически делает notebook

### Входы

В ячейке 1 зашиты:

```python
OBSERVATIONS_PATH = Path("nav_video_20260827_193739/observations.json")
PLANE_HEIGHT = 3.13
```

Ожидаемый `observations.json` содержит:

* `data`: список кадров; в каждом кадре `track_id -> {ray, code, ...}`;
* `camera_origin`: начало лучей в метрах;
* `frame_timestamps`: Unix timestamp каждого кадра.

В ячейке 7 выбирается первый `*.log` рядом с observations. Используются
столбцы `t,x,y,dgr`; остальные `nav_*` и `enc_*` игнорируются. В ячейке 9
загружается `../warehouse/labels2coords.json` в сантиметрах и переводится в
метры.

### Ray-plane intersection

`ray_plane_intersections()` берёт сохранённый единичный `ray`, отбрасывает
`|d_z| < 1e-9` и отрицательное пересечение, затем вычисляет точку на плоскости
`z=PLANE_HEIGHT`. Никакой повторной коррекции distortion или intrinsics в
notebook нет: это должно быть сделано upstream при создании `ray`.

### Перевод в глобальную систему

`interpolate_pose()` линейно интерполирует `x,y` между двумя строками move log,
а `dgr` — по кратчайшей дуге. Затем применяется:

```text
P_global = Rz(dgr + HEADING_OFFSET_DEG) P_local + [x, y]
```

В markdown написано, что базовый offset обычно `-90°`, а в исполняемом коде
ячейки 8 сейчас стоит `HEADING_OFFSET_DEG = 0.0`. Это опасная несогласованность:
результат зависит от того, какой offset был установлен перед запуском.

## Воспроизводимость notebook

Точный вход из notebook отсутствует в workspace:

```text
nav_video_20260827_193739/observations.json
```

Также `jupyter`, `nbformat` и `nbclient` не установлены. Поэтому сохранённые
outputs не выдавались за воспроизведённый результат. Код ячеек был прочитан
непосредственно из JSON notebook и его формула воспроизведена отдельным
аудит-скриптом на текущей DePHAN-записи.

Сохранённый output ячейки 11 — кадр 1233, семь наблюдений:

* коды 767, 128, 4, 1375, 1406 имеют ошибки примерно 1–2 см;
* код 167 имеет ошибку 4.039 м;
* код 193 имеет ошибку 3.181 м.

Следовательно, этот пример нельзя интерпретировать как точность алгоритма
«1–2 см»: пять точек удачны, две семантически неверны или имеют плохое
соответствие track/code.

## Эксперимент на текущем DePHAN

Исходные файлы:

* `ros-data/recordings/real/20260908/160617_dephan_m360_room/observations.json`;
* `frame_timestamps.csv` — 957 кадров;
* `fliftbot_move_logger_2026-09-08_12-34-48_Sep-08.log` — 63 543 строки;
* `camera_position.txt` — `112.5 -28.0 127.0` см;
* canonical Cartographer trajectory с `tracking_frame=lidar`;
* offline GT — `ceiling_labels_room_real_gt.json`.

Для адаптации notebook использовались сохранённые rays текущих observations,
camera origin из `camera_position.txt` без дополнительного отражения Y (так
устроен исходный notebook), timestamp interpolation и median по каждому коду.
GT применялся только после построения точек.

| Режим | Кодов | Physical MAE | RMSE | median | p95 | max | ≤10 см |
|---|---:|---:|---:|---:|---:|---:|---:|
| notebook-like, move log, H=3.03 м | 52 (51 без 98) | 11.7 см | 26.5 см | 2.6 см | 60.2 см | 119.6 см | 44/51 |
| notebook-like, move log, H=3.13 м | 52 (51 без 98) | 13.2 см | 26.1 см | 5.7 см | 59.6 см | 116.4 см | 42/51 |
| V6 auto, Cartographer, H≈3.046 м | 46 (45 без 98) | 5.7 см | 6.4 см | 5.4 см | 9.5 см | 9.6 см | 45/45 |

ID-aware для notebook-like H=3.03 м: MAE 24.0 см, RMSE 89.7 см,
median 2.4 см, p95 119.2 см, max 5.15 м, 47/51 в пределах 10 см.
Большая разница между physical и ID-aware подтверждает проблему track/code
идентичности, а не только геометрии.

При подаче в notebook-like формулу Cartographer trajectory напрямую результаты
получились значительно хуже: около 59 см physical MAE и только 3 из 51 точек
в пределах 10 см. Причина — notebook-origin задан в системе rotation axis,
тогда как canonical trajectory имеет `tracking_frame=lidar`; без преобразования
через `lidar.config` lever arm применяется в неправильной системе.

Результаты аудита сохранены отдельно от canonical outputs:

* [audit_colleague_ray_plane.py](../scripts/evaluation/audit_colleague_ray_plane.py)
* [height303_move_log/audit.json](../../ros-data/results/real/20260914/colleague_notebook_audit/dephan_current/height303_move_log/audit.json)
* [height313_move_log/audit.json](../../ros-data/results/real/20260914/colleague_notebook_audit/dephan_current/height313_move_log/audit.json)
* [height303_cartographer_lidar/audit.json](../../ros-data/results/real/20260914/colleague_notebook_audit/dephan_current/height303_cartographer_lidar/audit.json)
* [height313_cartographer_lidar/audit.json](../../ros-data/results/real/20260914/colleague_notebook_audit/dephan_current/height313_cartographer_lidar/audit.json)

Команда воспроизведения, например:

```bash
python3 fliftbot_mapping/scripts/evaluation/audit_colleague_ray_plane.py \
  --plane-height 3.03 \
  --gt ros-data/gt/real_room/real_room/ceiling_labels_room_real_gt.json \
  --out-dir ros-data/results/real/20260914/colleague_notebook_audit/dephan_current/height303_move_log
```

## Почему коллегин вариант мог выглядеть лучше

1. В notebook уже сохранён ray, рассчитанный upstream `CameraModel`, включая
   intrinsics и distortion. В V5/V6 ray строится из pixels, после чего
   применяется явная optical-to-map calibration. Эти две системы нельзя
   смешивать: у текущей записи stored ray и V5 fixed-angle branch имеют разные
   XY-конвенции.
2. Notebook использует move log в системе, для которой был задан его
   `camera_origin`. Это работает только если pose и origin находятся в одной
   rotation-axis/body системе.
3. Notebook проверяет один кадр и не усредняет длинную последовательность с
   переключениями decoder ID. Это может скрывать нестабильность track/code.
4. На старом примере использованы вручную выбранные `H`, camera origin и
   heading convention. Это не автоматическая, переносимая калибровка.

## Что переносим в production

Переносить стоит формулу пересечения, timestamp interpolation и использование
полной camera model. Нельзя переносить hardcoded origin/H/heading или прямое
смешивание frames.

Production-контракт должен быть таким:

```text
T_map_lidar — из trajectory.json и tracking_frame;
T_axis_lidar — из lidar.config;
T_axis_camera — из camera_position.txt;
ray_camera — из camera.json + distortion + согласованной mount orientation;
timestamp — frame timestamp + проверенный time offset;
point — робастное пересечение многих лучей либо общей plane;
```

Высота сцены должна оцениваться по нескольким лучам/маркерам либо передаваться
как runtime-параметр, но не браться из GT. Нужны robust track/code consensus,
hypothesis selection, residual/spread confidence и abstention.

Именно поэтому V6 сейчас оставляется production candidate: он уже учитывает
`tracking_frame=lidar`, `camera_position.txt`, `lidar.config`, auto ceiling и
robust multi-ray fit. Улучшение V7 имеет смысл только как отдельная проверка
согласованной CameraModel-ray ветки и не должно заменять V6 без regression на
DePHAN и независимых Pandar/RPLIDAR observations.

## Что ещё проверить на других лидарах

Сам ray-plane код переносим на Pandar/RPLIDAR, если запись содержит
`observations.json`, timestamps, camera sidecars и trajectory с явным frame
contract. Для доказательства обобщаемости нужны минимум один Pandar и один
RPLIDAR с тем же экспортом marker observations. Нельзя считать наличие только
LiDAR map достаточным: без camera rays и синхронизированной траектории оценка
меток невоспроизводима.

## Уточнение: физическая метка против номера (2026-09-14)

Предыдущая формулировка про ошибки кодов 167 и 193 была некорректной для задачи
поиска физических меток. В сохранённом выводе notebook ячейки 11 эти числа —
только декодированные `code`, а не независимое измерение положения. Пересчёт
центров семи найденных квадратов через one-to-one Hungarian matching даёт:

| decoded code | ближайший физический GT | ошибка центра |
|---:|---:|---:|
| 767 | 767 | 1.1 см |
| 128 | 128 | 1.3 см |
| 4 | 4 | 1.8 см |
| 1375 | 1375 | 2.3 см |
| 1406 | 1406 | 1.9 см |
| **167** | **1019** | **2.7 см** |
| **193** | **983** | **3.5 см** |

Итого для именно физических найденных центров notebook-примера: MAE **2.1 см**,
RMSE **2.2 см**, медиана **1.9 см**, p95 **3.2 см**, максимум **3.5 см**;
7/7 центров попали в 5 см. Ошибки 4.04 м и 3.18 м — это ID-aware ошибка
сравнения кодов 167/193 с GT-точками с теми же номерами; физическими выбросами
их считать нельзя. Однако это не доказывает правильность идентификации номера.

### Буквально по ячейкам notebook

Notebook не берёт один screenshot и не достраивает по `labels2coords` все
остальные точки.

* Ячейка 1 загружает уже подготовленный `observations.json`: `data` содержит
  кадры, `track_id -> ray, code`; `camera_origin` и timestamps также уже
  сохранены upstream.
* Ячейка 2 только находит кадры, где есть rays.
* Ячейка 3 (`ray_plane_intersections`) берёт rays **одного выбранного кадра**,
  пересекает их с `z = PLANE_HEIGHT` и возвращает только видимые в этом кадре
  наблюдения. Последнее присваивание в ячейке 5 — `frame_idx = 1233`.
* Ячейка 7 загружает movement log, а ячейка 8 интерполирует позу для timestamp
  кадра и переводит эти семь локальных точек в global.
* Ячейка 9 загружает `labels2coords.json`.
* Ячейки 10–11 рисуют **все GT-точки для справки**, а для каждого найденного
  `code` проводят пунктир к точке `markers[code]` и печатают ID-aware ошибку.

То есть `labels2coords` не порождает дополнительные prediction points и не
подменяет отсутствующие rays. Оно используется в notebook для визуального
reference и проверки по декодированному коду. В новой оценке ячейка 11 не
используется для физического соответствия: центры сопоставлены независимо от
`code`.

Для сохранённого примера marker 98 среди семи наблюдений отсутствует, поэтому
reference-выравнивание по 98 не применялось. Использована известная из notebook
поза: одна identity proper rotation (сохранённый heading) и никакого scale,
reflection или GT-оптимизированного преобразования.

### Пересчёт полного DePHAN без принудительных совпадений

В старой версии аудита Hungarian назначал каждой prediction некоторый GT даже
при метровом расстоянии. Это маскировало false positives. Новый аудит сначала
делает one-to-one assignment, затем применяет gate и отдельно считает
precision/recall, unmatched predictions и unmatched GT. Основная оценка ниже
использует только translation `pred_98 -> gt_98` и известную ориентацию; GT не
подгоняет rotation.

| H | gate | matches | precision | recall | MAE matched | RMSE | median | p95 | max |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 303 см | 5 см | 38/51 | 74.5% | 46.3% | 1.9 см | 2.3 см | 1.6 см | 3.9 см | 4.8 см |
| 303 см | 10 см | 47/51 | 92.2% | 57.3% | 2.7 см | 3.3 см | 2.0 см | 6.2 см | 6.7 см |
| 313 см | 5 см | 25/51 | 49.0% | 30.5% | 3.0 см | 3.3 см | 3.4 см | 4.6 см | 4.8 см |
| 313 см | 10 см | 44/51 | 86.3% | 53.7% | 4.5 см | 4.9 см | 4.5 см | 7.2 см | 9.4 см |

При H=303 см ещё четыре найденные точки находятся дальше 10 см и поэтому
являются unmatched/false-positive для этого порога; они не исчезают из
отчёта. Всего найдено 51 точка после исключения anchor 98, GT содержит 82
остальные точки. Это объясняет, почему физическая ошибка matched-кластера
не равна полной надёжности детектора.

Новые JSON:

* [saved_example_unlabeled_audit.json](../../ros-data/results/real/20260914/colleague_notebook_audit/saved_example/saved_example_unlabeled_audit.json)
* [height303/audit.json](../../ros-data/results/real/20260914/colleague_notebook_audit_v2/dephan_current/height303_move_log/audit.json)
* [height313/audit.json](../../ros-data/results/real/20260914/colleague_notebook_audit_v2/dephan_current/height313_move_log/audit.json)

Воспроизведение сохранённого примера:

```bash
python3 fliftbot_mapping/scripts/evaluation/recalc_notebook_saved_example.py
```
