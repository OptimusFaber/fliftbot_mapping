# Перенос потолочных маркеров: основной метод V5

V5 — единственный поддерживаемый метод переноса потолочных точек. Все три CLI ниже
используют один геометрический движок: проекция пиксельного центра через калибровку
камеры, интерполяция позы по времени, пересечение луча с плоскостью потолка,
робастный выбор пространственной гипотезы и объединение совпадающих декодирований.

GT никогда не участвует в переносе точек. Он читается только скриптом валидации,
после сохранения результата.

## Общие обязательные параметры

- `--observations` — `observations.json` записи камеры;
- `--trajectory` — `trajectory.json` Cartographer или `*_clouds.json`;
- `--map-dir` — каталог с `map.yaml` Cartographer; для `*_clouds.json` каталог результата;
- `--camera-json` и `--camera-position` — калибровка и положение камеры;
- `--ceiling-z` — высота потолка от пола в метрах;
- `--trajectory-pose-frame lidar` для текущего standalone Cartographer с нулевым mount в `lidar.config`;
- `--trajectory-pose-frame rotation_axis` для карты `*_clouds.json`.

Для траектории DePHAN, экспортированной в tracking frame `lidar`,
используется измеренное крепление `--lidar-x 1.29 --lidar-y -0.32`.
V5 при этом один раз переводит смещение камеры от оси вращения в
смещение камеры от LiDAR.

Текущие installation-angle defaults (`roll=-0.14`, `pitch=0.01`,
`yaw=-1.57` рад) перекалиброваны после ввода измеренных рычагов.
Критерий — сходимость повторных наблюдений одной метки, без GT.
Старые углы были связаны с прежним предполагаемым LiDAR-offset `(1.015, 0.0)` и
не должны использоваться вместе с новым креплением.

Калибровочные `--camera-roll`, `--camera-pitch`, `--camera-yaw` и `--time-offset`
относятся к установке камеры, а не к конкретной сцене. Их не следует подбирать по GT.

## 1. Только физические точки со сквозными номерами

```bash
python3 scripts/evaluation/transfer_ceiling_points.py \
  --observations /path/to/recording/observations.json \
  --trajectory /path/to/result/trajectory.json \
  --map-dir /path/to/result \
  --camera-json /path/to/camera.json \
  --camera-position /path/to/camera_position.txt \
  --trajectory-pose-frame lidar \
  --ceiling-z 3.03 \
  --out-dir /path/to/result/marker_localization_v5
```

Создаёт `marker_points.json`, `localization_diagnostics.json` и
`localization_points_only_highres.png`.

## 2. Точки с кодами декодера

Команда имеет те же параметры:

```bash
python3 scripts/evaluation/transfer_ceiling_marker_labels.py [общие параметры]
```

Создаёт `localization_marker_codes_highres.png`. Если несколько кодов относятся к
одной физической точке, они записываются в `candidate_marker_ids` и показываются
через `/`; поле `identity_ambiguous` будет `true`.

## 3. Offline-валидация по GT

```bash
python3 scripts/evaluation/validate_ceiling_marker_localization.py \
  [общие параметры] \
  --gt /path/to/ceiling_labels.json
```

Дополнительно создаются `offline_metrics.json` и
`localization_vs_gt_highres.png`. SE(2)-совмещение существует только внутри оценки
и не изменяет `marker_points.json`.

## Пример для карты clouds

```bash
python3 scripts/evaluation/transfer_ceiling_points.py \
  --observations ../ros-data/recordings/real/20260903/114127_dephan_m360_room/observations.json \
  --trajectory ../ros-data/results/real/20260903/nav_video_lidar_run_114127/run_nav_video_lidar_run_114127_clouds.json \
  --map-dir ../ros-data/results/real/20260903/nav_video_lidar_run_114127 \
  --camera-json ../auto_map/camera.json \
  --camera-position ../auto_map/camera_position.txt \
  --trajectory-pose-frame rotation_axis \
  --ceiling-z 3.03 \
  --out-dir ../ros-data/results/real/20260903/nav_video_lidar_run_114127/marker_localization_v5
```

Запускайте команды из каталога `fliftbot_mapping`. Для воспроизводимости итоговый
JSON сохраняет абсолютные пути входов, версию метода, геометрические параметры и
признак `runtime_gt_used: false`.
