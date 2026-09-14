# Pandar/RPLIDAR standalone Cartographer: диагностика 2026-09-11

## Итог

Главная причина плохого результата RPLIDAR была не только в отсутствии времени
точек. В runner был зашит `speed_scale=-1`, а реальная запись RPLIDAR
`20260901/182826` проверена по `tracks_meta.json`: её модель имеет
`vx=+enc_v*cos(enc_dgr)`. Поэтому одометрия шла в обратную сторону. Runner
теперь принимает знак из `lidar.config`/по типу лидара: RPLIDAR = `+1`,
DePHAN/Pandar без override сохраняют старую совместимую конвенцию `-1`.
Для отдельной записи знак можно задать ключом `encoder_speed_sign` или флагом
`--encoder_speed_sign=1`.

Пандаровские NPZ не имеют этой проблемы времени: `t[i]` реальные, монотонные
и различаются внутри scan. Формула Cartographer используется напрямую:

```text
scan_stamp = max_i t[i]
point_time_i = t[i] - scan_stamp <= 0
```

Для RPLIDAR старых HTTP-записей `t[i]` одинаковы. При явном включении timing
runner восстанавливает время из порядка лучей и периода:

```text
T = median(scan_stamp[k+1] - scan_stamp[k])
phase_i = clamp(direction * unwrap(angle_i-angle_0)/(2*pi), 0, 1)
t_i = scan_stamp - T + phase_i*T       # convention=end
```

Это физически обосновано для записи с известными 5 Hz и clockwise-порядком,
но не является доказательством того, что HTTP timestamp означает начало или
конец sweep. Поэтому безопасный default `--synthesize_rplidar_times=false`;
для проверенной записи лучше оказался `end`.

## Артефакты и A/B

Репрезентативная RPLIDAR запись:

`/home/optimus/ros-projects/ros-data/recordings/real/20260901/182826_rplidar_room`

4654 scan, 18226 encoder samples, median scan period 0.19658 s, 4654/4654
scan имеют collapsed `t`. Старый baseline:

`/home/optimus/ros-projects/ros-data/results/real/20260901/cartographer_standalone`

Полные результаты с правильным знаком encoder:

| вариант | constraints | bbox W×H, м | occupied | components | trajectory W×H, м |
|---|---:|---:|---:|---:|---:|
| speed +1, timing off | 5029 | 22.15×22.60 | 6416 | 567 | 5.57×4.45 |
| speed +1, timing start | 4870 | 21.25×22.55 | 6817 | 483 | 5.96×4.57 |
| speed +1, timing end | **5063** | 21.90×22.45 | 6752 | **452** | 5.86×4.54 |
| speed +1, timing auto (=end) | **5063** | 21.90×22.45 | 6752 | **452** | 5.86×4.54 |

Выбранный вариант: **speed +1 + timing end**. Карта и диагностика:

`ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/map.pgm`

`ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/map.png`

`ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/run_stats.json`

`ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/validation.json`

Для сравнения старый wrong-sign прогон сохранён в:

`ros-data/results/real/20260911/rplidar_182826_timing_off/`

Там bbox 23.05×24.30 м, 639 components и trajectory 4.80×5.67 м. Это
объясняет, почему одно только включение синтетического времени давало
нестабильный/хуже выглядящий результат: оно компенсировало не ту одометрию.

Команды воспроизведения:

```bash
fliftbot_mapping/scripts/mapping/run_standalone_cartographer.sh \
  --run_dir ros-data/recordings/real/20260901/182826_rplidar_room \
  --output ros-data/results/real/20260911/rplidar_182826_fixed_speed_end \
  --configuration_dir fliftbot_mapping/config/cartographer \
  --configuration_file fliftbot_2d_offline.lua --max_range=30 \
  --synthesize_rplidar_times=true --rplidar_timestamp_convention=end
python3 fliftbot_mapping/scripts/evaluation/validate_standalone_map.py \
  ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/map.pgm \
  --map-yaml ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/map.yaml \
  --trajectory ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/trajectory.json \
  --json ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/validation.json
```

## Pandar proof and result

Репрезентативная запись с предыдущим результатом:

`/home/optimus/ros-projects/ros-data/recordings/real/20260820/232828_pandar40p_room`

Старый результат:

`/home/optimus/ros-projects/ros-data/results/real/20260820/room_pandar_run_232828`

В первом/среднем/последнем NPZ: 3522/3546/3356 points, span времени около
0.10 s, 340--356 уникальных timestamp, `diff(t)>=0`. Runner не синтезирует и
не заменяет эти значения (`rplidar_timing_mode=not_applicable`,
`rplidar_collapsed_scans=0`).

У старых Pandar-конфигов был ошибочный унаследованный mount
`x=0.59,y=±0.035`. Исправлено на измеренную геометрию верхнего лидара проекта
`x=1.015,y=0,z=1.51` относительно rotation axis. Для реального Pandar также
зафиксирован проверенный по A/B знак encoder `+1`. При `max_range=12` полный
результат с этим знаком:

`ros-data/results/real/20260911/pandar_232828_max12_final/`

Независимый байтовый аудит NPZ сохранён рядом в
`ros-data/results/real/20260911/pandar_232828_max12_final/timing_audit.json`; для
RPLIDAR — в
`ros-data/results/real/20260911/rplidar_182826_fixed_speed_end/timing_audit.json`.

Метрики: 2690 scan, 1222 odometry samples, 1428 constraints, bbox
26.25×22.60 м, 8501 occupied cells, 340 components, trajectory 5.87×5.73 м.
Сравнение со старым знаком −1: bbox 28.15×24.60 м, 489 components;
смена знака уменьшила bbox и число разрывов. Контрольный LiDAR-only прогон
ещё компактнее по разрывам (111 components):
`ros-data/results/real/20260911/pandar_232828_max12_lidar_only/`.
Прогон с `max_range=30` сохранён в
`ros-data/results/real/20260911/pandar_232828_standalone_fixed_mount1015/`;
он даёт 51.55×46.45 м, поэтому для Pandar из-за дальних шумных returns
практически выбран `max_range=12`, а не 30.

Команда Pandar:

```bash
fliftbot_mapping/scripts/mapping/run_standalone_cartographer.sh \
  --run_dir ros-data/recordings/real/20260820/232828_pandar40p_room \
  --output ros-data/results/real/20260911/pandar_232828_max12_final \
  --configuration_dir fliftbot_mapping/config/cartographer \
  --configuration_file fliftbot_2d_offline.lua --max_range=12 \
  --encoder_speed_sign=1
```

## Ограничения

`validate_standalone_map.py` измеряет внутреннюю геометрию карты, но не делает
абсолютное совмещение с планом комнаты. Большой `components` и fan-shaped
returns на PNG остаются индикаторами качества исходной записи/дальномерных
пропусков, а не ошибкой формулы per-point time. Для новых RPLIDAR записей
нужно сохранять timestamp начала/конца sweep и RPM; тогда синтез можно
включать автоматически без неоднозначности.
