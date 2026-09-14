# fliftbot_mapping

Единая platform-independent постобработка записей LiDAR от настоящего и
симуляционного Fliftbot. Репозиторий строит глобальную карту из уже записанного
run-каталога; здесь нет Gazebo-моделей, аппаратных драйверов и управления.

## Допустимые входы траектории

Mapping использует только измерения ведущего колеса `enc_v`, `enc_dgr` и
данные LiDAR. Приращение позы вычисляется велосипедной моделью:

```text
vx = enc_v * cos(enc_dgr)
wz = enc_v * sin(enc_dgr) / wheelbase
```

`enc_dgr` — физический угол рулевого колеса относительно продольной оси.
`enc_v` — знаковая скорость ведущего колеса. Поэтому при заднем ходе с тем же
углом руля знак угловой скорости меняется.

Поля `odom_x`, `odom_y`, `odom_yaw` и `enc_wz`, встречающиеся в старых
симуляционных записях, принудительно удаляются до синхронизации, deskew,
построения траектории и factor-графа. Они не являются входом SLAM.

Имя ROS 2-пакета и Python-модуля пока остаётся `flifltbot` для совместимости.

## Кто использует пакет

- `../fliftbot_sim` записывает симуляционные LiDAR/encoder данные;
- `../fliftbot_real` записывает данные физического робота;
- оба передают одинаковый формат run-каталога в этот mapping pipeline;
- результаты записываются в соседний `../ros-data/results/`.

## Структура

```text
fliftbot_mapping/
├── src/flifltbot/mapping/       построение глобальной карты
├── src/flifltbot/common/        форматы данных, output и offline runtime
├── scripts/mapping/             CLI-обёртки постобработки
├── scripts/evaluation/          оценка построенных карт
├── tests/                       unit-тесты
├── examples/                    архивные эксперименты
└── docs/                        методика и frozen baseline
```

## Сборка

```bash
source /opt/ros/humble/setup.bash
colcon build --base-paths fliftbot_mapping --packages-select flifltbot --symlink-install
source install/setup.bash
```

Build-скрипты `fliftbot_sim` и `fliftbot_real` автоматически собирают эту
зависимость первыми.

## Построение карты

```bash
# Симуляция
./fliftbot_mapping/scripts/mapping/map_build.sh \
  --run-dir ./ros-data/recordings/sim/YYYYMMDD/HHMMSS_pandar40p_warehouse \
  --method improved --source sim --cameras up

# Настоящий робот
./fliftbot_mapping/scripts/mapping/map_build.sh \
  --run-dir ./ros-data/recordings/real/YYYYMMDD/HHMMSS_pandar40p_room \
  --method factor --source real --cameras left,right,up
```

Результат создаётся в соответствующем каталоге
`ros-data/results/{sim,real}/DATE/…`; исходная запись не изменяется.

```bash
./fliftbot_mapping/scripts/mapping/map_check_inputs.sh --source sim
./fliftbot_mapping/scripts/mapping/map_eval_accuracy.sh --run-dir RUN_DIR
```

Доступные методы: `clouds`, `odom`, `improved`, `factor`, `factor_no_loops`,
`odom_fixed_v2`, `submap`. Аргументы
команд можно посмотреть через `--help`.

`improved` — рекомендуемый быстрый baseline; `factor` добавляет pose-graph и
loop closures; `submap` предназначен для длинных складских проходов;
`clouds` и `odom` полезны как контрольные варианты. Для оценки можно передать
любой из этих методов через `map_eval_accuracy`.

### Режимы построения

Все методы работают офлайн на одном и том же run-каталоге и не используют
`/odom` из симулятора как вход SLAM:

| Метод | Данные и назначение |
|---|---|
| `clouds` | только LiDAR, контроль без колёсной одометрии и deskew |
| `odom` | велосипедная модель `enc_v`/`enc_dgr`, без scan constraints |
| `improved` | рекомендуемый быстрый вариант: SE(2)-deskew и синтетическое время лучей |
| `factor` | wheel prior + scan-to-submap ICP + loop closures |
| `factor_no_loops` | то же, но без глобальных loop closures для повторяющихся проходов |
| `odom_fixed_v2` | адаптивный wheel prior и scan-to-submap ICP, экспериментальный режим |
| `submap` | последовательные подкарты для длинных записей |

Проверить конкретный запуск можно без перезаписи записи: команда создаёт JSON с
метаданными и PNG в `ros-data/results/`. Для симуляции нужна ground-truth
одометрия только в качестве оценки; для комнаты реального робота передайте
эталонный `floorplan.json`:

```bash
# Симуляция, если в RUN_DIR есть odom/GT
./fliftbot_mapping/scripts/mapping/map_eval_accuracy.sh \
  --run-dir RUN_DIR --source sim --method improved --cameras up

# Реальная комната (эталон не используется для построения карты)
./fliftbot_mapping/scripts/mapping/map_eval_accuracy.sh \
  --run-dir RUN_DIR --source real --method factor --cameras up \
  --gt-path ./ros-data/gt/real_room/real_room/floorplan.json --align bbox
```

`map_factor_diag.sh` и поле `stats` в результате показывают число принятых и
отброшенных ICP/графовых связей. Сравнивайте как минимум median/p95 ошибки,
покрытие и визуальное совпадение стен; один RMSE с выбросами недостаточен.

### Запуск Cartographer 2D

Это отдельный ROS 2 baseline для живых топиков или ROS bag. NPZ-run из
`ros-data/recordings` сам по себе Cartographer не читает: нужен узел-конвертер,
который публикует `sensor_msgs/LaserScan` (`/scan`),
`nav_msgs/Odometry` (`/wheel_odom`) и TF `base_link -> <lidar_frame>`.
`/wheel_odom` должен быть интегрирован из `enc_v`/`enc_dgr`; `/odom` из Gazebo
остаётся только validation data.

После сборки пакета запустите baseline в одном терминале:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch flifltbot cartographer_mapping.launch.py \
  use_sim_time:=false scan_topic:=/scan odom_topic:=/wheel_odom
```

Для симуляционного bag используйте `use_sim_time:=true`, а во втором терминале
проиграйте запись с `/clock`:

```bash
ros2 bag play /path/to/bag --clock
```

Перед запуском проверьте контракт:

```bash
ros2 topic info /scan
ros2 topic info /wheel_odom
ros2 run tf2_ros tf2_echo base_link lidar_link
```

После завершения можно сохранить опубликованную occupancy map через Nav2:

```bash
ros2 run nav2_map_server map_saver_cli -f /tmp/fliftbot_cartographer_map
```

Конфигурация находится в `config/cartographer/fliftbot_2d.lua`; launch-файл
устанавливается вместе с пакетом и по умолчанию использует именно её. Для
изменения имени LiDAR или источника одометрии передайте `scan_topic:=...` и
`odom_topic:=...`; TF всё равно должен быть доступен до прихода первого scan.

Для надёжной инкрементальной записи rosbag2, повторной офлайн-обработки,
сохранения `.pbstream`/occupancy map и экспорта оптимизированной траектории в
JSON используйте [инструкцию офлайн-Cartographer](docs/cartographer_offline.md).

Автономная локализация без ROS напрямую из UDP Pandar/DePHAN описана в
[`scripts/localization/README.md`](scripts/localization/README.md).

Для длинной симуляционной записи склада (от 8000 кадров) `factor`
автоматически включает устойчивый профиль `long_run_wheel_manhattan_lidar`:
перемещение берётся из велосипедной модели `enc_v/enc_dgr`, ориентация
уточняется по длинным ортогональным линиям текущего LiDAR-скана, а неоднозначные
loop closure между одинаковыми проходами отключаются.

## Разделение ответственности

- Gazebo, модели, миры, bridge, teleop и `use_sim_time=true` — `fliftbot_sim`;
- HTTP/ROS-мост, реальные sensor config и `use_sim_time=false` — `fliftbot_real`;
- deskew, scan matching, pose graph, фильтрация, export и evaluation —
  `fliftbot_mapping`.

Канонические NPZ-записи находятся в `../ros-data/recordings/`, отдельные
rosbag2-входы Cartographer — в `../ros-data/cartographer_bags/`, результаты —
в `../ros-data/results/`. Подробнее:
[`../ros-data/README.md`](../ros-data/README.md).

## Проверка

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=fliftbot_mapping/src \
  /usr/bin/python3 -m pytest -q fliftbot_mapping/tests
```

Описание алгоритмов: [`docs/README.md`](docs/README.md).
