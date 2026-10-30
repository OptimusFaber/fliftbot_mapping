# fliftbot_lidar_nav

LiDAR-локализация Fliftbot по **готовой карте** и передача позы в навигацию.
Объединяет прежние `fliftbot_mapping` и `realtime_mapping`.

| Пакет | Ответственность |
|---|---|
| `fliftbot_sim` | Gazebo, датчики, измеренные энкодеры, запись |
| `fliftbot_map_build` | Офлайн Cartographer, occupancy/PBStream, метки, web/deploy |
| `fliftbot_lidar_nav` | Загрузка deploy-карты, локализация LiDAR, ROS/CSV/ContinuousNav |

Главная инструкция: **[simulation → map build → localization](docs/pipeline.md)**.
Построение новых карт выполняется в `fliftbot_map_build`.
Пакет выдаёт позицию для навигационного контроллера; построение маршрута,
управление движением и объезд препятствий выполняет внешний контроллер
(например, существующий ContinuousNav). Команды ниже сами робота не двигают.

## Установка

Linux, Python 3.10+, ROS 2 Humble для live-режима; offline/instant работают без ROS.
Из корня workspace:

```bash
sudo apt install python3-numpy python3-scipy python3-pil python3-yaml python3-zmq python3-protobuf
export PYTHONNOUSERSITE=1
source /opt/ros/humble/setup.bash
colcon build --base-paths fliftbot_lidar_nav --packages-select fliftbot_lidar_nav --symlink-install
source install/setup.bash
./fliftbot_lidar_nav/nav.sh verify
```

Для live нужен `cartographer_ros`, а на физическом роботе — собранные
`fliftbot_interfaces/msg/MC`. Схема `SensorMeasurement` для ContinuousNav включена в пакет; поддерживается
системный protobuf ROS Humble, внешние Python-модули контроллера для отправителя не нужны.
`requirements.txt` перечисляет Python-зависимости для отдельного окружения;
ROS-скрипты следует запускать системным Python из ROS-окружения.

`nav.sh` работает из любого текущего каталога и использует `.venv/bin/python`,
если такое окружение создано, иначе `/usr/bin/python3`. Интерпретатор можно
задать через `FLIFTBOT_LIDAR_NAV_PYTHON`; он передаётся также ROS subprocess.
Для создания окружения без изменения системных библиотек:

```bash
sudo apt install python3-venv
/usr/bin/python3 -m venv --system-site-packages fliftbot_lidar_nav/.venv
fliftbot_lidar_nav/.venv/bin/python -m pip install pyzmq
```

ROS-окружение для `live`/`sim`
нужно предварительно загрузить. Установленный эквивалент:
`ros2 run fliftbot_lidar_nav lidar_nav COMMAND ...`.

## Карта из fliftbot_map_build

```bash
./fliftbot_lidar_nav/nav.sh check-map --map-dir /data/deploy/run01
```

Deploy содержит `map.pbstream`, `map.yaml` с изображением, `origin_reference.json`,
`navigation_config.json`, web-матрицы и `manifest.json` с SHA-256.
Проверяются файлы, контрольные суммы, калибровка и соответствие origin матрице.
PBStream нужен Cartographer; occupancy нужна для первичного поиска и offline ICP.
Один JPEG/PGM не является полноценным deploy-каталогом.

`navigation_config.json` создаётся mapper из **эффективных параметров запуска**:
mount/yaw LiDAR, знаки энкодеров, steering zero, wheelbase и модель одометрии.
При `--map-dir` эти значения заменяют калибровку старой демонстрационной карты.
Тип LiDAR в bundle должен соответствовать выбранному `--sensor`.
Старые deploy-каталоги следует повторно упаковать новой `package_bundle.py`
из исходного RUN с `run_parameters.json`.

## Локализация записей без ROS

```bash
./fliftbot_lidar_nav/nav.sh localize-recording \
  --recording /data/canonical --map-dir /data/deploy/run01 \
  --output /data/localization/run01
```

Вход: `horizon/*.npz`, XY в метрах в системе LiDAR. GT и энкодеры не читаются.
Первый скан ищется по всей occupancy-карте; затем ICP использует предыдущую
принятую позу. Можно задать `--initial X_MAP Y_MAP YAW_RAD` для начального
приближения в исходной системе PBStream. Протяжённые однообразные проходы
лучше запускать с известным приближением; геометрически похожие места
могут дать неоднозначный глобальный поиск.

`trajectory.csv` содержит исходную map-позу и `x,y,alpha_rad` в локальной системе
bundle; `metrics.json` — долю принятых кадров и ошибки сопоставления.
Код `0`: принято не меньше `--min-acceptance` (по умолчанию 90%); `3`: не принято.
Каталог результата должен быть пустым. `--stride` позволяет проредить запись.
Это проверка сопоставления сканов карте; continuous live использует также одометрию.

## Live в симуляторе

При запущенном `fliftbot_sim/sim.sh up` и карте того же мира/датчика:

```bash
./fliftbot_lidar_nav/nav.sh sim --map-dir /data/deploy/run01 \
  pose_csv:=/data/localization/live_sim.csv
```

Входы: `/lidar_top/scan`, `/wheel_encoder_odom`, `/clock`.
GT `/odom` не используется. Отдельное TF-дерево
`lidar_nav_map → lidar_nav_odom → lidar_nav_base → lidar_nav_sensor`
не конфликтует с TF визуализации симулятора. Везде `use_sim_time=true`.
Поза: `/localization/lidar_pose` и CSV. ContinuousNav в этом режиме отключён.

Для первичной привязки остановите робота. До старта Cartographer первый скан
сопоставляется с occupancy-картой.
Траектория запускается через `/start_trajectory` только после принятого
совпадения (RMSE ≤ 0,20 м, overlap ≥ 0,50); до этого map-поза не выдаётся.
Можно передать `initial_pose:="X_MAP,Y_MAP,YAW_RAD"`; приближение тоже
проверяется сканом. В симметричном помещении проверяйте выбранное место.

## Live на физическом роботе

```bash
source /opt/ros/humble/setup.bash
source /home/user/fliftbot/install/setup.bash
./fliftbot_lidar_nav/nav.sh live --sensor dephan --map-dir /data/deploy/run01 \
  send_to_navigation:=false mc_topic:=/fliftbot204/mc_telemetry
```

`--sensor pandar` выбирает Pandar40P; `dephan` — DePHAN M360.
Один процесс принимает UDP LiDAR, MC-адаптер публикует `/wheel_odom` и TF,
Cartographer локализуется по замороженному PBStream. При отсутствии `mc_topic`
номер робота берётся из hostname вида `flbot204`. UDP-порт/IP — в
`config/lidar_profiles.json`; их нужно согласовать с сетью датчика.
На один UDP-порт запускается один receiver.

Для включения передачи измерений работающему ContinuousNav задайте
`send_to_navigation:=true`. Сообщение `top_navigation` содержит
`[x_m,y_m,yaw_rad]` через ZMQ, endpoint по умолчанию
`ipc:///tmp/fliftbot_continuous_nav_sensor.ipc`.
ROS `/localization/marker98_pose` и live CSV сохраняют исторический угол
в **градусах** (`alpha_deg`); JSON содержит `alpha_unit=deg`.
`instant` JSON/CSV также используют **градусы** и явно содержат единицу угла.
Offline `localize-recording` CSV использует **радианы**.
`origin_reference.json` задаёт произвольный origin; физическая метка 98 не обязательна.

Одометрия новых карт использует скорость ведущего колеса:
`vx = speed_sign*enc_v*cos(delta)`, `wz = speed_sign*enc_v*sin(delta)/wheelbase`,
где `delta = steering_sign*(enc_dgr-steering_zero_deg)`.
Для старых демонстрационных карт сохранён их `legacy_sin` профиль.

## Один оборот LiDAR без ROS

```bash
./fliftbot_lidar_nav/nav.sh instant --sensor dephan --map-dir /data/deploy/run01
```

Первый неполный оборот отбрасывается. Выход: JSON с `status=localized`,
`x,y,alpha,ts`, RMSE/overlap и каталог с диагностикой в `results/`.
`low_confidence` возвращает код `3`. Для Pandar используйте `--sensor pandar`.
Без `--map-dir` доступны сохранённые reference-карты и их собственная калибровка;
они предназначены для воспроизведения старых запусков, а не для нового помещения.

## Структура и миграция

```text
nav.sh                    единая команда
src/fliftbot_lidar_nav/    CLI установленного ROS-пакета
scripts/                  UDP, ICP, MC/TF, replay и экспорт позы
launch/                   live robot и live sim
config/                   профили датчиков и pure localization Lua
maps/                     сохранённые reference-карты
tests/                   проверки контракта и локализации
docs/                    связанный pipeline и результаты проверки
```

ROS-пакет и Python-модуль называются `fliftbot_lidar_nav`.
Старое имя `flifltbot` больше не является именем рабочего ROS-пакета.
Полный прежний offline mapping, evaluation copy, frozen эксперименты,
неактуальные shell-обёртки и заменённая карта сохранены вне активного пакета:
`../tools/fliftbot_lidar_nav_archive/20261030/`. Там есть `COLCON_IGNORE`.
История Git сохранена в этом репозитории; remote пока указывает на прежний
GitHub `fliftbot_mapping` (удалённый репозиторий не переименовывался).
Проверки и их ограничения: [verification](docs/verification.md).
