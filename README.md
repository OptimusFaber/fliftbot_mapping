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
sudo apt install ros-humble-cartographer-ros python3-numpy python3-scipy python3-pil python3-yaml python3-zmq python3-protobuf
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

В контейнере робота загрузите ROS и интерфейсы MC. Если discovery в вашей
сети настроен через XML-профиль, используйте тот же файл, что и в прежнем запуске:

```bash
apt-get update                                  # если cartographer еще
apt-get install -y ros-humble-cartographer-ros  # не установлен на fliftbot-е

source /opt/ros/humble/setup.bash
source /home/user/fliftbot/install/setup.bash
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/user/fliftbot/src/super_client_configuration_file.xml
ros2 daemon stop
ros2 daemon start
```

При переносе с компьютера можно упаковать исходники вместе с reference-картами,
исключив локальное Python-окружение и журналы:

```bash
# На компьютере, из workspace:
tar --exclude='.venv' --exclude='__pycache__' --exclude='logs' \
  --exclude='.git' -czf /tmp/fliftbot_lidar_nav.tar.gz fliftbot_lidar_nav
scp /tmp/fliftbot_lidar_nav.tar.gz USER@ROBOT_IP:/tmp/

# На хосте робота (подставьте имя ROS-контейнера):
docker cp /tmp/fliftbot_lidar_nav.tar.gz CONTAINER_NAME:/tmp/
docker exec -it CONTAINER_NAME bash

# В контейнере:
tar -xzf /tmp/fliftbot_lidar_nav.tar.gz -C /home/user
```

Установите Python/ROS-зависимости из раздела «Установка» в среде робота.
При прямом `ros2 launch /путь/к/launch.py` сборка этого Python-пакета не нужна;
`fliftbot_interfaces` и Cartographer должны быть установлены и загружены через
ROS setup. Локальную `.venv` с компьютера на робот не переносите.
Путь к пакету ниже — `/home/user/fliftbot_lidar_nav`; замените его своим.
Старый путь `realtime_mapping/config/ros/…` после объединения пакетов заменён
на `fliftbot_lidar_nav/launch/…`. Изменять `/home/user/fliftbot` не требуется.

### Воспроизведение выбранного варианта logs5

Итоговая команда из `lidar_localization.txt` от 08.10 использовала yaw 180°,
sin-модель без cos-поправки скорости, wheelbase 1,41 м, global sampling 0,5 и
дополнительное смещение выходной точки (−0,04; −0,045) м. Для **той же
сохранённой карты помещения** полный современный эквивалент:

```bash
ros2 launch /home/user/fliftbot_lidar_nav/launch/mc_lidar_localization.launch.py \
  sensor:=dephan \
  send_to_navigation:=true \
  navigation_send_rate_hz:=10 \
  sensor_x:=1.329 sensor_y:=0.315 sensor_yaw_deg:=180.0 \
  speed_sign:=-1.0 steering_sign:=1.0 steering_zero_deg:=-90.0 \
  odom_model:=legacy_sin wheelbase:=1.41 \
  correlative_linear_window_m:=0.15 correlative_angular_window_deg:=10.0 \
  rangefinder_sampling_ratio:=0.5 global_sampling_ratio:=0.5 \
  optimize_every_n_nodes:=10 \
  pose_offset_x_m:=-0.04 pose_offset_y_m:=-0.045 \
  scan_rate_hz:=20
```

Без `map_dir` этот launch выбирает bundled reference-карту DePHAN
`maps/dephan_20260922_corrected_run_170557`. `mc_topic` автоматически определяется
по hostname `flbot204` → `/fliftbot204/mc_telemetry`; при другом hostname явно
добавьте `mc_topic:=/fliftbot204/mc_telemetry` с номером вашего робота.

`odom_model:=legacy_sin` заменяет старое `use_legacy_sin:=true`.
Старый переключатель поддерживается для совместимости: `true` выбирает
`legacy_sin`, `false` выбирает `bicycle`. Отдельного алгоритма у него нет;
в новых командах используйте `odom_model`. Противоречивые значения двух
параметров отклоняются до запуска процессов.

В истории датчик работал на 25 Гц. После выявленных сбоев выбран режим
20 Гц; поэтому здесь указано 20. `scan_rate_hz` задаёт `LaserScan.scan_time`,
**он не переключает частоту физического лидара и не ограничивает публикацию**:
receiver публикует каждый собранный оборот. Частоту датчика настройте отдельно,
а фактическую частоту проверьте через `ros2 topic hz /scan`.
Для точного воспроизведения исторических метаданных 25 Гц задайте 25,
если физический датчик действительно работает в этом режиме.

Варианты yaw 176,73°, `bicycle` (tan) и wheelbase 1,385 м в историческом logs5
ухудшали результат. Эти наблюдения относятся к той карте и тем проездам.
Средняя ошибка около 5,1 см и P95 около 8,6 см в истории не означают,
что условие ошибки до 5 см уже выполнено на всём маршруте.

### Работа с новой картой из mapper

Для новой карты используйте согласованную с ней калибровку из deploy-bundle:

```bash
/home/user/fliftbot_lidar_nav/nav.sh live \
  --sensor dephan --map-dir /data/deploy/run01 \
  send_to_navigation:=false \
  mc_topic:=/fliftbot204/mc_telemetry \
  scan_rate_hz:=20 \
  scenario_config:=/data/scenarios/route01.json \
  marker_pose_csv:=/data/localization/run01/poses.csv
```

Здесь mount, yaw, знаки энкодеров, steering zero, wheelbase и модель берутся
из `navigation_config.json`. Дополнительные `pose_offset_x_m/y_m` по умолчанию
нулевые. Историческую коррекцию центра и калибровку logs5 нельзя автоматически
переносить на другую карту: они могут повторно внести уже учтённые поправки.
Явные launch-параметры имеют приоритет над bundle, затем идут профили датчиков.

После проверки общей системы координат и выходных поз включите передачу:
`send_to_navigation:=true`. Её launch-default — **true**, поэтому в команде
для первичной проверки выше отключение указано явно.

### Поток данных и проверка

Один процесс принимает UDP LiDAR и преобразует сканы в `base_link`.
MC-адаптер публикует `/wheel_odom` и `odom → base_link`.
Cartographer загружает frozen PBStream и публикует `map → odom`.
Для начальной привязки робот должен стоять: первый скан проверяется по
occupancy/ICP, затем вызывается `/start_trajectory`. Приближение
`initial_pose:=X_MAP,Y_MAP,YAW_RAD` задаётся в системе PBStream и тоже
проверяется сканом. Принятие начального ICP не гарантирует выбор правильного
места в симметричном помещении; проверьте его перед движением.

```bash
ros2 topic hz /scan
ros2 topic hz /wheel_odom
ros2 topic hz /localization/marker98_pose
ros2 topic echo /localization/marker98_pose --once
ros2 run tf2_ros tf2_echo map base_link
```

Ожидаются частота физических оборотов LiDAR (сейчас около 20 Гц),
одометрия по поступлению MC (обычно около 20 Гц), выход позы около 10 Гц.
До успешной первичной привязки позы нет. На один UDP-порт запускается один
receiver: DePHAN — `50007`, источник `192.168.0.120`; Pandar — `2368`,
источник `192.168.0.223`. Порт и IP можно переопределить параметрами ниже.

ROS `/localization/marker98_pose` содержит JSON вида
`{"x":1.23,"y":-0.45,"alpha":90.0,"alpha_unit":"deg","ts":1790000000.0}`.
Поза задана в локальной системе `origin_reference` или старой marker98-reference:
`x,y` — метры, `alpha` — **градусы**, `ts` — Unix-время.
Live CSV: `x,y,alpha_deg,ts`. Исторический CSV может повторять или сохранять
устаревшие позы; для доступности и срывов используйте JSONL-журнал.

Для ContinuousNav отправляется protobuf `SensorMeasurement` типа
`top_navigation`: `[x_m,y_m,yaw_rad]`, время исходного TF, variance
`[0.05²,0.05²,(3° в радианах)²]`. Это заданные дисперсии измерения,
а не измеренная точность локализации. Endpoint по умолчанию
`ipc:///tmp/fliftbot_continuous_nav_sensor.ipc`. Sender подключается (`connect`)
к endpoint потребителя. Очередь отправки ограничена одним сообщением;
повторные timestamps и позы старше `max_sensor_age` не отправляются.
Успешный ZMQ send сам по себе не подтверждает приём контроллером.

Live CSV по умолчанию создаётся в `fliftbot_lidar_nav/logs/`; рядом автоматически
создаётся новый `.journal.jsonl`. Для разных заездов используйте разные пути.
Ctrl+C останавливает launch и дочерние процессы. После остановки проверьте:

```bash
pgrep -af 'cartographer_node|direct_udp_scan_publisher|mc_telemetry_odom_adapter|live_marker98_pose|initialize_localization'
```

Второй экземпляр Cartographer или адаптера для того же робота запускать
параллельно не следует. Большой JSONL-журнал занимает больше места, чем старый
pose CSV: контролируйте свободное место при длительных запусках.

### Модели одометрии

`v = speed_sign × enc_v`, `δ = steering_sign × (enc_dgr − steering_zero_deg)`:

| `odom_model` | Линейная скорость оси | Угловая скорость |
|---|---|---|
| `legacy_sin` | `v` | `v × sin(δ) / wheelbase` |
| `bicycle` | `v` | `v × tan(δ) / wheelbase` |
| `drive_wheel` | `v × cos(δ)` | `v × sin(δ) / wheelbase` |

`drive_wheel` соответствует скорости, измеренной на ведущем управляемом
колесе. `bicycle` предполагает скорость продольного движения оси и становится
сингулярной около ±90°. `legacy_sin` сохранён для старых карт и воспроизведения
проверенных запусков. Замена sin на tan не является эквивалентным исправлением
модели ведущего колеса: у `drive_wheel` меняется и линейная скорость.

### Все параметры live-launch

Значения ниже — defaults **кода**, а не подставленные значения команды logs5.
Пустое значение (`""`) означает выбор из bundle/профиля, как указано в таблице.
Параметры Cartographer читаются при создании траектории; для их изменения
нужен перезапуск. При `configuration_basename` с другим Lua-файлом переменные
окружения действуют только если этот файл их использует.

| Параметр | Default | Назначение |
|---|---|---|
| `journal` | `""` | Новый JSONL-файл; пусто → `marker_pose_csv + .journal.jsonl`. |
| `scenario_config` | `""` | JSON сценария; пусто → defaults журнала, порог угла не задан. |
| `map_dir` | `""` | Deploy-каталог; пусто → сохранённая reference-карта датчика. |
| `initial_pose` | `""` | `X_MAP,Y_MAP,YAW_RAD` для первичного поиска; пусто → глобальный поиск. |
| `sensor` | `dephan` | `dephan` или `pandar`; выбирает сетевой профиль и reference-карту. |
| `map_pbstream` | `""` | Явный PBStream; его каталог используется для первичного occupancy-поиска. Калибровка и reference должны соответствовать ему. |
| `marker_reference` | `""` | Legacy reference JSON; требует `map_pbstream`, несовместим с `origin_reference`. |
| `origin_reference` | `""` | Origin JSON; требует `map_pbstream`, несовместим с `marker_reference`. Без overrides берётся из bundle/профиля. |
| `mc_topic` | `""` | Автоматически `/fliftbotNNN/mc_telemetry` по hostname. |
| `scan_topic` | `/scan` | Выход UDP receiver / вход Cartographer и журнала. |
| `odom_topic` | `/wheel_odom` | Выход MC-адаптера / вход Cartographer и журнала. |
| `udp_bind` | `0.0.0.0` | Локальный IP для bind UDP-сокета. |
| `udp_port` | `""` | Порт из профиля: DePHAN 50007; Pandar 2368. |
| `udp_source` | `""` | Фильтр source IP из профиля: DePHAN 192.168.0.120; Pandar 192.168.0.223. |
| `scan_rate_hz` | `""` | Только `scan_time=1/rate`, Гц; default из профиля: DePHAN 25, Pandar 10. Физическую частоту не меняет. |
| `range_max` | `""` | Верхняя дальность LaserScan, м; из bundle/профиля, fallback 30. Lua Cartographer дополнительно ограничивает дальность 30 м. |
| `sensor_x` | `""` | X LiDAR относительно оси робота, м; bundle/профиль. |
| `sensor_y` | `""` | Y LiDAR относительно оси робота, м; bundle/профиль. |
| `sensor_yaw_deg` | `""` | Yaw преобразования скана в корпус, градусы; bundle/профиль. В logs5 явно 180; bundled default DePHAN около 176,73. |
| `speed_sign` | `""` | Знак энкодерной скорости, −1 или +1; bundle/профиль (DePHAN −1). |
| `steering_sign` | `""` | Знак угла колеса, −1 или +1; bundle/профиль, fallback +1. |
| `steering_zero_deg` | `""` | Сырой угол колеса при прямом движении; bundle/профиль (старый DePHAN −90°). |
| `wheelbase` | `""` | База, м; bundle/профиль (старый DePHAN 1,41). |
| `odom_model` | `""` | `drive_wheel`, `legacy_sin`, `bicycle`; bundle/профиль, fallback `drive_wheel`. |
| `use_legacy_sin` | `""` | Совместимость: true → `legacy_sin`, false → `bicycle`; пусто → `odom_model`/bundle. Конфликт моделей запрещён. |
| `marker_pose_csv` | `""` | Путь CSV; пусто → `logs/<robot>_<sensor>_<timestamp>.csv`. |
| `pose_offset_x_m` | `0.0` | Дополнительный X выходной точки в `base_link`, м; применяется после локализации. |
| `pose_offset_y_m` | `0.0` | Дополнительный Y выходной точки в `base_link`, м; поворачивается вместе с роботом. |
| `pose_rate_hz` | `10.0` | Частота ROS/CSV и проверок состояния журнала, Гц. |
| `publish_tf` | `true` | Публиковать `map → odom`; требуется для экспорта позы и ContinuousNav. |
| `publish_tracked_pose` | `false` | Дополнительный топик Cartographer tracked_pose. |
| `pose_publish_period_sec` | `0.05` | Период публикации позы Cartographer, с. |
| `trajectory_publish_period_sec` | `3600.0` | Период публикации траектории, с; 3600 практически выключает регулярную визуализацию. |
| `submap_publish_period_sec` | `3600.0` | Период публикации submaps, с; 3600 практически выключает регулярную визуализацию. |
| `rangefinder_sampling_ratio` | `0.5` | Доля входных сканов, подаваемых в Cartographer. |
| `global_sampling_ratio` | `0.2` | Частота попыток глобального matching; в итоговой команде истории явно 0,5. |
| `optimize_every_n_nodes` | `10` | Оптимизация pose graph через N nodes. |
| `motion_filter_max_time_sec` | `0.5` | Максимальный интервал между добавленными nodes, с. |
| `motion_filter_max_distance_m` | `0.2` | Порог перемещения motion filter, м. |
| `motion_filter_max_angle_deg` | `1.0` | Порог поворота motion filter, градусы. |
| `use_online_correlative_scan_matching` | `true` | Включить online correlative matcher. |
| `correlative_linear_window_m` | `0.15` | Линейное окно online correlative matcher, м. |
| `correlative_angular_window_deg` | `10.0` | Угловое окно online correlative matcher, градусы. |
| `log_matches` | `false` | Диагностические сообщения о совпадениях Cartographer. |
| `log_residual_histograms` | `false` | Диагностические гистограммы residuals Cartographer. |
| `send_to_navigation` | `true` | Передавать позу в ContinuousNav; default включён. |
| `navigation_send_rate_hz` | `10.0` | Максимальная частота отправки новых поз в ContinuousNav, Гц. |
| `max_sensor_age` | `0.5` | Максимальный возраст позы для ContinuousNav, с. Допуск будущего timestamp фиксирован: 0,25 с. |
| `continuous_nav_sensor_socket` | `ipc:///tmp/fliftbot_continuous_nav_sensor.ipc` | ZMQ endpoint потребителя ContinuousNav. |
| `publish_occupancy_grid` | `false` | Включить дополнительный occupancy-grid ROS node; обычно не требуется. |
| `configuration_directory` | `<пакет>/config/cartographer` | Каталог Lua Cartographer; default `<пакет>/config/cartographer`. |
| `configuration_basename` | `fliftbot_2d_pure_localization.lua` | Lua-файл, используемый и Cartographer, и `/start_trajectory`. |

Фиксированные параметры стандартного Lua/receiver, которые не имеют
отдельного launch-переключателя:

- 2D без IMU; одна LaserScan и одометрия; скан уже в `base_link`.
- LaserScan: 720 угловых bins, минимум дальности 0,15 м, `time_increment=0`
  после преобразования и угловой перегруппировки точек.
- Первичная привязка: не меньше 100 точек, ICP RMSE ≤ 0,20 м, overlap ≥ 0,50.
- Lua: диапазон 0,15–30 м, 90 range data на submap, хранение трёх активных
  localization submaps; `min_score=0.65`, `global_localization_min_score=0.60`.
- Веса одометрии в pose graph: translation=0,25, rotation=0,25.
- `use_sim_time=false` на физическом роботе, без изменения карт при локализации.

Исходные значения: [pure localization Lua](config/cartographer/fliftbot_2d_pure_localization.lua)
и [базовый Lua](config/cartographer/fliftbot_2d.lua). Они также влияют на результат:
одного совпадения wheelbase и yaw недостаточно для доказательства прежней точности.

## Один оборот LiDAR без ROS

```bash
./fliftbot_lidar_nav/nav.sh instant --sensor dephan --map-dir /data/deploy/run01
```

Первый неполный оборот отбрасывается. Выход: JSON с `status=localized`,
`x,y,alpha,ts`, RMSE/overlap и каталог с диагностикой в `results/`.
`low_confidence` возвращает код `3`. Для Pandar используйте `--sensor pandar`.
Без `--map-dir` доступны сохранённые reference-карты и их собственная калибровка;
они предназначены для воспроизведения старых запусков, а не для нового помещения.

На проверенном 90-секундном участке реальной записи 02.10 со старой калибровкой
`legacy_sin` дал среднее расхождение 6,9 см, P95 11,4 см и 0 срывов по порогам
20 см / 10°. `drive_wheel` и `bicycle` на той же старой калибровке дали срывы.
Это поддерживает сохранение `legacy_sin` для исторического запуска; требование
точности до 5 см этой проверкой не подтверждено.
Результаты сверки с историей, проверки моделей на реальной записи и ограничения:
[аудит live-локализации](docs/live_localization_audit.md).

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

## Журнал и испытания по сценариям

Live-запуск сохраняет рядом с pose CSV журнал `.journal.jsonl`: свежесть позы,
состояние входов и события. Команда `nav.sh report` сравнивает его с эталонным
CSV и формирует отчёт по точности, доступности и срывам.
[Настройка сценария и методика](docs/localization_trials.md).

## Отправка protobuf внешнему потребителю

Функция `send_sensor_measurement` находится в
[scripts/navigation_output.py](scripts/navigation_output.py), читаемая схема —
[config/protobuf/sensor_measurement.proto](config/protobuf/sensor_measurement.proto).
Для передачи коллеге без ROS достаточно этих двух файлов и
[scripts/navigation_measurement.py](scripts/navigation_measurement.py).
Последний содержит готовую protobuf-схему; запускать protoc не требуется.
Зависимости: `pyzmq`, `protobuf`.

Пример из каталога `scripts/`, после создания **одного постоянного** PUB-сокета:

```python
import math
import zmq
from navigation_output import send_sensor_measurement

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.setsockopt(zmq.LINGER, 0)
socket.setsockopt(zmq.SNDHWM, 1)
socket.connect("tcp://192.168.0.10:5555")  # IP/порт принимающей стороны

# Вызывать из рабочего цикла после подключения SUB-потребителя.
def send_pose(measurement_timestamp, x_m, y_m, yaw_rad):
    send_sensor_measurement(
        socket, timestamp=measurement_timestamp,
        value=[x_m, y_m, yaw_rad],
        variance=[0.05**2, 0.05**2, math.radians(3)**2],
        sensor_type="top_navigation",  # либо "lidar", если так ожидает потребитель
    )

# При остановке: socket.close(linger=0); context.term()
```

Получатель создаёт `zmq.SUB`, выполняет `bind("tcp://*:5555")`,
`setsockopt(zmq.SUBSCRIBE, b"")` и декодирует принятый байтовый пакет через
`SensorMeasurement.FromString(payload)`. При старте PUB/SUB требуется время
на обнаружение подписчика; доставка каждого пакета не подтверждается.
Для обмена между компьютерами используйте TCP; `ipc://` работает локально.
Штатный live exporter продолжает отправлять тип `top_navigation` (0).
