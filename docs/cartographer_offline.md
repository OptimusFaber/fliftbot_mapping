# Надёжная запись и офлайн Cartographer

## Replay без rosbag2

Для уже сохранённой DePHAN-записи rosbag2 не требуется. Запускайте штатный
Cartographer и тонкий ROS 2 адаптер, который читает `horizon/*.npz` и
`encoder.jsonl` напрямую:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch flifltbot cartographer_npz_replay.launch.py \
  run_dir=/home/optimus/ros-projects/ros-data/recordings/real/20260903/114127_dephan_m360_room \
  write_state=/home/optimus/ros-projects/ros-data/results/real/20260903/cartographer_npz/map.pbstream \
  rate:=1.0
```

Адаптер публикует `/scan` (`sensor_msgs/LaserScan` в `lidar_top`), `/odom`
(`nav_msgs/Odometry` с велосипедной моделью из `enc_v`/`enc_dgr`),
`wheel_odom -> base_link` в `/tf`, статический `base_link -> lidar_top` в
`/tf_static` и rebased `/clock`. Поэтому время исходной записи (Unix epoch)
не конфликтует с текущими часами машины. `rate:=0` публикует максимально
быстро; для полной записи 24.7 Hz обычно используйте `rate:=1`.

Параметр `write_state` необязателен. Если Cartographer запущен, адаптер после
последнего скана подождёт `settle` секунд и вызовет `/write_state`, сохраняя
`.pbstream`; пустой `write_state` оставляет узлы запущенными для ручного
сохранения. Для записи с другим креплением можно переопределить параметры
CLI-узла `--lidar-x`, `--lidar-z`, `--lidar-frame`.

Входом офлайн-Cartographer является каталог `rosbag2` (SQLite3), а не NPZ-файлы.
Чтобы не создавать второй формат внутри нормализованного дерева
`ros-data/recordings`, bags по умолчанию сохраняются отдельно в
`ros-data/cartographer_bags/{real|sim}/YYYYMMDD/HHMMSS_cartographer_{scene}`.
Запись выполняется инкрементально на локальный диск, а при достижении лимита
времени rosbag2 начинает следующий `.db3` chunk. После остановки запись можно
проверять, копировать и обрабатывать сколько угодно раз с разными Lua-настройками.

## Запись

В симуляции после запуска Gazebo и bridge:

```bash
./fliftbot_mapping/scripts/mapping/record_cartographer_run.sh \
  --kind sim --sim-time \
  --scene real_room \
  --scan-topic /lidar_top/scan --odom-topic /wheel_encoder_odom \
  --output /home/optimus/ros-projects/ros-data/cartographer_bags/sim/20260907/153000_cartographer_real_room
```

На реальном роботе запускайте ту же команду на самом роботе (или edge-компьютере):

```bash
./fliftbot_mapping/scripts/mapping/record_cartographer_run.sh \
  --kind real --scene room --scan-topic /scan --odom-topic /odom \
  --output /data/cartographer_bags/real/20260907/153000_cartographer_room
```

Для HTTP-робота `10.0.2.6` сначала запустите мост. Он публикует `/scan`,
объединяя доступные `left` и `right` в `base_link`, и `/odom` из энкодера:

```bash
ros2 run fliftbot_real real_http_lidar_bridge --ros-args \
  -p ip:=http://10.0.2.6:10101 -p poll_hz:=20.0
```

В запись входят LiDAR `LaserScan`, колесная `Odometry`, `/tf`, `/tf_static`, и
при `--sim-time` `/clock`. `--chunk-seconds 300` по умолчанию ограничивает
размер одного файла. `--max-cache-size 0` просит rosbag2 писать каждое
сообщение напрямую в SQLite. Это снижает окно потерь при аварии, но не может
гарантировать сохранность данных при отключении питания или отказе диска.
При сетевом обрыве данные не теряются только если recorder работает на
источнике данных; удалённый recorder не получит сообщения, которые не дошли по
сети. После обычного `Ctrl-C` манифест получает `status=complete`; при жёстком
обрыве остаются уже закрытые chunks и `metadata.yaml`.

## Проверка

```bash
./fliftbot_mapping/scripts/mapping/validate_cartographer_recording.py \
  RUN_DIR --scan-topic /lidar_top/scan --odom-topic /wheel_encoder_odom \
  --clock --json RUN_DIR/validation.json
```

Проверяются наличие обязательных топиков, число сообщений и монотонность
времени SQLite-записи. Затем offline-скрипт запускает штатный
`cartographer_rosbag_validate`, который десериализует сообщения и проверяет
временные метки датчиков. `--clock` нужен для симуляционных записей.

## Построение и повторные эксперименты

```bash
./fliftbot_mapping/scripts/mapping/offline_cartographer.sh \
  --bag RUN_DIR --sim-time \
  --output RUN_DIR/cartographer_v1
```

Скрипт использует установленный `cartographer_offline_node`, затем
`cartographer_pbstream_to_ros_map`. В каталоге результата будут:

* `map.pbstream` — редактируемый/повторно экспортируемый SLAM state;
* `map.pgm` и `map.yaml` — occupancy map;
* `validation_*.json` — отчёт входной проверки;
* `trajectory.json` — оптимизированная траектория;
* `trajectory_query_node.log` — журнал запроса поз.

Путь к Lua можно менять без повторной записи:

```bash
./fliftbot_mapping/scripts/mapping/offline_cartographer.sh --bag RUN_DIR \
  --configuration-dir fliftbot_mapping/config/cartographer \
  --configuration-file fliftbot_2d.lua --output RUN_DIR/cartographer_v2
```

`--bag` также принимает список каталогов через запятую. Для одного run,
разбитого на chunks, достаточно передать его корневой каталог: Cartographer
прочитает `metadata.yaml` и все SQLite-файлы.

## Формат trajectory.json

Документ имеет схему `fliftbot.cartographer_trajectory.v1`. Каждая запись в
`poses` содержит `timestamp_ns`, `global` и `pose`. `global` — координаты и
ориентация в Cartographer frame (`map`), включая `x,y,z`, `roll,pitch,yaw` и
quaternion. `pose` — та же поза в локальной СК, где первая валидная поза ровно
`(0,0,0)` и yaw равен нулю. Для XY применяется полный SE(2)-переход
`R(-origin.yaw) * (global_xy - origin_xy)`, а не только вычитание координат.
Исходная поза и параметры преобразования сохраняются в `origin` и
`normalization`.

Положения запрашиваются сервисом `cartographer_ros_msgs/srv/TrajectoryQuery`
после загрузки `map.pbstream`, поэтому это оптимизированные позы после loop
closure. Они дискретны (узлы Cartographer), а не непрерывный pose при каждом
скане; интерполяцию при необходимости выполняйте потребителем JSON. Если
сервис недоступен, экспорт завершается ошибкой, не подменяя результат менее
точным `/tracked_pose`.
