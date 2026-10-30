# Симуляция → карта → навигационная локализация

Все команды ниже выполняются из workspace, где рядом находятся
`fliftbot_sim`, `fliftbot_map_build`, `fliftbot_lidar_nav`.
Для физической записи первые два шага заменяются
`fliftbot_map_build/src/map_build/build_from_raw.py` с калибровкой робота.

## 1. Запустить симуляцию и записать данные

```bash
source /opt/ros/humble/setup.bash
./fliftbot_sim/sim.sh build
source install/setup.bash
./fliftbot_sim/sim.sh up room dephan --headless --lidar-rate 25
```

В других терминалах с тем же ROS-окружением:

```bash
./fliftbot_sim/sim.sh teleop
./fliftbot_sim/sim.sh record 20 10 /data/sim_run 60
```

Проехать по доступной области и остановить запись. Восемь секунд на месте
могут не накопить достаточно попаданий для occupancy-порога 0,65.
Короткий успешный прогон не измеряет качество большой складской карты.
Выход: `/data/sim_run/lidar/{horizon,encoder.jsonl,lidar.config,liftbot.config}`
и отдельная камера. GT находится в `lidar/odom.json` и остаётся только для оценки.
Времена Gazebo — секунды simulation time; epoch 1970 при таком входе нормален.

## 2. Подготовить canonical

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 fliftbot_map_build/src/map_build/import_sim_recording.py \
  --recording /data/sim_run --output /data/canonical/run01
```

Импорт понимает RUN и RUN/lidar, а также старый `lidar.config.v1` симулятора.
Он копирует конфиги, сохраняет NPZ обычными файлами (hardlink при возможности)
и переносит только измеренные поля энкодеров. GT, старые глобальные odom-поля
и camera/GT-треки не передаются mapper. Исходная запись не изменяется.
Новый canonical-каталог должен быть пустым.
Для старой записи без `liftbot.config` wheelbase по умолчанию 1,410 м;
при другой геометрии передайте `--wheelbase`.

## 3. Построить карту и deploy

Сначала установите/соберите Cartographer по README `fliftbot_map_build`.

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 fliftbot_map_build/src/map_build/build_from_canonical.py \
  --recording /data/canonical/run01 --output /data/results --name room01 \
  --deploy-output /data/deploy/room01
```

Симуляционный config задаёт mount, yaw=0, speed sign=-1 и wheelbase=1,410 м.
Значения физического DePHAN (например yaw=180) к нему не подставляются.
При повторном построении выбирайте новое имя deploy-каталога.
Отдельные этапы видео и потолочных меток требуют camera.config/intrinsics;
для LiDAR-карты и локализации они не обязательны.

В `/data/results/DATE_TIME_room01` сохраняются PBStream/occupancy,
траектории, web-карта, параметры и отчёты. `/data/deploy/room01` содержит
полный runtime-контракт, включая эффективную калибровку.

## 4. Проверить карту и локализацию

```bash
./fliftbot_lidar_nav/nav.sh check-map --map-dir /data/deploy/room01
./fliftbot_lidar_nav/nav.sh localize-recording \
  --recording /data/canonical/run01 --map-dir /data/deploy/room01 \
  --output /data/localization/replay01
```

Без `--initial` первый кадр ищется глобально. Для известной начальной позы:
`--initial X_MAP Y_MAP YAW_RAD`. Это поза **оси робота** в PBStream frame,
не позиция LiDAR, не web-пиксели и не координаты после origin transform.
Для независимой проверки записывайте второй проезд и локализуйте его по
той же карте. Повторное чтение исходной записи проверяет интеграцию и
согласованность форматов, но не переносимость на другой заезд.

## 5. Использовать карту online

При продолжающейся симуляции:

```bash
./fliftbot_lidar_nav/nav.sh sim --map-dir /data/deploy/room01 \
  pose_csv:=/data/localization/live01.csv
```

После смены мира или датчика нужна соответствующая ему карта/калибровка.
Для физического робота:

```bash
./fliftbot_lidar_nav/nav.sh live --sensor dephan --map-dir /data/deploy/room01 \
  send_to_navigation:=false mc_topic:=/fliftbot204/mc_telemetry
```

Здесь `/data/deploy/room01` должна быть картой **этого физического помещения**
с физической калибровкой; карта произвольного мира Gazebo для него не подходит.
Live выдаёт ROS/CSV-позу; `send_to_navigation:=true` подключает существующий
ContinuousNav. Он должен быть отдельно запущен и иметь ту же систему координат.

## Координаты

`map.yaml` описывает raster origin occupancy и не является началом навигационной СК.
`origin_reference.json` задаёт точку `(x0,y0)` и угол `a0` в PBStream frame:

```text
x_local =  cos(a0)*(x_map-x0) + sin(a0)*(y_map-y0)
y_local = -sin(a0)*(x_map-x0) + cos(a0)*(y_map-y0)
a_local = wrap(a_map-a0)
```

Это тот же transform, что `pbstream_to_local_matrix.npy`.
Изменение web origin не требует изменять PBStream или occupancy pixels;
после изменения карту нужно заново упаковать. Проверяйте весь bundle,
чтобы не смешать PBStream одной карты с origin/калибровкой другой.
