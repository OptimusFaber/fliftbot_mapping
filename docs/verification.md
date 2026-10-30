# Проверка интеграции — 30.10.2026

Среда: Linux, системный Python 3.10, ROS 2 Humble, Gazebo Fortress,
системная библиотека Cartographer. Использован `PYTHONNOUSERSITE=1`:
пользовательский matplotlib в этом workspace несовместим с системным NumPy.

## Результаты

| Проверка | Результат |
|---|---|
| `fliftbot_sim/sim.sh verify` | 25 tests passed, 326 asset checksums, 22 SDF, local URI |
| `fliftbot_map_build` unittest | 18 tests passed |
| CTest standalone Cartographer | 1 passed |
| `fliftbot_lidar_nav/nav.sh verify` | 8 tests passed, включая локальную ZMQ передачу |
| `colcon build` + `colcon test` navigation | Пакет собран; 8 pytest tests passed (pyzmq из локальной .venv) |
| Отдельная обычная установка в `/tmp/fliftbot_nav_install_check` | `ros2 run ... check-map` и `verify` работают вне source directory |
| Запуск/остановка ROS localization | PBStream загружается, все процессы завершаются по SIGINT, включая остановленный sim clock |

Тесты передачи карты проверяют импорт старого симуляционного config,
исключение GT/глобальной одометрии, эффективную калибровку, знак yaw при экспорте
origin, отказ от перезаписи deploy, повреждение файла/manifest, несовпадающий
LiDAR, расхождение origin и матрицы. Локализация проверяется с ненулевым
монтажным yaw и повёрнутой локальной системой; пустые сканы возвращают отказ.

## Сквозной прогон

1. `fliftbot_sim`: `room`, `dephan`, 25 Hz; запись 35 s simulation time.
   Измеренное движение около 0,2 м/с с последующей остановкой.
2. Получено 848 NPZ-кадров, 707 encoder samples, 181 из них при движении.
   Запись энкодеров 20 Hz, LiDAR 25 Hz. Потерь сохранённых кадров нет.
3. Импорт в canonical без `odom.json`/GT; конфиги взяты из симулятора:
   mount `(1.37,0.315,2.04)`, yaw=0, speed sign=-1, wheelbase=1.410.
4. `fliftbot_map_build`: 847 сканов поданы в Cartographer (первый служит
   инициализации/согласованию времени), 698 odom samples, 13 constraints.
   PBStream, occupancy, web и deploy созданы; bundle validation accepted.
5. `fliftbot_lidar_nav`: приняты **848/848** кадров;
   median ICP RMSE **0,01725 м**, возвращён код 0.
6. В новом экземпляре Gazebo запущен live `nav.sh sim` на том же deploy.
   Первичное глобальное сопоставление: RMSE **0,018 м**, overlap **0,996**;
   затем `/start_trajectory` принял trajectory 1 на frozen trajectory 0.
   В CSV получено 109 поз. Использовались `/wheel_encoder_odom`, scan, clock;
   GT `/odom` и его TF-дерево не подключались к локализации.

Данные и результаты, исключённые из Git:

```text
ros-data/recordings/sim/20261030/lidar_nav_room_motion/
ros-data/results/integration_lidar_nav/20261030_170729_room_motion/
ros-data/deploy/lidar_nav_room_motion/
ros-data/results/integration_lidar_nav/localization_room_motion/
ros-data/results/integration_lidar_nav/live_sim_acquired_pose.csv
ros-data/results/integration_lidar_nav/verification_summary.json
ros-data/results/integration_lidar_nav/reproducible_check/verification.json
```

Повторяемый сквозной offline-прогон из корня workspace:

```bash
PYTHONNOUSERSITE=1 /usr/bin/python3 tools/verify_mapping_pipeline.py \
  --recording ros-data/recordings/sim/20261030/lidar_nav_room_motion \
  --output /data/new_integration_check
```

Команда создаёт canonical, строит новую карту, упаковывает и проверяет deploy,
локализует исходную запись, сохраняет `verification.json`.
Начальное приближение берётся из построенной robot-axis траектории,
а не из GT; последующие позы оцениваются по LiDAR.

## Найденное и исправленное

- Старый mapping suite не собирался: тесты ссылались на удалённые v5/v6 scripts.
  Весь прежний offline mapping и тесты сохранены в архиве с `COLCON_IGNORE`;
  активный набор проверяет новый контракт и используемую локализацию.
- Симулятор писал legacy config без component-схемы, speed sign и robot config.
  Добавлены актуальные config-поля и importer для старых записей.
- Deploy не переносил калибровку. Теперь `navigation_config.json` создаётся
  из эффективного `run_parameters.json` и включён в SHA-256 manifest.
- При выводе origin из матрицы был инвертирован знак yaw. Исправлено и проверено
  для ненулевого поворота, чтобы web/local координаты совпадали с runtime.
- Live-одометрия применяла скорость ведущего колеса как скорость оси без cos(delta).
  Новые bundle используют общую модель `drive_wheel`, совпадающую с mapper.
- Cartographer мог выдавать identity-позу до глобальной привязки.
  Теперь траектория стартует после принятого occupancy/ICP сопоставления.
- Generated protobuf контроллера требовал runtime 5.28 и внешний checkout.
  Включена совместимая wire-схема с поддержкой protobuf 3+, транспорт проверен
  отдельным локальным потребителем. На этом хосте pyzmq установлен в локальной
  `.venv` через uv, без sudo; системные библиотеки не менялись.
- Повторный SIGINT мог прерывать cleanup; синхронный shutdown в signal handler
  мог зависать при остановленном sim clock. Обработчик теперь ставит флаг,
  executor завершается и освобождает ROS вне signal handler.

Очень короткая первая запись стоящего робота (173 кадра, меньше 8 s) дала
только два Cartographer nodes и occupancy ниже порога занятости. Mapper
корректно отказался создавать web-карту. Успешный основной прогон содержит
движение и более длительное накопление данных.

## Границы проверки

ICP RMSE — ошибка сопоставления скана карте, а не абсолютная ошибка позиции
по независимому GT. Offline replay исходной mapping-записи проверяет интеграцию;
live-прогон проверяет новую симуляционную сессию. Большие склады, повторяющиеся
проходы и физический UDP LiDAR/MC на роботе отдельно не измерялись.
Формат и доставку ZMQ проверили локальным SUB-потребителем: тип top_navigation,
время, метры, перевод 90° в π/2 рад, variance. Схема контроллера включена в пакет
и работает на protobuf 3.12; абсолютные пути и внешний generated pb2 не нужны.
В запущенный ContinuousNav измерения не передавали, реальным роботом не управляли.
Для первичного поиска робот должен стоять.
