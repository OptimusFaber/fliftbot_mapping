# V7 stored-ray candidate — 2026-09-14

## Итог

Создан отдельный экспериментальный кандидат V7. V6, его файлы и canonical
outputs не изменялись.

V7 переносит из notebook именно stored-ray геометрию:

```text
P = O + ((H - O_z) / d_z) d
```

При этом runtime читает `camera_position.txt`, `lidar.config`, timestamps и
явно интерполированную траекторию. Высота потолка оценивается по многим линиям
наблюдений; GT и room geometry не загружаются. После медианной агрегации по
decoder code V7 объединяет близкие центры независимо от ID. Это убирает
дубликаты, возникшие из-за переключения номера декодера.

## DePHAN regression

Запуск с тем же соглашением координат, которое использует notebook для этого
move log (его `camera_position.txt` lateral sign не отражается):

```bash
python3 fliftbot_mapping/scripts/evaluation/ceiling_marker_localization_v7.py \
  --camera-y-sign 1 \
  --out-dir ros-data/results/real/20260914/ceiling_marker_localization_v7_stored_ray_candidate/yminus_move_log_v2
```

Результат:

* auto ceiling: **3.0302 м**;
* 52 line fits, 48 согласованных по высоте, confidence **0.655**;
* 48 физических кластеров, включая anchor 98; после исключения anchor — 47;
* one-to-one matching, только translation `pred_98 -> gt_98`, rotation=0°,
  scale/reflection/GT-fitting отсутствуют;
* при gate 5 см: 39/47, precision 82.98%, matched MAE 1.90 см;
* при gate 10 см: **47/47**, precision **100%**, recall 57.32%, matched MAE
  **2.58 см**, RMSE 3.19 см, median 1.88 см, p95 6.21 см, max 6.80 см;
* false positives дальше 10 см: **0**.

До физического merge notebook-like набор содержал 51 точку после anchor и 4
дубликата: пары центров были практически в одном месте, но имели разные
decoder ID. V7 оставляет один наиболее поддержанный центр каждой такой группы.

Для Cartographer trajectory с `tracking_frame=lidar` V7 на stored rays не
выдаёт карту, а корректно abstain-ится. Это важно: stored ray из текущего
`observations.json` согласован с move-log/camera frame notebook, но не доказано,
что он уже выражен в optical frame, который можно напрямую вращать Cartographer
heading. Прямое смешивание этих conventions даёт неверную геометрию.

Поэтому V7 пока **не production**. Достигнутые 2.58 см — хороший результат
для физической геометрии, но ещё не доказательство переносимости на Pandar,
RPLIDAR и JSON trajectory. Нужен единый upstream контракт stored ray:
`ray_frame`, optical-to-camera mount rotation и timestamp source должны быть
записаны в observations metadata. Пока этого нет, команда с `--camera-y-sign 1`
явно выбирает соглашение текущего DePHAN move log и не должна автоматически
применяться к другим сценам.

## Тесты

```bash
python3 -m unittest -v \
  fliftbot_mapping/scripts/evaluation/test_ceiling_marker_localization_v7.py
```

Проверяются высоты 3.03/3.13/4.20 м, разные camera mounts, lidar-frame lever
arm, timestamp offset, rotated/translating trajectories, duplicate decoder IDs
и abstention без достаточной поддержки. Все 5 тестов проходят.

Артефакты:

* [ceiling_marker_localization_v7.py](../scripts/evaluation/ceiling_marker_localization_v7.py)
* [test_ceiling_marker_localization_v7.py](../scripts/evaluation/test_ceiling_marker_localization_v7.py)
* [marker_points_v7.json](../../ros-data/results/real/20260914/ceiling_marker_localization_v7_stored_ray_candidate/yminus_move_log_v2/marker_points_v7.json)
* [offline_unlabeled_audit.json](../../ros-data/results/real/20260914/ceiling_marker_localization_v7_stored_ray_candidate/yminus_move_log_v2/offline_unlabeled_audit.json)

## Обязательное исправление: encoder-only trajectory (2026-09-14)

Предыдущий запуск `yminus_move_log_v2` больше не является валидной метрикой
V7: он использовал `t,x,y,dgr` из move logger. Этот результат оставлен как
исторический эксперимент и не должен использоваться для приёмки алгоритма.

Текущая реализация V7 интегрирует только:

```text
enc_t, enc_v, enc_dgr
```

по bicycle model с контрактом:

```text
wheelbase = 1.410 м
v = -enc_v
steering = radians(enc_dgr)
wz = v * sin(steering) / wheelbase
```

Поля `x`, `y`, `dgr`, `nav_*` и `enc_unix` не читаются. Это проверяется
тестом: добавление заведомо неверных запрещённых полей не меняет интегрированную
траекторию. В JSON отчёта записываются `source_fields` и `forbidden_fields`.

На текущем DePHAN encoder-only запуск:

```bash
python3 fliftbot_mapping/scripts/evaluation/ceiling_marker_localization_v7.py \
  --encoder-log ros-data/recordings/real/20260908/160617_dephan_m360_room/encoder.jsonl \
  --camera-y-sign 1 \
  --out-dir ros-data/results/real/20260914/ceiling_marker_localization_v7_stored_ray_candidate/encoder_bicycle_v1
```

получил только 19 line fits из 71 decoder ID и корректно **abstain**-ился при
оценке общей плоскости потолка. Ослабление line-residual gate до 0.4, 0.8 и
1.2 м также не дало принятой плоскости. Поэтому прежние 2.58 см нельзя
приписывать encoder-only алгоритму.

Вывод: stored-ray geometry сама по себе даёт хороший результат, но для этой
записи bicycle odometry из одних `enc_*` недостаточно стабильна для безопасного
восстановления общей карты. Нельзя подбирать начальный yaw, знак времени или
wheelbase по GT/nav и выдавать это за обобщённое решение. V7 остаётся
экспериментальным и не продвигается в production.

Проверка знака `+enc_v` была только офлайн-разведкой. Она дала принятую
плоскость, но без корректного начального yaw физически совпали лишь 2 из 48
точек в пределах 10 см (6 из 48 в пределах 20 см). Это не решение и не новый
runtime default: line-fit самосогласованность может принять неверную encoder
траекторию.

Так как прежние 2.58 см получены на запрещённом move-log `x/y/dgr`, они
аннулированы как V7-метрика. Encoder-only кандидат на текущей записи безопасно
abstain-ится; улучшение V6 следует делать через LiDAR/Cartographer constraints,
построенные из разрешённых sensor данных, а не через возврат к nav/pose полям.
