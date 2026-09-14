# Marker 98: предварительный anchor-аудит стен

Дата: 2026-09-11. ВАЖНО: перечисленные ниже значения были использованы в ходе
предварительного анализа, но их независимое происхождение и допуски не подтверждены
пользователем. Их нельзя считать ground truth или доказательством точности. Они не
записываются в `floorplan.json`:

- east: `4.77 м`;
- south: `3.495 м`;
- west: `11.50..11.62 м` (исходное `11.50 м` плюс возможная поправка `+0.09..0.12 м`);
- north: `3.045 м`.

Anchor сохранён в [marker98_inner_wall_measurement_anchor_20260911.json](marker98_inner_wall_measurement_anchor_20260911.json). Машинный отчёт с полными residuals и wall-fit diagnostics — [marker98_measurement_anchor_audit_20260911.json](/home/optimus/ros-projects/ros-data/results/real/20260910/160617_dephan_m360_room/markers_lidar_tracking_calibrated_ceiling303/cloudcompare_export/wall_geometry_analysis/marker98_measurement_anchor_audit_20260911.json).

## Сравнение native rigid

GT-нормализация в этом сравнении не используется. Сравниваются только перпендикулярные расстояния от marker 98 до native fitted inner lines.

| Стена | Ручной anchor | 303 native | ошибка 303 | 313 native | ошибка 313 |
|---|---:|---:|---:|---:|---:|
| west | 11.50..11.62 | 11.592 | `-0.028..+0.092` | 11.598 | `-0.022..+0.098` |
| east | 4.770 | 4.665 | `-0.105` | 4.629 | `-0.141` |
| south | 3.495 | 3.573 | `+0.078` | 3.531 | `+0.036` |
| north | 3.045 | 2.975 | `-0.070` | 3.000 | `-0.045` |

Суммы:

- ручной EW interval: `16.27..16.39 м`; 303: `16.257 м`, 313: `16.228 м`;
- ручной NS: `6.540 м`; 303: `6.548 м`, 313: `6.530 м`.

Если временно считать midpoint west=`11.56 м`, а остальные три ручные величины точными, RMSE получается примерно `7.6 см` для 303 и `7.8 см` для 313; MAE — `7.1 см` и `6.5 см` соответственно. Эти метрики дают разных формальных победителей и не имеют статистического смысла без допусков east/south/north.

## Вывод о ceiling height

Однозначно выбрать `3.03` или `3.13 м` по этим четырём anchor-измерениям нельзя. Оба варианта попадают в западный plausible interval. 303 ближе по EW-сумме к ручному интервалу и немного лучше по RMSE midpoint-а; 313 лучше по south/north и MAE. Разница marker-положения между вариантами около 5 см, но это может быть нормальным эффектом пересечения лучей с разными ceiling planes.

Следовательно, ручной anchor не подтверждает утверждение «313 точно лучше» или «303 точно лучше». Для выбора высоты нужны допуски ручных east/south/north измерений либо независимая геодезическая привязка нескольких меток.

## Обобщаемость

Общие части метода: координаты `x,y,z` PLY, автоматический dominant map/marker plane separation, rigid marker correspondence, Manhattan-frame estimate, robust line fitting, density-derived line tolerance, abstention при отсутствии второй грани. Для post-hoc wall distance camera/LiDAR extrinsics уже не нужны — они важны upstream, когда строятся `marker_points.json`.

Внешние/DePHAN-зависимые предположения: текущий CloudCompare PLY вручную очищен; неизвестна происхождение каждой точки, поэтому occupancy/noise band нельзя гарантированно трактовать как физическую толщину стены. Метод ожидает примерно прямоугольную Manhattan-комнату. Map scale и SLAM trajectory наследуются от Cartographer и этим анализом не калибруются. Marker ID 98 требует marker JSON/CSV; если его нет в PLY subset, расстояние корректно остаётся unavailable.

В `ros-data/results` нет экспортов Pandar/RPLIDAR в формате `result.ply`. Для настоящего cross-lidar теста нужен полный occupied map cloud, marker points со стабильными ID в той же системе координат, единицы/resolution, sensor/frame metadata и отметка о ручной очистке. Одного `map.pgm` или изображения недостаточно для проверки marker correspondence и физических inner/outer граней.

Синтетические тесты покрывают rotated/translated cloud, шум, проёмы, внутренние точки, отсутствие outer edge, partial marker subset и изменение density/noise/resolution. Production map/marker pipeline не изменялся.
