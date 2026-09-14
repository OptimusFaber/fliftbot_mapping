# Audit: `result.ply` wall geometry and marker 98

Дата: 2026-09-11. Аудит выполнен для двух сохранённых CloudCompare облаков:

- `markers_lidar_tracking_calibrated_ceiling303/cloudcompare_export/result.ply`;
- `markers_lidar_tracking_calibrated/cloudcompare_export/result.ply` (ceiling 3.13 m).

Машинные результаты находятся рядом с каждым облаком в `wall_geometry_analysis/wall_geometry_audit.json`. Сводка двух вариантов — `ceiling303/.../wall_geometry_audit_comparison.json`.

## Что проверено

`analyze_result_ply_wall_geometry.py` разделяет крупный map z-cluster и малый marker z-cluster автоматически; не требует фиксированного `z=0.12`. Для полного облака соответствие marker points восстанавливается по инвариантным distance fingerprints, затем уточняется rigid Kabsch fit. Для частичного облака используется two-point rigid RANSAC и one-to-one assignment. Marker 98 не подменяется другой точкой.

Стены выделяются по Manhattan-направлению, найденному из локальных соседских направлений и уточнённому робастным line residual score. Для каждой стороны берутся только центральные 76% продольного участка. Кандидаты параллельных граней ищутся нормальным Hough scan и уточняются Theil–Sen. `inner` выбирается как поддержанная грань со стороны помещения; `outer` принимается только при независимой поддержке и покрытии; `median` является ровно арифметическим средним двух fitted lines. Если второй край не наблюдается, результат `null`, а линия/стрелка не рисуются.

В PNG стрелка проводится от marker 98 к каждой реально доступной линии, подпись содержит расстояние. Для отсутствующей грани отображается только явное `unavailable` в панели.

## Validated facts

Для обоих текущих PLY marker 98 найден. Rigid correspondence имеет RMSE порядка `1e-6 м`, то есть округление/порядок строк CloudCompare не объясняет ошибку положения метки.

В обоих вариантах расстояние между двумя наблюдаемыми параллельными кандидатами правой стены нестабильно: для 303 baseline outer не принят; для 313 baseline outer принят, но при изменении окна сегмента availability меняется. Это означает, что вторую правую физическую грань нельзя считать валидированной без дополнительного облака/измерения.

Входные метрики самой локализации marker 98:

| Вариант | spread median | ray residual | p95 spread | p95 ray residual |
|---|---:|---:|---:|---:|
| ceiling 3.03 | 0.0210 м | 0.0206 м | 0.0917 м | 0.0898 м |
| ceiling 3.13 | 0.0299 м | 0.0290 м | 0.0803 м | 0.0790 м |

Это оценка повторяемости лучей, а не независимая абсолютная accuracy.

## Native, rigid-only geometry

Текущая native оценка не использует GT и не масштабируется к GT. В зависимости от версии:

| Вариант | native inner X | native inner Y | marker 98 native | right outer |
|---|---:|---:|---|---|
| 303 | 16.252 м | 6.548 м | (11.587, 3.573) м | unavailable |
| 313 | 16.230 м | 6.520 м | (11.591, 3.520) м | fitted, but sensitivity-dependent |

Вторая существенная проверка — sensitivity по `central_quantile={0.08,0.12,0.16,0.20}` и `wall_side_quantile={0.30,0.35,0.40}` при фиксированной сцене. Диапазоны из JSON:

- 303: X `16.248..16.256 м`, Y `6.457..6.565 м`, marker Y `3.528..3.590 м`;
- 313: X `16.230..16.297 м`, Y `6.472..6.577 м`, marker X `11.591..11.635 м`, marker Y `3.518..3.575 м`.

Следовательно, X достаточно устойчив только на уровне сантиметров для выбранного окна, а Y и расстояния до горизонтальных стен чувствительнее к продольному окну. Это uncertainty метода выделения стены, не ошибка GT.

Разность native marker 98 между ceiling 3.03 и 3.13: примерно `0.053 м`, главным образом по Y. Это ожидаемо для пересечения камеры̆ных лучей с плоскостями потолка, и не доказывает ошибку Cartographer.

## GT: reference, not absolute truth

`floorplan.json` задаёт `room_size_m=[16.465,6.56]`. Происхождение указано в `flifltbot/mapping/real_room_gt.py`: X — ручная цепочка измерений верхней стены, сумма `1646.5 см`; Y — ручное измерение `656 см`. `floorplan.json` прямо помечен как `2026-08-13 remeasure`.

Растровый `floorplan.png` имеет padding origin `[-0.5,-0.5]`, размер ячейки `0.05 м`, extent `17.5×8.05 м`; это не следует смешивать с физическим shell `16.465×6.56 м`. В `ceiling_labels.json` marker 98 задан ручными расстояниями `478.0 см` от восточной стены и `352.5 см` от южной, то есть `[11.685,3.525] м`. Это отдельные tape measurements с неизвестными допусками и возможной несогласованностью углов/граней.

Сравнение с GT допускает только диагностическую интерпретацию:

- native X меньше `16.465 м` примерно на `0.17..0.24 м` в зависимости от варианта/окна;
- native Y отличается примерно на `0.01..0.04 м`;
- marker 98 отличается от ручного GT примерно на `0.09..0.11 м` в baseline native frame.

Эти числа нельзя назвать абсолютной ошибкой алгоритма, пока не заданы допуски для цепочки Ox, положения внутренних граней стены, угла помещения и marker-98 tape measurement. GT-normalized PNG явно применяет X/Y scale и предназначен только для просмотра.

## Generalization audit and limitations

В исходной версии были скрытые scene assumptions: фиксированное разделение `z`-плоскостей, fixed metre tolerances, fixed central quantiles, прямоугольник с четырьмя сторонами и обязательный marker plane. Сейчас z separation и metre tolerances выводятся из median nearest-neighbour spacing, параметры окна доступны CLI, marker subset поддержан, а отсутствующие грани приводят к abstention. `--gt-floorplan` остаётся обязательным CLI для совместимости, но native расчёт GT не использует.

Остающиеся ограничения:

- алгоритм ожидает dominant planar map cloud и четыре Manhattan-like sides; произвольный polygon или наклонные стены требуют отдельного режима;
- заполненная полоса noisy occupied points не всегда идентифицирует физическую толщину стены; right outer toggling на 313 это показывает;
- `wall_score` использует histogram/Hough heuristic, поэтому bootstrap/sensitivity должны считаться частью uncertainty;
- полная статистическая bootstrap с независимыми повторными сканами невозможна по одному CloudCompare PLY; resampling map points не является заменой acquisition bootstrap.

Других `result.ply` в `ros-data/results` на дату аудита не найдено; поэтому generalization проверена синтетическими perturbations, а не Pandar/RPLIDAR PLY. Production mapping/marker pipeline не изменялся.

## Synthetic tests

`fliftbot_mapping/tests/test_result_ply_wall_geometry.py` проверяет:

1. rotated/translated rectangle с шумом, проёмами, углами и внутренним clutter;
2. две физические грани и arithmetic median;
3. отсутствие outer edge с корректным abstention;
4. partial marker subset и rigid correspondence;
5. marker-independent Manhattan orientation.

`pytest` в окружении отсутствует; все четыре test-функции запущены напрямую через Python и прошли (`all pass`). Скрипт также прошёл `py_compile`; оба текущих PLY были обработаны и PNG просмотрены.

## Итог

Вычисление marker 98 и inner distances корректно как воспроизводимая native estimate при оговорённых residuals. GT-normalized картинки не являются доказательством истинного масштаба. Outer/median следует использовать только там, где diagnostics подтверждает независимую вторую грань; для правой стены 303 это `unavailable`, а 313 требует дополнительной проверки из-за sensitivity toggle. Следующий надёжный шаг — сохранить исходное полное map cloud с цветами/источниками точек или выполнить повторный scan wall faces; это позволит различить физическую толщину стены и occupancy/noise band.
