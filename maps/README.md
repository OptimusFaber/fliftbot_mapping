# Reference-карты

- `tver_agro_20261109` — комплект Тверь-Агро: occupancy из `base.jpeg`,
  исходный PBStream и калибровка запуска 20261109_141038.
  Выходная СК: X вправо, Y вверх на веб-карте, угол положительный против часовой
  стрелки от X (`y_axis_sign: 1`). Ноль — выбранный центр нижней левой колонны.
  Выбор: `./fliftbot_lidar_nav/nav.sh check-map --map-dir fliftbot_lidar_nav/maps/tver_agro_20261109`.

- `dephan_20260922_corrected_run_170557` — принятая историческая DePHAN-карта;
- `pandar_20260914_cartographer_best` — историческая Pandar40P-карта.

Это сохранённые PBStream/occupancy с собственными marker-98 reference и
калибровкой в `../config/lidar_profiles.json`. Они нужны для воспроизведения
старых запусков, не являются картой произвольного нового помещения.
Новые карты передавайте внешним deploy-каталогом через `--map-dir`;
этот каталог создаёт `fliftbot_map_build`, копировать его сюда не обязательно.
Старый DePHAN snapshot и Crocus без полного calibration-контракта перенесены
в `../../tools/fliftbot_lidar_nav_archive/20261030/`.
