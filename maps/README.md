# Reference-карты

- `dephan_20260922_corrected_run_170557` — принятая историческая DePHAN-карта;
- `pandar_20260914_cartographer_best` — историческая Pandar40P-карта.

Это сохранённые PBStream/occupancy с собственными marker-98 reference и
калибровкой в `../config/lidar_profiles.json`. Они нужны для воспроизведения
старых запусков, не являются картой произвольного нового помещения.
Новые карты передавайте внешним deploy-каталогом через `--map-dir`;
этот каталог создаёт `fliftbot_map_build`, копировать его сюда не обязательно.
Старый DePHAN snapshot и Crocus без полного calibration-контракта перенесены
в `../../tools/fliftbot_lidar_nav_archive/20261030/`.
