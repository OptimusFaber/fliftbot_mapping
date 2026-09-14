# Документация mapping pipeline

`fliftbot_mapping` содержит только platform-independent обработку готовых
записей: синхронизацию, deskew, scan matching, pose graph, фильтрацию карты,
рендеринг и оценку.

- `frozen/` — зафиксированные baseline и контрольные суммы;
- `mapping-package-migration.md` — история разделения пакетов;
- изображения и схемы — измерения геометрии, используемые алгоритмом;
- `references/` — внешние материалы.

Запуск Gazebo документирован в `../../fliftbot_sim/README.md`, физического
робота — в `../../fliftbot_real/README.md`.

Основная команда:

```bash
./fliftbot_mapping/scripts/mapping/map_build.sh --help
```
