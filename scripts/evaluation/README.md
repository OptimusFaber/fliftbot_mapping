# Evaluation scripts

Потолочные маркеры переносятся только методом V5. Команды для точек, подписей
декодера и offline-валидации описаны в
[`../../docs/ceiling_marker_localization_v5.md`](../../docs/ceiling_marker_localization_v5.md).

Скрипты оценки карт лежат непосредственно здесь. Старые разовые эксперименты
с жёстко заданными путями перенесены в `examples/legacy_experiments/` и не
считаются рабочими entrypoint-командами.

Новые результаты необходимо записывать в
`../ros-data/results/{sim|real}/DATE/SCENE_LIDAR_run_TIME/`.
