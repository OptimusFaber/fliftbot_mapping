# Постобработка run-каталогов

Этот каталог не запускает робота и не записывает датчики. Он принимает готовую
запись из `ros-data/recordings/{sim,real}/` и строит глобальную карту.

Для надёжной записи ROS-топиков и повторяемой офлайн-обработки Cartographer
используйте [`docs/cartographer_offline.md`](../../docs/cartographer_offline.md):
`record_cartographer_run.sh`, `validate_cartographer_recording.py` и
`offline_cartographer.sh`.

```bash
./fliftbot_mapping/scripts/mapping/map_check_inputs.sh --source sim
./fliftbot_mapping/scripts/mapping/map_build.sh \
  --run-dir RUN_DIR --method improved --source sim --cameras up
./fliftbot_mapping/scripts/mapping/map_eval_accuracy.sh --run-dir RUN_DIR
```

Дополнительные команды:

- `map_snapshot.sh` — построить карту по доступной части записи;
- `map_factor_diag.sh` — диагностика factor/trajectory graph;
- `map_check_inputs.sh` проверяет live ROS-топики (для записи достаточно
  проверить наличие `horizon/*.npz` и `encoder.jsonl` в run-каталоге).

Методы `clouds`, `odom`, `improved`, `factor`, `factor_no_loops`,
`odom_fixed_v2` и `submap` перечислены с назначением и командами в корневом
[`README.md`](../../README.md). Там же описана валидация по ground truth и
точный запуск ROS 2 Cartographer; внешний baseline использует
[`docs/external_slam.md`](../../docs/external_slam.md).

Симуляционный workflow: [`../../../fliftbot_sim/README.md`](../../../fliftbot_sim/README.md).
Реальный робот: [`../../../fliftbot_real/scripts/mapping/README.md`](../../../fliftbot_real/scripts/mapping/README.md).
