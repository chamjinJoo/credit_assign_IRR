import json
import sys
from pathlib import Path

from configs.config import Config
from not_for_submit.figures.tab2_smax_checkpoint_eval_utils import evaluate_run


def main() -> None:
    payload_path = Path(sys.argv[1]).resolve()
    payload = json.loads(payload_path.read_text())
    run_row = dict(payload["run_row"])
    run_row["config"] = Config.from_dict(run_row["config"])

    result = evaluate_run(
        run_row,
        checkpoint_choice=payload["checkpoint_choice"],
        device="cpu",
        eval_episodes=int(payload["eval_episodes"]),
        eval_seed=int(payload["eval_seed"]),
        deterministic=bool(payload["deterministic"]),
    )
    Path(payload["output_path"]).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
