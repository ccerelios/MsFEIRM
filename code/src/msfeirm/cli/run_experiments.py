"""Run one formal training slice or one evaluation-only variable-size experiment."""

import argparse
import json

from msfeirm.experiments import run_one
from msfeirm.variable_experiments import evaluate_one
from msfeirm.runtime import require_project_virtualenv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiments/chapter4.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--experiment-id", help="Evaluate one non-primary experiment without training")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate the first pending non-primary experiment")
    parser.add_argument("--max-runs", type=int, choices=(1,), default=1)
    parser.add_argument("--max-wall-time-min", type=float, help="Optional smaller training slice budget")
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    if args.experiment_id is not None or args.evaluate_only:
        if args.init_only or args.max_wall_time_min is not None:
            parser.error("Evaluation-only mode does not accept init-only or a training budget")
        result = evaluate_one(args.manifest, experiment_id=args.experiment_id, run_id=args.run_id)
    else:
        result = run_one(args.manifest, run_id=args.run_id, max_wall_time_min=args.max_wall_time_min, init_only=args.init_only)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
