"""Build the Chapter IV final report from saved experiment files."""

import argparse
import json
from pathlib import Path

from msfeirm.reporting import build_report
from msfeirm.runtime import project_output_path, require_project_virtualenv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("chapter4",), default="chapter4")
    parser.add_argument("--manifest", default="configs/experiments/chapter4.yaml")
    parser.add_argument("--from-inputs", type=Path)
    parser.add_argument("--output", default="reports/chapter4")
    args = parser.parse_args(argv)
    require_project_virtualenv()
    result = build_report(project_output_path(args.output), manifest_path=args.manifest, from_inputs=args.from_inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
