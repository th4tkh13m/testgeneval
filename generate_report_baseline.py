# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import json

from swebench_docker.swebench_utils import (
    get_eval_reports_for_dir,
    get_instances,
    get_model_eval_summary,
    get_model_report,
)
from swebench_docker.utils import get_eval_refs


def generate_report(swe_bench_tasks: str, log_dir: str, output_dir: str):
    instances = get_eval_refs(swe_bench_tasks)

    report_net = get_eval_reports_for_dir(
        log_dir, instances, verbose=False, raw_only=True, model_name="baseline"
    )

    with open(f"{output_dir}/baseline_full.json", "w") as f:
        f.write(json.dumps(report_net, indent=4))

    summary = get_model_eval_summary(
        predicts_path="",
        swe_bench_instances=instances,
        eval_dir=log_dir,
        model_name="baseline",
        is_baseline=True,
    )

    with open(f"{output_dir}/baseline_summary.json", "w") as f:
        f.write(json.dumps(summary, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log_dir", type=str, help="Path to log directory", required=True
    )
    parser.add_argument(
        "--swe_bench_tasks",
        type=str,
        help="Path to dataset file or HF datasets name",
        required=True,
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path to output directory", required=True
    )
    args = parser.parse_args()
    generate_report(**vars(args))
