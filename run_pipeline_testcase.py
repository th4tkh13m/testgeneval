import os
import argparse
import subprocess
from utils.data import Data
from utils.console import console


def main(args):

    console.log(
        "NOTE: Make sure you have built the docker images for the appropriate dataset"
    )

    data_suf = args.dataset.split("/")[-1]
    model_suf = args.model.split("/")[-1]

    if model_suf == "Meta-Llama-3.1-405B-Instruct":
        args.model = model_suf

    # print(
    #     f"Running pipeline for {args.model} with pass@{args.num_samples_full} (full) and pass@{args.num_samples_completion} (completion) on {data_suf}"
    # )

    base_dir = os.path.join(os.path.abspath(args.results_dir), data_suf)
    print(base_dir)
    os.makedirs(base_dir, exist_ok=True)

    log_dir = os.path.join(base_dir, "data_logs", model_suf)
    os.makedirs(log_dir, exist_ok=True)

    pred_dir = os.path.join(base_dir, "preds")
    os.makedirs(pred_dir, exist_ok=True)

    pred_output_filename = f"{model_suf}__{data_suf}__{args.temperature}__test.jsonl"
    print(pred_output_filename)
    preds_file = os.path.join(pred_dir, pred_output_filename)

    if os.path.exists(preds_file) and args.rerun_preds:
        os.remove(preds_file)

    # Load the dataset
    dataset = Data(
        data_name=args.dataset,
        save_path=args.data_path,
        console=console,
        data_path=None,
    )
    dataset.load_raw_data()
    dataset.process_raw_data()

    if args.debug:
        console.log("Debug mode on")

    # Run LLM translate
    # if not args.skip_translate:
    #     dataset.run_llm_translate()

    # Run evaluation
    # dataset.extract_branch()
    eval_cmd = [
        "python",
        "run_eval_testcase.py",
        "--log_dir",
        log_dir,
        "--num_processes",
        str(args.num_processes),
        "--namespace",
        args.namespace,
        "--repo",
        args.repo,
        "--data_path",
        os.path.join(args.data_path, f"{data_suf}.jsonl"),
    ]
    if args.debug:
        eval_cmd.append("--debug")
    subprocess.run(eval_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the pipeline testcase")
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to use",
        required=True,
        default="kjain14/testgenevallite",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="directory to save the results",
        required=True,
        default="./results",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="LLM to translate the test cases",
        required=False,
        default="Qwen/CodeQwen1.5-7B-Chat",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        help="Docker repository namespace",
        required=False,
        default="aorwall",
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Repository name",
        required=False,
        default="astropy/astropy",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Path to the data",
        required=False,
        default="./data",
    )
    parser.add_argument(
        "--temperature", type=float, help="(Optional) Model temperature", default=0.2
    )
    parser.add_argument(
        "--num_processes", type=int, help="Number of processes to run", default=1
    )
    parser.add_argument(
        "--skip_translate", action="store_true", help="(Optional) Skip LLM translation"
    )
    parser.add_argument(
        "--rerun_eval", action="store_true", help="(Optional) Skip LLM translation"
    )
    parser.add_argument(
        "--skip_mutation", action="store_true", help="(Optional) Skip LLM translation"
    )
    parser.add_argument("--debug", action="store_true", help="(Optional) Debug mode")
    args = parser.parse_args()
    main(args)
