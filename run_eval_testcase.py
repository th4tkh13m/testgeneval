import argparse
import asyncio
import logging
import os
import json
from rich.pretty import pretty_repr
from swebench_docker.constants import KEY_BASELINES, KEY_ID, REPO_ID
from swebench_docker.run_docker import run_docker_evaluation
from swebench_docker.utils import get_test_tasks

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("run_evaluation_baseline")


async def main(
    data_path: str,
    res_path: str,
    namespace: str,
    log_dir: str,
    repo: str = None,
    timeout: int = 60,
    num_processes: int = -1,
    debug: bool = False,
):
    """
    Runs evaluation on predictions for each model/repo/version combination.
    """
    if not os.path.exists(log_dir) or not os.path.isdir(log_dir):
        raise ValueError("--log_dir must exist and point at a directory")
    os.chmod(log_dir, 0o777)

    tasks = get_test_tasks(data_path)
    if not isinstance(tasks, list):
        raise ValueError(f"Data from {data_path} must contain an array of tasks")

    print(f"Debug mode: {debug}")
    if debug:
        print(f"First task keys: {pretty_repr(tasks[0].keys())}")
        print(f"Number of tasks: {len(tasks)}")
        # exit()
    if repo is not None:
        tasks = [t for t in tasks if t[REPO_ID] == repo]
    if debug:
        print(f"Number of tasks after filtering by repo: {len(tasks)}")

    num_test_case = 0
    for task in tasks:
        num_test_case += len(task["test_cases"].keys())
    logger.info(
        f"# of task to evaluate: {len(tasks)}. # of test cases: {num_test_case}"
    )

    sem = asyncio.Semaphore(num_processes if num_processes > 0 else len(tasks))
    asyncio_tasks = []
    if debug:
        tasks = tasks[:1]
        test_case_keys = ["test_case_0"]

    task_dict = {task[KEY_ID]: task for task in tasks}

    for task_instance in tasks:
        if debug:
            print(f"An example of task_instance: {pretty_repr(task_instance.keys())}")
            # exit()
            for testcase in test_case_keys:

                async def run_docker_throttled(task_instance, testcase):
                    async with sem:
                        return await run_docker_evaluation(
                            task_instance,
                            namespace,
                            log_dir,
                            testcase,
                            timeout,
                            only_baseline=True,
                            verbose=True,
                            skip_mutation=True,
                        )

                task = asyncio.create_task(
                    run_docker_throttled(task_instance, testcase)
                )
                asyncio_tasks.append(task)
        else:
            for testcase in task_instance["test_cases"].keys():

                async def run_docker_throttled(task_instance, testcase):
                    async with sem:
                        return await run_docker_evaluation(
                            task_instance,
                            namespace,
                            log_dir,
                            testcase,
                            timeout,
                            only_baseline=True,
                            verbose=True,
                            skip_mutation=True,
                        )

                task = asyncio.create_task(
                    run_docker_throttled(task_instance, testcase)
                )
                asyncio_tasks.append(task)

    results = await asyncio.gather(*asyncio_tasks)
    for result in results:
        print(result)
        res, setting = result
        task_dict[result[KEY_ID]]["branches"][setting] = res["branches"][setting]

    with open(res_path, "w") as f:
        for item in task_dict.values():
            f.write(json.dumps(item) + "\n")
    print(f"Evaluation complete: {res['branches']}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--namespace", type=str, default="aorwall")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--num_processes", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(**vars(args)))
