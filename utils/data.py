import os
import json
from utils.utils import (
    extract_preamble_classes_and_functions,
    postprocess_functions,
    postprocess_tests,
)
from datasets import load_dataset, load_from_disk, concatenate_datasets

# typing
from rich.console import Console
from typing import Dict


class Data(object):

    def __init__(
        self,
        data_name: str,
        data_path: str = None,
        save_path: str = None,
        console: Console = None,
    ) -> None:

        if (
            data_name not in ["kjain14/testgeneval", "kjain14/testgenevallite"]
            and data_path is None
        ):
            raise ValueError(
                "Invalid data name without data path, please provide data path"
            )
        self.data_name = data_name
        self.data_path = data_path
        self.save_path = save_path
        self.console = console

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

    def load_raw_data(self) -> None:
        if self.data_path is None:
            if self.data_name == "kjain14/testgeneval":
                train_dataset = load_dataset(self.data_name, split="train")
                valid_dataset = load_dataset(self.data_name, split="dev")
                test_dataset = load_dataset(self.data_name, split="test")
                dataset = concatenate_datasets(
                    [train_dataset, valid_dataset, test_dataset]
                )
            elif self.data_name == "kjain14/testgenevallite":
                valid_dataset = load_dataset(self.data_name, split="dev")
                test_dataset = load_dataset(self.data_name, split="test")
                dataset = concatenate_datasets([valid_dataset, test_dataset])
        else:
            dataset = load_from_disk(self.data_path)
        self.dataset = dataset
        self.console.log(f"Data {self.data_name} loaded successfully")

    def process_raw_data(self) -> None:
        data_list = []
        num_test_cases = 0
        num_data_point = 0
        for i in range(len(self.dataset)):
            data = self.dataset[i]
            processed_data = self.process_one_raw(data)
            data_list.append(processed_data)
            num_data_point += 1
            num_test_cases += len(processed_data["test_cases"])
        # save to jsonl
        with open(
            os.path.join(self.save_path, f"{self.data_name.split('/')[-1]}.jsonl"), "w"
        ) as f:
            for item in data_list:
                f.write(json.dumps(item) + "\n")
        self.console.log(
            f"Num data points: {num_data_point}, num test cases: {num_test_cases}"
        )
        self.console.log(f"Data {self.data_name} processed successfully")

    def process_one_raw(self, data: Dict) -> Dict:
        repo = data["repo"]
        commit_id = data["base_commit"]
        version = data["version"]
        instance_id = data["instance_id"]
        patch = data["patch"]
        test_patch = data["test_patch"]
        preds_context = data["preds_context"]
        code_src = data["code_src"]
        test_src = data["test_src"]
        code_file = data["code_file"]
        test_file = data["test_file"]
        local_imports = data["local_imports"]
        idx = data["id"]
        baseline_covs = data["baseline_covs"]

        preamble, classes, test_functions = extract_preamble_classes_and_functions(
            code=test_src
        )
        test_cases = {}
        for class_name, methods, start in classes:
            test_cases = postprocess_tests(
                repo=repo,
                preamble=preamble,
                class_name=class_name,
                methods=methods,
                test_cases=test_cases,
            )

        test_cases = postprocess_functions(
            repo=repo,
            preamble=preamble,
            test_functions=test_functions,
            test_cases=test_cases,
        )
        branches = {}
        for key in test_cases.keys():
            branches[key] = []
        return {
            "repo": repo,
            "base_commit": commit_id,
            "version": version,
            "instance_id": instance_id,
            "patch": patch,
            "test_patch": test_patch,
            "preds_context": preds_context,
            "code_src": code_src,
            "test_src": test_src,
            "code_file": code_file,
            "test_file": test_file,
            "local_imports": local_imports,
            "id": idx,
            "baseline_covs": baseline_covs,
            "test_cases": test_cases,
            "branches": branches,
        }

    # def get_branches(self):
    #     return self.dataset["base_commit"].unique()
