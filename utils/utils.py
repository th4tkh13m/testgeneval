import re
from typing import Tuple, List, Dict


def extract_preamble_classes_and_functions(code: str) -> Tuple[str, list, list]:
    class_pattern = re.compile(
        r"(^(\s*@[\w\.\(\)\', ]+\s*)*^\s*class ([\w]+)\([^)]+\):)", re.MULTILINE
    )
    # Capture methods with or without decorators
    test_method_pattern = re.compile(
        r"(^(\s*@.*\s*)*^\s*def\s+test\w+\(.*\):)", re.MULTILINE
    )

    # Capture functions with or without decorators
    test_function_pattern = re.compile(
        r"(^(\s*@.*\s*)*^\s*def\s+test\w+\(.*\):)", re.MULTILINE
    )

    preamble = ""
    classes = []
    test_functions = []

    current_position = 0

    def extract_class_body(code: str, start_index: int) -> Tuple[str, int]:
        """
        Extracts the body of a class from the given code starting from the specified index.
        Returns the class body and the end index of the class body.
        """
        if not code or start_index < 0 or start_index >= len(code):
            raise ValueError("Invalid code or start index")

        # Split the code into lines
        lines = code[start_index:].split("\n")
        class_body_lines = []

        # Find the starting indentation level of the class definition
        class_start_line = lines[0]
        start_indent = len(class_start_line) - len(class_start_line.lstrip())

        inside_multiline_comment = False
        end_index = start_index
        for i, line in enumerate(lines[1:], start=1):
            stripped_line = line.strip()
            current_indent = len(line) - len(line.lstrip())

            # Handle multiline comments or docstrings
            if stripped_line.startswith('"""') or stripped_line.startswith("'''"):
                if inside_multiline_comment:
                    inside_multiline_comment = False
                else:
                    inside_multiline_comment = True

            if not inside_multiline_comment:
                # Stop when we reach a line with less indentation than the class definition
                if current_indent <= start_indent and stripped_line:
                    break

            # Add lines that are part of the class body
            class_body_lines.append(line)
            # Update the end index to the current line end
            end_index = start_index + len("\n".join(lines[: i + 1])) + 1

        return code[start_index:end_index], end_index

    while current_position < len(code):
        class_match = class_pattern.search(code, current_position)
        method_match = test_function_pattern.search(code, current_position)

        if class_match and (
            not method_match or class_match.start() < method_match.start()
        ):
            class_name = class_match.group(0)
            class_body, end_idx = extract_class_body(code, class_match.end())
            current_position = end_idx

            methods = []
            class_prefix = class_name
            set_prefix = False
            for method_match in test_method_pattern.finditer(class_body):
                method_name = method_match.group()
                method_start = method_match.start()
                if not set_prefix:
                    class_prefix = class_name + class_body[:method_start]
                    set_prefix = True
                next_method = test_method_pattern.search(
                    class_body, method_start + len(method_name)
                )
                method_body = (
                    class_body[method_start : next_method.start()]
                    if next_method
                    else class_body[method_start:]
                )
                methods.append((method_name, method_body))

            if methods:
                classes.append((class_prefix, methods, class_match.start()))
            else:
                preamble += class_name + class_body

        elif method_match:
            function_name = method_match.group(0)
            start_idx = method_match.start()
            next_function = test_function_pattern.search(
                code, start_idx + len(function_name)
            )
            function_body = (
                code[start_idx : next_function.start()]
                if next_function
                else code[start_idx:]
            )
            test_functions.append((function_body, start_idx))
            current_position = method_match.end()

        else:
            break

    if classes and test_functions:
        preamble = code[: min(classes[0][2], test_functions[0][1])]
    else:
        preamble = (
            code[: classes[0][2]]
            if classes
            else code[: test_functions[0][1]] if test_functions else code
        )

    return preamble.strip(), classes, test_functions


def indent_text(text, indent_level):
    return "\n".join(
        " " * indent_level + line if line.strip() else line for line in text.split("\n")
    )


def postprocess_tests(
    repo: str,
    preamble: str,
    class_name: str,
    methods: List[Tuple[str, str]],
    test_cases: Dict[str, str],
) -> Dict[str, str]:
    django_repo = repo == "django/django"

    def needs_django_harness(preamble):
        no_django_test = "TestCase" not in preamble
        no_unittest = "unittest" not in preamble
        no_simple_test_case = "SimpleTestCase" not in preamble
        return no_django_test and no_unittest and no_simple_test_case

    if django_repo and needs_django_harness(preamble):
        preamble = "from django.test import SimpleTestCase\n" + preamble
        preamble += "\n\nclass TestsHarness(SimpleTestCase):\n"
        added_class = True
    else:
        added_class = False

    test_id = len(test_cases.keys())
    print(f"Processing with {len(methods)} methods, id begins at {test_id}")
    for method_name, test_case in methods:
        if django_repo and added_class:
            if "(self):" not in test_case:
                test_case = test_case.replace("():", "(self):", 1)

        class_content = f"{class_name}\n{test_case}\n"
        test_content = preamble + "\n\n" + class_content

        test_cases[f"test_case_{test_id}"] = test_content
        print(f"Added test case {test_id}")
        test_id += 1

    return test_cases


def postprocess_functions(
    repo: str,
    preamble: str,
    test_functions: List[Tuple[str, str]],
    test_cases: Dict[str, str],
) -> Dict[str, str]:
    django_repo = repo == "django/django"

    def needs_django_harness(preamble):
        no_django_test = "TestCase" not in preamble
        no_unittest = "unittest" not in preamble
        no_simple_test_case = "SimpleTestCase" not in preamble
        return no_django_test and no_unittest and no_simple_test_case

    added_class = False
    if django_repo and needs_django_harness(preamble):
        preamble = "from django.test import SimpleTestCase\n" + preamble
        class_wrapper_start = "\n\nclass TestsHarness(SimpleTestCase):\n"
        preamble += class_wrapper_start
        added_class = True

    test_id = len(test_cases.keys())

    class_content = ""
    for test_function, start in test_functions:
        if django_repo and added_class:
            if "(self):" not in test_function:
                test_function = test_function.replace("():", "(self):", 1)
            test_content = preamble + "\n\n" + indent_text(test_function, 4)
        else:
            test_content = preamble + "\n\n" + test_function
        test_cases[f"test_case_{test_id}"] = test_content
        test_id += 1

    return test_cases
