from utils.loader import load_data
from utils.printer import dump_data
from os.path import join
import os
import subprocess as sp
import json
import argparse
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)

LOG_FILE = "jax-filter.log"

dtype_precision_dict = {
    "bfloat16": 0,
    "float16": 1,
    "float32": 2,
    "complex64": 2,
    "float64": 3,
    "complex128": 3,
}


def run_code(code):
    ret = {"ret": "Pass"}
    try:
        exec(code)
    except Exception as e:
        return "Fail", str(e)
    else:
        return "Success", ret["ret"]


def run(file_name):
    try:
        res = sp.run(["python", file_name], timeout=60, capture_output=True)
    except sp.TimeoutExpired:
        return "Timeout", ""
    else:
        if res.returncode == 0:
            return "Success", res.stdout.decode("utf-8")
        else:
            return "Fail", res.stderr.decode("utf-8")


def get_fn_code(code: str):
    code_lines = code.splitlines()
    st_idx = -1
    end_idx = -1
    for idx, line in enumerate(code_lines):
        if line.startswith("def fn("):
            st_idx = idx
        elif line.strip().startswith("return "):
            end_idx = idx
            break
    assert st_idx != -1 and end_idx != -1, "Error Format: No `fn` Code"
    return "\n".join(code_lines[st_idx : end_idx + 1]) + "\n"


def get_input_tensors(code: str):
    tensors = []
    for line in code.splitlines():
        if line.startswith('# {"name"'):
            # name, shape, dtype
            tensors.append(json.loads(line[2:]))
    return tensors


def get_tensor_code(code: str, tensors: list[dict]):
    def _get_code(name: str, type, code_lines):
        if "JAX_ARRAY" in type:
            return _get_array_code(name, code_lines)
        elif "JAX_SCALAR" in type:
            return [], _get_scalar_code(name, code_lines)
        else:
            assert 0, f"No such type: {type}"

    def _get_array_code(name: str, code_lines):
        create_code = []
        clone_code = ""
        clone_pattern = f"{name} = {name}_array."
        for idx, line in enumerate(code_lines):
            line = line.strip()
            if (
                line.startswith(f"{name}_array = jax.")
                and len(create_code) == 0
            ):
                create_code += [code_lines[idx - 1].strip(), line]
            elif line.startswith(clone_pattern) and clone_code == "":
                clone_code = line
            if len(create_code) and clone_code:
                return create_code, clone_code
        assert 0, f"Not find the code for {name}"

    def _get_scalar_code(name: str, code_lines):
        scalar_pattern = f"{name} = jax.numpy.array"
        for line in code_lines:
            line = line.strip()
            if line.startswith(scalar_pattern):
                return line
        assert 0, f"Not find the code for scalar {name}"

    code_lines = code.splitlines()
    create_codes = []
    clone_codes = []
    for tensor in tensors:
        c1, c2 = _get_code(tensor["name"], tensor["type"], code_lines)
        create_codes += c1
        clone_codes.append(c2)
    return "\n".join(create_codes + clone_codes) + "\n"


def comment_code(code: str):
    code_lines = code.splitlines()
    new_code = ""
    for line in code_lines:
        new_code += "# " + line + "\n"
    return new_code


def dtype_precision(dtype: str):
    if "int" in dtype:
        return 0
    for dt, precision in dtype_precision_dict.items():
        if dt in dtype:
            return precision
    assert 0, f"No such dtype: {dtype}"


def is_high_to_low_precision(inputs, output):
    def _check(output):
        for input in inputs:
            if dtype_precision(input["dtype"]) > dtype_precision(
                output["dtype"]
            ):
                return True
        return False

    if output is None:
        assert 0, "None in output"
        return False
    elif isinstance(output, list):
        for o in output:
            if _check(o):
                return True
        return False
    else:
        return _check(output)


def is_nan_code(code: str):
    if (
        "nan," in code
        or "nan]" in code
        or "nan)" in code
        or "[nan" in code
        or "(nan" in code
        or "nan+" in code
        or "nanj," in code
    ):
        return True
    else:
        return False


def analyze_api(api: str, results: list[dict], output_file):
    def true_positive(res: dict):
        return (
            res["status"] == "Success"
            and not res["has_nan"]
            and not res["is_high_to_low"]
            and res["is_diff"]
        )

    num_tp = 0
    num_fp = 0
    for res in results:
        if true_positive(res):
            dump_data(f"{api}, {res['id']}\n", output_file, "a")
            num_tp += 1
        else:
            num_fp += 1
    return num_tp, num_fp


def filter(data_dir: Path):
    def filter_dir(target_dir: Path, csv_file: Path, is_ND):
        if not os.path.exists(target_dir):
            return

        status_count = {"Timeout": 0, "Success": 0, "Fail": 0}
        ret_count = {
            "NaN": 0,
            "Precision": 0,
            "Non-Diff": 0,
            "Pass": 0,
            "NaN-error": 0,
        }
        for file_name in os.listdir(target_dir):
            if file_name.startswith(DIFF_PREFIX):
                continue
            codelines = load_data(target_dir / file_name, multiline=True)
            input_list = eval(codelines[-1])
            codelines += [
                "from helper_jax import Filter",
                # f"print(Filter(fn, ({','.join(input_list)},), is_ND={is_ND}))",
                f"ret['ret'] = Filter(fn, ({','.join(input_list)},), is_ND={is_ND})",
            ]
            code = "\n".join(codelines)

            dump_data(code, RUN_FILE)
            # status, ret = run(RUN_FILE)
            status, ret = run_code(code)
            code += f"\n# {status} {ret}\n"
            dump_data(code, target_dir / (DIFF_PREFIX + file_name))

            status_count[status] += 1
            if status == "Success":
                # ret_type = ret.split("\n")[-2]
                ret_type = ret
                dump_data(f"{target_dir}: {ret_type}\n", LOG_FILE, "a")
                ret_count[ret_type] += 1
            else:
                dump_data(f"{target_dir}: {status}\n{ret}\n", LOG_FILE, "a")

        data = [str(i) for i in status_count.values()] + [
            str(i) for i in ret_count.values()
        ]
        dump_data(
            f"{api_name}, {','.join(data)}\n",
            csv_file,
            "a",
        )

    RUN_FILE = "temp-jax-filter.py"
    DIFF_PREFIX = "diff-"

    # first order
    REV_FWD_CSV = data_dir / "filter-grad-rev-fwd.csv"
    ND_CSV = data_dir / "filter-grad-nd.csv"
    # second order
    GRAD_REV_FWD_CSV = data_dir / "filter-second-grad-rev-fwd.csv"
    GRAD_ND_CSV = data_dir / "filter-second-grad-nd.csv"

    dump_data("", REV_FWD_CSV)
    dump_data("", ND_CSV)
    dump_data("", GRAD_REV_FWD_CSV)
    dump_data("", GRAD_ND_CSV)

    for api_name in sorted(os.listdir(data_dir)):
        if not api_name.startswith("jax."):
            continue

        # {api}/grad-rev-fwd
        # {api}/grad-nd
        print(api_name)
        api_dir = data_dir / api_name
        filter_dir(api_dir / "grad-rev-fwd", REV_FWD_CSV, False)
        filter_dir(api_dir / "grad-nd", ND_CSV, True)

        filter_dir(api_dir / "grad" / "grad-rev-fwd", GRAD_REV_FWD_CSV, False)
        filter_dir(api_dir / "grad" / "grad-nd", GRAD_ND_CSV, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jax Auto-gradient Filter")

    parser.add_argument(
        "--target",
        type=str,
        default="union",
        help="The output dir (default: union)",
    )
    args = parser.parse_args()
    target = args.target

    dump_data("", LOG_FILE)

    data_dir = Path("../output-ad/jax", target)
    filter(data_dir)
