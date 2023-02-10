import argparse
from pathlib import Path
import subprocess as sp
import os
import shutil
import time
from classes.database import JaxDatabase
from utils.printer import dump_data
from utils.loader import load_data

JaxDatabase.database_config("127.0.0.1", 27017, "jax-test-unique")


def load_log_and_put(output_dir: Path, target_dir: Path, api_name):
    data = load_data(output_dir / "temp.py")
    dump_data(data, target_dir / f"{api_name}.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NablaFuzz for Jax")

    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="The number of mutants for each API or API pair (default: 1000)",
    )
    parser.add_argument(
        "--max_api",
        type=int,
        default=-1,
        help="The number of API that will be tested (default: -1, which means all API)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean the output dir (default: False)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output-ad",
        help="The output directory (default: 'output-ad')",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="The device (default: 'cpu')"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="The suffix of the output dir (default: '')",
    )

    args = parser.parse_args()
    print(args)

    if args.suffix != "":
        suffix = f"-{args.suffix}"
    else:
        suffix = ""

    output_dir = Path("..", args.output, "jax", f"union{suffix}")
    if args.clean and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    timeout_dir = output_dir / "run-timeout"
    crash_dir = output_dir / "run-crash"
    os.makedirs(timeout_dir, exist_ok=True)
    os.makedirs(crash_dir, exist_ok=True)

    # TEST
    timeout = 600
    max_api_number = args.max_api

    time_file = output_dir / "time.csv"
    api_list = JaxDatabase.get_api_list()

    i = 0
    for api_name in api_list:
        if api_name in [
            "jax._src.custom_derivatives.custom_jvp",
            "jax._src.custom_derivatives.custom_vjp",
        ]:
            continue

        i += 1
        if max_api_number != -1 and i > max_api_number:
            break
        print(f"{api_name}, {i} / {len(api_list)}")
        st_time = time.time()
        try:
            ret = sp.run(
                [
                    "python",
                    "jax_adtest.py",
                    "--api",
                    api_name,
                    "--num",
                    str(args.num),
                    "--dir",
                    output_dir,
                ],
                timeout=timeout,
                shell=False,
                capture_output=True,
            )
        except sp.TimeoutExpired:
            dump_data(
                f"{api_name}\n", output_dir / "test-run-timeout.txt", mode="a"
            )
            load_log_and_put(output_dir, timeout_dir, api_name)
            print("TIMEOUT\n")
        else:
            if ret.returncode != 0:
                dump_data(
                    f"{api_name}\n",
                    output_dir / "test-run-crash.txt",
                    mode="a",
                )
                error_msg = ret.stdout.decode("utf-8") + ret.stderr.decode(
                    "utf-8"
                )
                print(error_msg)
                dump_data(
                    f"{api_name}\n{error_msg}\n\n",
                    output_dir / "crash.log",
                    "a",
                )
                print("ERROR\n")
                load_log_and_put(output_dir, crash_dir, api_name)
            else:
                print(ret.stdout.decode("utf-8"))
        running_time = time.time() - st_time
        dump_data(f"{api_name}, {running_time}\n", time_file, "a")
