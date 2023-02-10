import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import subprocess
import time
from pathlib import Path

import tensorflow as tf
from termcolor import colored

from classes.database import TFDatabase
from classes.library_def import tf_lib_def
from constant.keys import *
from utils.loader import load_data
from utils.printer import dump_data
from utils.utils import if_skip_api
from tf_fuzz import fuzz_all


def load_log_and_put(output_dir: Path, target_dir: Path, api_name):
    data = load_data(output_dir / "temp.id")
    if data is None:
        data = api_name
    with open(target_dir / f"{api_name}.py", "w") as f:
        f.write(data)

if __name__ == "__main__":
    #region args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num',
        type=int,
        default=10,
        help="The number of mutants for each API (default: 1000)")
    parser.add_argument(
        '--max_api',
        type=int,
        default=-1,
        help="The number of API that will be tested (default: -1, which means all API)")
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--expr_dir', 
        type=str,
        default="expr_outputs"
    )
    parser.add_argument(
        '--run_dirname', 
        type=str,
        default="test"
    )
    parser.add_argument(
        '--restart',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--device', 
        type=str,
        default="gpu"
    )

    args = parser.parse_args()
    print(args)
    #endregion

    num = args.num
    max_api = args.max_api
    device = args.device
    expr_dir = args.expr_dir
    run_dirname = args.run_dirname

    expr_dir = Path(expr_dir)
    run_dir = expr_dir / run_dirname
    
    if args.restart:
        subprocess.run( ["rm", "-r", f"{run_dir.resolve()}"])
        
    log_dir = run_dir / "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    result_dir = os.path.join(run_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    crash_dir = os.path.join(run_dir, "crash")
    crash_dir = Path(crash_dir)
    os.makedirs(crash_dir, exist_ok=True)
    
    from classes.library_def import tf_lib_def
    tf_lib_def.load_apis(lazy=True)

    db = TFDatabase
    apis = db.get_api_list()
    apis.sort()
    if max_api != -1:
        apis = apis[:max_api]
    with open(expr_dir / "apis.txt", "w") as f:
        f.writelines([x + "\n" for x in apis])

    
    with open(log_dir / "config.log", "w") as f:
        f.write(str(args) + "\n")
    with open(log_dir / "gradtested.log", "w") as f:
        f.write("\n")
    tested_apis = []

    print("Total apis to test: ", len(apis))

    fuzz_all(num, apis, expr_dir, log_dir)
    tests_dir = expr_dir / "mutant"

    timeout = 300
    capture_output = False
    
    for api_name in apis:
        if if_skip_api(api_name): continue
        print(colored(api_name, "blue"))

        with open(log_dir / "gradtested.log", "a") as f:
            f.write(api_name + "\n")
        result_fp = os.path.join(result_dir, api_name + ".json")
        testfn = os.path.join(tests_dir, api_name + ".json")
        timefn = os.path.join(log_dir, "time.txt")
        
        st_time = time.time()
        run_args = [
                "python",
                "tf_api_worker.py",
                "--api_name", api_name,
                "--expr_dir", expr_dir,
                "--run_dirname", run_dirname,
                "--device", device,
        ]
        try:
            ret = subprocess.run(
                    run_args, timeout=timeout, shell=False, capture_output=True
                    )
        except subprocess.TimeoutExpired:
            dump_data(f"{api_name}\n", 
                    log_dir / "test-run-timeout.txt", mode='a')
        else:
            if ret.returncode != 0:
                dump_data(f"{api_name}\n", 
                        log_dir / "test-run-crash.txt", mode='a')
                error_msg = ""    
                if ret.stdout is not None:
                    error_msg = ret.stdout.decode("utf-8") + ret.stderr.decode(
                    "utf-8"
                )
                dump_data(
                    f"{api_name}\n{error_msg}\n\n",
                    log_dir / "crash.log",
                    "a",
                )
                load_log_and_put(log_dir, crash_dir, api_name)
            else:
                if ret.stdout is not None:
                    print(ret.stdout.decode("utf-8"))
        running_time = time.time() - st_time
        dump_data(f"{api_name}, {running_time}\n", timefn, "a")
