import argparse
import os

from tf_utils import BUGGY_RESULT_TYPES
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress TF warnings

from pathlib import Path
import random
from random import choice
from classes.oracles import OracleTF, ResultType
import sys
from os.path import join
from termcolor import colored
import configparser
import tensorflow as tf
import json

from classes.argument import ArgType
from classes.database import TFDatabase
from classes.tf_api import TFAPI, TFArgument
from utils.printer import dump_data
from utils.loader import load_data
from utils.grad_utils import count_results
from tf_grad_utils import run_and_check
from classes.tf_gradoracles import IntegratedGradientOracleTF

need_count = [
    "SUCCESS", 
    "GRAD_VALUE_MISMATCH",
    "VALUE_MISMATCH",

    "DIRECT_CRASH",
    "REV_CRASH",
    "FWD_CRASH",
    
    "REV_STATUS_MISMATCH",
    "REV_GRAD_FAIL",
    "FWD_GRAD_FAIL",
    "ND_FAIL",

    "ARGS_FAIL",
    "DIRECT_FAIL",
    "GRAD_NOT_COMPUTED",
    "RANDOM",
    "DIRECT_NAN"
    ]

        
def run_mutants(api_name, mutants, device, verbose=False, out_dir=None, result_log=None, result_second_log=None):
    statuses, statuses_second = [], []
    status_dict, status_second_dict = {}, {}
    
    for key, mutant in mutants.items():
        code = mutant["code"]
        input_names = mutant["input_names"]
        random_seed = mutant["random_seed"]
        api_record_dict = mutant["record"]
        if out_dir is not None:
            with open(os.path.join(out_dir, "temp.id"), "w") as f:
                f.write(f"{api_name}-{key}\n")
                
        status = run_and_check(code, random_seed, input_names, use_grad=False, device=device)
        status_second = None

        if status == ResultType.SUCCESS:
            status_second = run_and_check(code, random_seed, input_names, use_grad=True, device=device)
            mutant["secondrestyp"] = str(status_second)
            statuses_second.append(status_second)
            status_dict[key] = str(status_second)
            if result_second_log is not None:
                with open(result_second_log, "a") as f:
                    second_typ_str = str(status_second).replace('ResultType.', '') if status_second is not None else ""
                    f.write(f"{api_name}-{key} {second_typ_str}\n")

        mutant["restyp"] = str(status)
        if result_log is not None:
            with open(result_log, "a") as f:
                typ_str = str(status).replace('ResultType.', '')
                f.write(f"{api_name}-{key} {typ_str}\n")

        statuses.append(status)
        status_dict[key] = str(status)

    return statuses, status_dict, statuses_second, status_second_dict


def write_info(random_seed):
    set_seed_code = "import tensorflow as tf\n"
    set_seed_code += "import numpy as np\n"
    set_seed_code += "import random\n"
    set_seed_code += "tf.random.set_seed({})\n".format(random_seed)
    set_seed_code += "np.random.seed({})\n".format(random_seed)
    set_seed_code += "random.seed({})\n".format(random_seed)
    return set_seed_code

def gradtest_API_with_tests(api_name, mutants, result_fp, log_dir, verbose, device):
    
    for order in [1, 2]:
        order_log_dir = join(log_dir, str(order))
        os.makedirs(order_log_dir, exist_ok=True)

    result_log = join(log_dir, "1", f"mutant_result.log")
    result_second_log = join(log_dir, "2", f"mutant_result.log")
    
    statuses, status_dict, statuses_second, status_second_dict = run_mutants(api_name, mutants, device, 
        out_dir=log_dir, result_log=result_log, result_second_log=result_second_log, verbose=verbose)

    with open(result_fp, "w") as f:
        f.write(json.dumps({"1": status_dict, "2": status_second_dict}))
    for order in [1, 2]:
        order_log_dir = join(log_dir, str(order))

        log_file = join(order_log_dir, f"test-log.txt")
        neq_status_file = join(order_log_dir, f"neq_status.txt")
        neq_value_file = join(order_log_dir, f"neq_value.txt")
        neq_grad_file = join(order_log_dir, f"neq_grad.txt")
        fail_grad_file = join(order_log_dir, f"fail_grad.txt")
        result_log = join(order_log_dir, f"mutant_result.log")
        
        results = statuses if order == 1 else statuses_second
        results_dict = status_dict if order == 1 else status_second_dict
        with open(log_file, "a") as f:
            f.write(f"{api_name} ")
            if results == None:
                f.write("REJECT\n")
                return
            else:
                count = count_results(results, need_count)
                f.write(str(count) + "\n")

        for key, restyp in results_dict.items():
            if "VALUE" in str(restyp):
                with open(neq_value_file, "a") as f:
                    f.write(f"{api_name}-{key}\n")
            if "STATUS" in str(restyp):
                with open(neq_status_file, "a") as f:
                    f.write(f"{api_name}-{key}\n")
            if "GRAD_VALUE_MISMATCH" in str(restyp):
                with open(neq_grad_file, "a") as f:
                    f.write(f"{api_name}-{key}\n")
            if "GRAD_FAIL" in str(restyp) or "ND_FAIL" in str(restyp):
                with open(fail_grad_file, "a") as f:
                    f.write(f"{api_name}-{key}\n")
    return status_dict, status_second_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--api_name', 
        type=str,
        default=None
    )

    parser.add_argument(
        '--oracle_name',
        type=str,
        default="direct_bwd"
    )
    parser.add_argument(
        '--num',
        type=int,
        default=1000,
        help="The number of mutants for each API")
    parser.add_argument(
        '--expr_dir',
        type=str,
        default="./expr_outputs"
    )
    parser.add_argument(
        '--run_dirname',
        type=str,
        default="test"
    )
    parser.add_argument(
        '--device',
        type=str,
        default="gpu"
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False
    )
    args = parser.parse_args()

    api_name = args.api_name
    oracle_name = args.oracle_name
    num = args.num
    device = args.device
    verbose = args.verbose

    mode = "mutant"    
    expr_dir = args.expr_dir
    run_dirname = args.run_dirname

    expr_dir = Path(expr_dir)
    run_dir = expr_dir / run_dirname
    test_dir = expr_dir / mode
    log_dir = run_dir / "logs"
    result_dir = run_dir / "result"
    result_fp = result_dir / f"{api_name}.json"
    tests_fp = test_dir / f"{api_name}.json"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    with open(tests_fp, "r") as f:
        tests = json.load(f)

    results_dict, results_second_dict = gradtest_API_with_tests(api_name, tests, result_fp, log_dir, verbose, device)