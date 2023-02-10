import subprocess
import argparse
import random
from pathlib import Path
import os

MAX_API_NUM = 299

output_dir = Path("errors")
os.makedirs(output_dir / "direct", exist_ok=True)
os.makedirs(output_dir / "status", exist_ok=True)
os.makedirs(output_dir / "value", exist_ok=True)
os.makedirs(output_dir / "grad", exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NablaFuzz for Oneflow"
    )

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
    args = parser.parse_args()
    print(args)

    api_num = args.max_api
    fuzzing_times = args.num
    use_cuda = False
    random_seed = 10

    if api_num == -1:
        api_num = MAX_API_NUM
    api_num = max(min(api_num, MAX_API_NUM), 1)
    api_idx = [i for i in range(api_num)]

    for api in api_idx:
        print(f"{api} / {api_num}")
        subprocess.run(
            f"python src/grad.py --current_api {api} --fuzzing_times {fuzzing_times} --use_cuda {use_cuda} --random_seed {random_seed}",
            shell=True,
            close_fds=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        subprocess.run(
            f"python src/status_value.py --current_api {api} --fuzzing_times {fuzzing_times} --use_cuda {use_cuda} --random_seed {random_seed}",
            shell=True,
            close_fds=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
