from pathlib import Path
from random import choice
import os
import argparse
import torch
from helper_torch import (
    DirectInv,
    RevInv,
    FwdInv,
    NDCheck,
    Grad,
    allow_error,
    is_crash,
)
from classes.torch_library import TorchLibrary
from classes.torch_api import TorchAPI, TorchArgument, Argument
from classes.database import TorchDatabase
from constant.returntypes import ResType
from utils.printer import dump_data

TorchDatabase.database_config("127.0.0.1", 27017, "torch")


def test(fn, inputs):
    inputs = tuple(inputs)

    direct_status, direct_value, direct_err = DirectInv(fn, inputs)
    if is_crash(direct_err):
        return ResType.DIRECT_CRASH

    for _ in range(9):
        direct_status_, direct_value_, direct_err_ = DirectInv(fn, inputs)
        if direct_status != direct_status_ or not TorchLibrary.is_equal(
            direct_value, direct_value_, equal_nan=True
        ):
            return ResType.RANDOM
        elif not TorchLibrary.is_equal(direct_value, direct_value_):
            return ResType.NAN

    rev_status, rev_value, rev_grad, rev_err = RevInv(fn, inputs)
    fwd_status, fwd_value, fwd_grad, fwd_err = FwdInv(fn, inputs)
    rev_restype = ResType.PASS
    fwd_restype = ResType.PASS

    if rev_status != direct_status and not allow_error(rev_err):
        rev_restype = ResType.REV_STATUS
    elif is_crash(rev_err):
        rev_restype = ResType.REV_CRASH
    elif rev_status == "fail":
        # print(rev_err)
        rev_restype = ResType.SKIP
    elif not TorchLibrary.is_equal(direct_value, rev_value):
        rev_restype = ResType.REV_VALUE

    if fwd_status != direct_status and not allow_error(fwd_err):
        fwd_restype = ResType.FWD_STATUS
    elif is_crash(fwd_err):
        fwd_restype = ResType.FWD_CRASH
    elif fwd_status == "fail":
        fwd_restype = ResType.SKIP
    elif not TorchLibrary.is_equal(direct_value, fwd_value):
        fwd_restype = ResType.FWD_VALUE

    if (
        rev_restype == ResType.PASS
        and fwd_restype == ResType.PASS
        and not TorchLibrary.is_equal(rev_grad, fwd_grad)
    ):
        # print(rev_grad)
        # print(fwd_grad)
        return ResType.REV_FWD_GRAD

    if rev_restype == ResType.PASS:
        nd_status, nd_err = NDCheck(fn, inputs, "rev")
        if is_crash(nd_err):
            rev_restype = ResType.ND_CRASH
        elif nd_status == "fail":
            if allow_error(nd_err):
                rev_restype = ResType.SKIP
            elif "Jacobian" in nd_err:
                rev_restype = ResType.ND_GRAD
            else:
                rev_restype = ResType.ND_FAIL
    if fwd_restype == ResType.PASS:
        nd_status, nd_err = NDCheck(fn, inputs, "fwd")
        if is_crash(nd_err):
            fwd_restype = ResType.ND_CRASH
        elif nd_status == "fail":
            if allow_error(nd_err):
                fwd_restype = ResType.SKIP
            elif "Jacobian" in nd_err:
                fwd_restype = ResType.ND_GRAD
            else:
                fwd_restype = ResType.ND_FAIL
    return (rev_restype, fwd_restype)


def testAPI(
    api_name,
    num=1000,
    output_dir: Path = Path("../output-ad/torch"),
    mutate=True,
):
    def get_clean_counts(counts):
        clean_counts = dict()
        for key, value in counts.items():
            if value > 0:
                clean_counts[str(key).replace("ResType.", "")] = value
        return clean_counts

    apiout_dir = output_dir / api_name
    all_dir = apiout_dir / "all"
    os.makedirs(all_dir, exist_ok=True)
    first_dirs = {
        ResType.RANDOM: apiout_dir / "random",
        ResType.STATUS: apiout_dir / "status",
        ResType.VALUE: apiout_dir / "value",
        ResType.REV_FWD_GRAD: apiout_dir / "grad-rev-fwd",
        ResType.ND_GRAD: apiout_dir / "grad-nd",
        ResType.REV_STATUS: apiout_dir / "status-rev",
        ResType.REV_VALUE: apiout_dir / "value-rev",
        ResType.FWD_STATUS: apiout_dir / "status-fwd",
        ResType.FWD_VALUE: apiout_dir / "value-fwd",
        ResType.PASS: apiout_dir / "pass",
        # ResType.SKIP: apiout_dir / "skip",
        ResType.CRASH: apiout_dir / "crash",
        ResType.DIRECT_CRASH: apiout_dir / "crash-direct",
        ResType.REV_CRASH: apiout_dir / "crash-rev",
        ResType.FWD_CRASH: apiout_dir / "crash-fwd",
        ResType.ND_CRASH: apiout_dir / "crash-nd",
        ResType.NAN: apiout_dir / "nan",
        ResType.ND_FAIL: apiout_dir / "nd-fail",
    }

    second_out_dir = apiout_dir / "grad"
    os.makedirs(second_out_dir, exist_ok=True)
    second_dirs = {
        ResType.RANDOM: second_out_dir / "random",
        ResType.STATUS: second_out_dir / "status",
        ResType.VALUE: second_out_dir / "value",
        ResType.REV_FWD_GRAD: second_out_dir / "grad-rev-fwd",
        ResType.ND_GRAD: second_out_dir / "grad-nd",
        ResType.REV_STATUS: second_out_dir / "status-rev",
        ResType.REV_VALUE: second_out_dir / "value-rev",
        ResType.FWD_STATUS: second_out_dir / "status-fwd",
        ResType.FWD_VALUE: second_out_dir / "value-fwd",
        ResType.PASS: second_out_dir / "pass",
        ResType.SKIP: second_out_dir / "skip",
        ResType.CRASH: second_out_dir / "crash",
        ResType.DIRECT_CRASH: second_out_dir / "crash-direct",
        ResType.REV_CRASH: second_out_dir / "crash-rev",
        ResType.FWD_CRASH: second_out_dir / "crash-fwd",
        ResType.ND_CRASH: second_out_dir / "crash-nd",
        ResType.NAN: second_out_dir / "nan",
        ResType.ND_FAIL: second_out_dir / "nd-fail",
    }

    # set the tensor_size_limit as 1024 to reduce memory and time cost
    TorchArgument._tensor_size_limit = 1024
    # gradcheck should avoid large number, which can cause false positive
    temp_values = []
    for v in Argument._float_values:
        if abs(v) < 1024:
            temp_values.append(v)
    Argument._float_values = temp_values

    api = TorchAPI(api_name)
    records = TorchDatabase.get_all_records(api_name)

    first_counts = {t: 0 for t in ResType}
    second_counts = {t: 0 for t in ResType}

    for k in range(num):
        if mutate:
            api.get_invocation(choice(records))
            api.mutate()
        else:
            if k < len(records):
                api.get_invocation(records[k])
            else:
                break

        first_ret = testrun(api, first_dirs, all_dir, output_dir, first_counts)
        if first_ret == (ResType.PASS, ResType.PASS):
            testrun(api, second_dirs, None, output_dir, second_counts, True)

    first_clean_counts = get_clean_counts(first_counts)
    second_clean_counts = get_clean_counts(second_counts)
    print(first_clean_counts)
    print(second_clean_counts)

    log_file = output_dir / "log.txt"
    dump_data(
        f"{api_name}\n{first_clean_counts}\n{second_clean_counts}\n",
        log_file,
        "a",
    )

    log_csv_file = output_dir / "log.csv"
    first_str_list = [str(i) for i in first_counts.values()]
    second_str_list = [str(i) for i in second_counts.values()]
    dump_data(
        f"{api_name}, {', '.join(first_str_list)}, {', '.join(second_str_list)}\n",
        log_csv_file,
        "a",
    )


def testrun(
    api: TorchAPI, dirs, all_dir, output_dir, counts: dict, use_grad=False
):
    def get_log_code():
        log_code = "import torch\n"
        log_code += fn_code
        log_code += inv_code
        log_code += str(input_list)
        return log_code

    def log(restype: ResType):
        log_code = get_log_code()
        if restype in dirs.keys():
            out_dir = dirs[restype]
            os.makedirs(out_dir, exist_ok=True)
            TorchLibrary.write_to_dir(out_dir, log_code)
        counts[restype] += 1
        if all_dir is not None:
            TorchLibrary.write_to_dir(all_dir, log_code)

    fn_code, inv_code, input_list = api.to_differential_fn_code()
    dump_data(get_log_code(), output_dir / "temp.py")
    if len(input_list):
        try:
            exec(fn_code)
            exec(inv_code)
        except Exception:
            ret = ResType.SKIP
        else:
            try:
                inputs_str = f"({', '.join(input_list)},)"
                if use_grad:
                    ret = eval(f"test(Grad(fn, {inputs_str}), {inputs_str})")
                else:
                    ret = eval(f"test(fn, {inputs_str})")
            except Exception:
                ret = ResType.CRASH
    else:
        ret = ResType.SKIP

    if isinstance(ret, tuple):
        # Merge the restype
        if ResType.REV_STATUS in ret and ResType.FWD_STATUS in ret:
            ret = ResType.STATUS
            log(ret)
        elif ResType.REV_STATUS in ret and ResType.FWD_STATUS in ret:
            ret = ResType.VALUE
            log(ret)
        elif ret[0] == ret[1]:
            log(ret[0])
        else:
            for t in ret:
                if t != ResType.SKIP:
                    log(t)
    else:
        log(ret)
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Autodiff Unit Test")
    parser.add_argument(
        "--api",
        type=str,
        help="The name of API to be test, e.g., torch.sum",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        help="The number of mutants for each API (default: 1000)",
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="../output-ad/torch",
        help="The output dir",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        default=False,
        help="Use gradgradcheck to test forward-over-rev mode (default: False)",
    )

    args = parser.parse_args()
    os.makedirs(Path(args.dir), exist_ok=True)

    mutate = not args.db
    testAPI(args.api, args.num, Path(args.dir), mutate)
