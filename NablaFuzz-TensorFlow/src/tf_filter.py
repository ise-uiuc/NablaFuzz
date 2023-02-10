import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR') # Suppress TF warnings
import argparse
from pathlib import Path
from termcolor import colored
from tf_utils import *
from tf_grad_utils import *

TEST_FUNC_EXEC_NAME = "test_func_exec"
NEIGHBOR_NUM = 5
NEIGHBOR_STEP = 1e-4
EPS = 1e-6

def get_numerical_jacobian(fn, inputs, eps=EPS):
    nd_status, nd_theoretical, nd_numerical, nd_err = NDINV(fn, inputs, delta=eps)
    return nd_numerical

def is_differentiable(fn, inputs, 
    delta=NEIGHBOR_STEP, n_sample=NEIGHBOR_NUM, eps=EPS, atol=ATOL, rtol=RTOL):
    def get_neighbor(inputs, delta):
        res = []
        for t in inputs:
            if t.dtype.is_complex:
                ftype = "float64" if t.dtype == tf.complex128 else "float32"
                
                delta_t = tf.complex(tf.random.uniform(
                    shape=t.shape, 
                    minval=-delta,
                    maxval=delta,
                    dtype=ftype), tf.random.uniform(
                    shape=t.shape, 
                    minval=-delta,
                    maxval=delta,
                    dtype=ftype))
            else:
                delta_t = tf.random.uniform(
                    shape=t.shape, 
                    minval=-delta,
                    maxval=delta,
                    dtype=t.dtype)
            t_ = tf.identity(t) + delta_t
            res.append(t_)
        return tuple(res)
        
    output = fn(*CopyInputs(inputs))
    jac = get_numerical_jacobian(fn, inputs, eps=eps)
    for _ in range(n_sample):
        inputs_ = get_neighbor(inputs, delta)
        output_ = fn(*CopyInputs(inputs_))
        if not is_equal(output, output_, atol=atol, rtol=rtol):
            return False
        jac_  = get_numerical_jacobian(fn, inputs_, eps=eps)
        if not is_gradient_equal(jac, jac_, atol=atol, rtol=rtol):
            return False
    return True


dtype_precision_dict = {
    "bfloat16": 0,
    "bfloat16": 1,
    "float16": 1,
    "float32": 2,
    "complex64": 2,
    "float64": 3,
    "complex128": 3,
}

def dtype_precision(dtype):
    dtype = dtype.name
    if "int" in dtype:
        return 0
    elif dtype in dtype_precision_dict.keys():
        return dtype_precision_dict[dtype]
    else:
        assert 0, f"No such dtype: {dtype}"


def is_high_to_low_precision(inputs, output):
    def _check(output):
        for input in inputs:
            if dtype_precision(input.dtype) > dtype_precision(output.dtype):
                return True
            if dtype_precision(input.dtype) < dtype_precision(output.dtype):
                return True
        return False

    if output is None:
        return False
    elif isinstance(output, list):
        for o in output:
            if _check(o):
                return True
        return False
    else:
        return _check(output)


def filter_api(fn, inputs, is_ND=False, use_jacobian=True, device='cpu', 
    use_diff=True, use_prec=True, delta=NEIGHBOR_STEP, n_sample=NEIGHBOR_NUM):
    def is_nan_grad(grad):
        ret = False
        for t in grad:
            if t is None: continue
            if is_complex_tensor_or_nparray(t):                
                ti = tf.math.imag(t)
                tr = tf.math.real(t)
                ret = ret or tf.reduce_any(tf.math.is_nan(ti)) \
                    or tf.reduce_any(tf.math.is_nan(tr))
            else:
                ret = ret or (t is not None and tf.reduce_any(tf.math.is_nan(t)))
        return ret

    # Bool
    for x in inputs:
        if x.dtype.is_bool: return "Bool"

    # NaN
    rev_status, rev_value, rev_grad, rev_err = RevInv(fn, inputs, device=device)
    fwd_status, fwd_value, fwd_grad, fwd_err = FwdInv(fn, inputs, use_jacobian=use_jacobian, device=device)
    
    if isinstance(rev_value, bool):
        return "Bool"
    if tf.is_tensor(rev_value):
        if tf.convert_to_tensor(rev_value).dtype.is_bool: return "Bool"

    has_rev_grad = rev_grad is not None
    has_fwd_grad = fwd_grad is not None
    if has_rev_grad:
        rev_nan = is_nan_grad(rev_grad)
    if has_fwd_grad:
        fwd_nan = is_nan_grad(fwd_grad)
    if (not has_rev_grad or rev_nan) and (not has_fwd_grad or fwd_nan):
        return "NaN"
    elif has_rev_grad and has_fwd_grad and rev_nan != fwd_nan:
        return "NaN-error"

    if rev_grad is None: return "Grad-None"
    for revg in rev_grad:
        if revg is None:
            return "Grad-None"
    if is_ND:
        if rev_grad is None: return "Grad-None"
        for revg in rev_grad:
            if revg is None:
                return "Grad-None"
    if has_rev_grad and has_fwd_grad:
        for i in range(len(rev_grad)):
            if is_complex_tensor_or_nparray(rev_grad[i]) and \
                not is_complex_tensor_or_nparray(fwd_grad[i]):
                return "ComplexRevFloatFwd"
    # Presicion
    if use_prec:
        if is_ND:
            inputs = CopyInputs(inputs, precise=True)
            _, output_value, _ = DirectInv(fn, inputs)
        else:
            output_value = rev_value if rev_value is not None else fwd_value
        if is_high_to_low_precision(inputs, output_value):
            return "Precision"

    # Differentiability
    if use_diff:
        inputs_high_prec = CopyInputs(inputs, precise=True)
        dir_status, output_value, dir_err = DirectInv(fn, inputs_high_prec)
        if dir_status == "success":
            if not is_differentiable(fn, inputs_high_prec, delta=delta, n_sample=n_sample):
                return "Non-Diff"
    return "Pass"

def filter_driver(api_name, key, t, test_result, device, verbose=True, 
    use_diff=True, use_prec=True, delta=1e-4, n_sample=5):
    code = t["code"]
    input_names = t["input_names"]
    random_seed = t["random_seed"]
    api_record_dict = t["record"]
    
    set_seed(random_seed)
    
    try:
        exec(code)
        exec("inputs = []")
        for tensor_name in input_names:
            eval(f"inputs.append({tensor_name})")
    except Exception as e:
        return ResultType.ARGS_FAIL, []
    else:
        is_ND = "ND_FAIL" in str(test_result)
        with tf.device(device):
            result = eval(f"filter_api({TEST_FUNC_EXEC_NAME}, inputs, " \
                f"is_ND={is_ND}, device=\"{device}\", " \
                f"use_diff={use_diff}, use_prec={use_prec}, " \
                f"delta={delta}, n_sample={n_sample})")
                
    return result


def inspect_API_inconsistency(api_name, key, t, test_result, device, verbose=True):
    code = t["code"]
    input_names = t["input_names"]
    random_seed = t["random_seed"]
    api_record_dict = t["record"]
    
    set_seed(random_seed)
    try:
        exec(code)
        exec("inputs = []")
        for tensor_name in input_names:
            eval(f"inputs.append({tensor_name})")
    except Exception as e:
        return ResultType.ARGS_FAIL
    else:
        with tf.device(device):
            result = eval(f"testAD({TEST_FUNC_EXEC_NAME}, inputs, device=\"{device}\")")
    return result


def if_skip_inconsistent_api(api_name):
    skip_keywords = [
        "Assert",
    ]
    for kw in skip_keywords:
        if kw in api_name:
            return True
    return False

def load_key(fn):
    if not os.path.exists(fn): return []
    with open(fn, "r") as f:
        data = f.readlines()
    keys = [x.strip().split(' ')[0] for x in data]
    return keys 

def inspect(api_name, key, t, test_result, 
    device='cpu', use_diff=True, use_prec=True, delta=NEIGHBOR_STEP, n_sample=NEIGHBOR_NUM,
    bugcandidate_fn=None, filtered_fn=None, notreproduced_fn=None
    ):
    """ Returns a tuple of (rerun_result, is_inconsistency, is_bug_candidate_after_filter)
    """
    
    rerun_res = inspect_API_inconsistency(api_name, key, t, test_result, device)

    is_inconsistency = False
    is_bug_candidate_after_filter = False

    if rerun_res not in BUGGY_RESULT_TYPES:
        dump_res(f"{api_name}-{key} {rerun_res}", notreproduced_fn)
    else:
        filter_res = filter_driver(api_name, key, t, rerun_res, 
            device=device, use_diff=use_diff, use_prec=use_prec, delta=delta, n_sample=n_sample)
            
        if filter_res == "Pass":
            dump_res(f"{api_name}-{key} {rerun_res}", bugcandidate_fn)
            is_inconsistency = True
            is_bug_candidate_after_filter = True
        else:
            if filter_res in ["Precision", "Non-Diff"]:
                is_inconsistency = True
            dump_res(f"{api_name}-{key} {filter_res}", filtered_fn)
    return rerun_res, is_inconsistency, is_bug_candidate_after_filter
            
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
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
        '--mode', 
        type=str,
        default="mutant"
    )
    parser.add_argument(
        '--device', 
        type=str,
        default="gpu"
    )
    
    args = parser.parse_args()
    print(args)
    
    device = args.device
    expr_dir = Path(args.expr_dir)
    run_dir = expr_dir / args.run_dirname
    tests_dir = expr_dir / args.mode

    with open(expr_dir / "apis.txt", "r") as f:
        apis = [x.strip() for x in f.readlines()]
    
    result_dir = os.path.join(run_dir, "result")
    rerun_dir = os.path.join(run_dir, "rerun")
    os.makedirs(rerun_dir, exist_ok=True)
    
    notreproducedfn = os.path.join(rerun_dir, "notreproducedfn.txt")
    filteredfn = os.path.join(rerun_dir, "filteredfn.txt")
    bugcandidatefn = os.path.join(rerun_dir, "bugcandidatefn.txt")
    

    skip_apis = [
        "tf.broadcast_to",
        "tf.compat.v1.broadcast_to"
    ]

    all_test_results_fp = run_dir / "logs" / "1" / "mutant_result.log"
    all_test_results = {}
    with open(all_test_results_fp, "r") as fr:
        data = fr.readlines()
        for line in data:
            api_key, res = line.strip().split(" ")
            api_name, key = api_key.split("-")
            if api_name not in all_test_results:
                all_test_results[api_name] = dict()
            all_test_results[api_name][key] = "ResultType." + res


    for api_name in apis:
        if if_skip_inconsistent_api(api_name): continue

        result_fp = os.path.join(result_dir, api_name + ".json")
        testfn = os.path.join(tests_dir, api_name + ".json")
        with open(testfn, "r") as f:
            tests = json.load(f)
            
        if api_name not in all_test_results: continue
        test_results = all_test_results[api_name]
        
        for key, test in tests.items():
            keyid = int(key)
            if key not in test_results: continue
            if api_name in skip_apis: continue
            
            test_result = test_results[key]
            if test_result != "unknown":
                test_result = eval(test_result)
                
            is_bug_candidate = inspect(api_name, key, test, test_result,
                bugcandidate_fn=bugcandidatefn, filtered_fn=filteredfn, notreproduced_fn=notreproducedfn)