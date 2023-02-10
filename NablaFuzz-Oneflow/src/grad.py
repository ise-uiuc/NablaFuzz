from utils.load_data import get_info
from preprocess.process_data import handle_params
from utils.fuzz_data import fuzz
from oracles.grad_oracle import grad_check
from skip.null_check import check_null_tensor
from skip.skip_error import check_acceptable
from to_code.to_code_init import initialize_code
from to_code.to_code_grad_check import to_code_grad_check
import numpy as np
import argparse
import oneflow
import os

parser = argparse.ArgumentParser()

parser.add_argument("--fuzzing_times", default=10, type=int)
parser.add_argument("--use_cuda", default=False, type=bool)
parser.add_argument("--random_seed", default=10, type=int)
parser.add_argument("--current_api", default=0, type=int)
fuzzing_params = vars(parser.parse_args())

covered = {}
filtered = []
g_errors = []
d_errors = []

#check for ND. vs. RD.
for i in range(fuzzing_params['fuzzing_times']):
    seed = i + fuzzing_params['random_seed']
    np.random.seed(seed)
    current_api = fuzzing_params['current_api']
    code = initialize_code(seed, current_api)

    api_name, params = get_info(current_api)
    arg_list, kwarg_dict, input_arg_list, input_kwarg_dict = handle_params(params)
    if api_name in covered.keys():
        covered[api_name] += 1
    else:
        covered[api_name] = 1
    if check_null_tensor(arg_list) or check_null_tensor(kwarg_dict) or\
         check_null_tensor(input_arg_list) or check_null_tensor(input_kwarg_dict):
        print("test skipped due to null tensor")
        continue
    use_cuda = fuzzing_params['use_cuda']
    arg_fuzzed = fuzz(arg_list, use_cuda)
    kwarg_fuzzed = fuzz(kwarg_dict,use_cuda)
    input_arg_fuzzed = fuzz(input_arg_list,use_cuda)
    input_kwarg_fuzzed = fuzz(input_kwarg_dict,use_cuda)
    api = eval(api_name)

    code = to_code_grad_check(api_name,code,use_cuda=False)
    f_crash_code = open("errors/direct/"+api_name+str(i)+".py",'w')
    f_crash_code.write(code)
    f_crash_code.close()

    try:
        error_message = grad_check(
            api, 
            arg_fuzzed, 
            kwarg_fuzzed, 
            input_arg_fuzzed, 
            input_kwarg_fuzzed,
            eps=1e-4, 
            atol=1e-2, 
            rtol=1e-2
        )
        if error_message == 'filtered':
            os.remove("errors/direct/"+api_name+str(i)+".py")
            if api_name not in filtered:
                filtered.append(api_name)
        elif error_message in ["abs_error rel_error", "nan error"]:
            print(error_message)
            os.remove("errors/direct/"+api_name+str(i)+".py")
            f_grad_code = open("errors/grad/"+api_name+str(i)+".py", 'w')
            f_grad_code.write(code)
            f_grad_code.close()
            if api_name not in g_errors:
                g_errors.append(api_name)
        else:
            os.remove("errors/direct/"+api_name+str(i)+".py")
    except Exception as e:
        print(type(e).__name__)
        if check_acceptable(str(e), type(e).__name__):
            os.remove("errors/direct/"+api_name+str(i)+".py")
            print("Accepatable error!")
        else:
            if api_name not in d_errors:
                d_errors.append(api_name)


print("covered apis:", covered)

print("filtered:", filtered)
print("gradient errors:", g_errors)
print("direct invocation errors:", d_errors)