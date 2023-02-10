from utils.load_data import get_info
from preprocess.process_data import handle_params
from utils.fuzz_data import fuzz_for_value_check
from oracles.status_value_oracle import status_and_value_check
from skip.null_check import check_null_tensor
from skip.skip_error import check_acceptable
from to_code.to_code_init import initialize_code
from to_code.to_code_consistent_check import to_code_consistent_check
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
v_errors = []
s_errors = []
for i in range(fuzzing_params['fuzzing_times']):
    seed = i+fuzzing_params['random_seed']
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
    
    use_cuda = fuzzing_params["use_cuda"]
    arg_fuzzed_reverse, arg_fuzzed_direct = fuzz_for_value_check(arg_list, use_cuda)
    kwarg_fuzzed_reverse, kwarg_fuzzed_direct = fuzz_for_value_check(kwarg_dict, use_cuda)
    input_arg_fuzzed_reverse, input_arg_fuzzed_direct = fuzz_for_value_check(input_arg_list, use_cuda)
    input_kwarg_fuzzed_reverse, input_kwarg_fuzzed_direct = fuzz_for_value_check(input_kwarg_dict, use_cuda)
    api = eval(api_name)
    
    code = to_code_consistent_check(api,api_name,code,use_cuda=False)
    f_crash_code = open("errors/direct/"+api_name+str(i)+".py",'w')
    f_crash_code.write(code)
    f_crash_code.close()

    hint, message = status_and_value_check(
        api, 
        arg_fuzzed_reverse, 
        kwarg_fuzzed_reverse, 
        input_arg_fuzzed_reverse, 
        input_kwarg_fuzzed_reverse,
        arg_fuzzed_direct, 
        kwarg_fuzzed_direct, 
        input_arg_fuzzed_direct, 
        input_kwarg_fuzzed_direct,
        atol=1e-4,
        rtol=1e-4
    )
        
    if hint == "direct invocation failed":
        print("Skip the test due to direct invocation failure")
        if check_acceptable(message[0], message[1]):
            os.remove("errors/direct/"+api_name+str(i)+".py")
            print("Accepatable error!")
    elif hint == "status error":
        os.remove("errors/direct/"+api_name+str(i)+".py")
        f_status_code = open("errors/status/"+api_name+str(i)+".py", 'w')
        f_status_code.write(code)
        f_status_code.close()
        if api_name not in s_errors:
            s_errors.append(api_name)
    elif hint == "value error":
        os.remove("errors/direct/"+api_name+str(i)+".py")
        f_value_code = open("errors/value/"+api_name+str(i)+".py", 'w')
        f_value_code.write(code)
        f_value_code.close()
        if api_name not in v_errors:
            v_errors.append(api_name)
    elif hint == "OK":
        print("OK.")
        os.remove("errors/direct/"+api_name+str(i)+".py")
        
print("covered apis:",covered)
print("potential status error:", s_errors)
print("potential value error:", v_errors)