import numpy as np
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from utils.handle_output import output_handler

def status_and_value_check(api, arg_fuzzed_reverse, kwarg_fuzzed_reverse, input_arg_fuzzed_reverse, input_kwarg_fuzzed_reverse,\
                arg_fuzzed_direct, kwarg_fuzzed_direct, input_arg_fuzzed_direct, input_kwarg_fuzzed_direct,atol,rtol):
    hint = None  
    #direct-invocation
    try:
        if type(api).__name__ == 'type':
            api_obj = api(*input_arg_fuzzed_direct, **input_kwarg_fuzzed_direct)
            output_direct = output_handler(api_obj(*arg_fuzzed_direct, **kwarg_fuzzed_direct))
        else:
            output_direct = output_handler(api(*arg_fuzzed_direct, **kwarg_fuzzed_direct))
    except Exception as e:
        hint = "direct invocation failed"
        return hint, [str(e), type(e).__name__]
    #reverse-mode invocation
    try:
        if type(api).__name__ == 'type':
            api_obj = api(*input_arg_fuzzed_reverse,**input_kwarg_fuzzed_reverse)
            output_reverse = output_handler(api_obj(*arg_fuzzed_reverse, **kwarg_fuzzed_reverse))
        else:
            output_reverse = output_handler(api(*arg_fuzzed_reverse, **kwarg_fuzzed_reverse))
    except Exception as e:
            hint = "status error"
            return hint, [str(e), type(e).__name__]

    if type(output_reverse).__name__ not in ['Tensor', 'Parameter']:
        if output_reverse != output_direct:
            hint = "value error"
            return hint, None
        else:
            return "OK", None
    else:
        if not np.allclose(output_direct.numpy(),
            output_reverse.numpy(),
            atol=atol, 
            rtol=rtol,
            equal_nan=True
        ):
            print(output_direct-output_reverse)
            print(output_direct, output_reverse)
            hint = "value error"
            return hint, None
        else:
            return "OK", None

