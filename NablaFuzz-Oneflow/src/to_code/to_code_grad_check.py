def to_code_grad_check(api_name, code, use_cuda=False):
    code += f"arg_fuzzed = fuzz(arg_list,use_cuda={use_cuda})\n"  
    code += f"kwarg_fuzzed = fuzz(kwarg_dict,use_cuda={use_cuda})\n"
    code += f"input_arg_fuzzed = fuzz(input_arg_list,use_cuda={use_cuda})\n"
    code += f"input_kwarg_fuzzed = fuzz(input_kwarg_dict,use_cuda={use_cuda})\n"
    code += f"grad_check({api_name}, arg_fuzzed, kwarg_fuzzed, input_arg_fuzzed, input_kwarg_fuzzed,eps=1e-4, atol=1e-2, rtol=1e-2)\n"
    return code