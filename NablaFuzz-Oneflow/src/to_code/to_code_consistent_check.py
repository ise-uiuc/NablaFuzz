def to_code_consistent_check(api, api_name,code,use_cuda=False):
    code += f"arg_fuzzed_reverse, arg_fuzzed_direct = fuzz_for_value_check(arg_list,use_cuda={use_cuda})\n"  
    code += f"kwarg_fuzzed_reverse, kwarg_fuzzed_direct = fuzz_for_value_check(kwarg_dict,use_cuda={use_cuda})\n"
    code += f"input_arg_fuzzed_reverse, input_arg_fuzzed_direct = fuzz_for_value_check(input_arg_list,use_cuda={use_cuda})\n"
    code += f"input_kwarg_fuzzed_reverse, input_kwarg_fuzzed_direct = fuzz_for_value_check(input_kwarg_dict,use_cuda={use_cuda})\n"
    if type(api).__name__ == 'type':
        code += f"api_direct_obj={api_name}(*input_arg_fuzzed_direct, **input_kwarg_fuzzed_direct)\n"
        code += f"output_direct = api_direct_obj(*arg_fuzzed_direct, **kwarg_fuzzed_direct)\n"
    else:
        code += f"output_direct = oneflow.{api_name}(*arg_fuzzed_direct, **kwarg_fuzzed_direct)\n"
    code += f"output_direct = oneflow.sum(output_direct)\n"

    if type(api).__name__ == 'type':
        code += f"api_reverse_obj=oneflow.{api_name}(*input_arg_fuzzed_direct, **input_kwarg_fuzzed_direct)\n"
        code += f"output_reverse = api_direct_obj(*arg_fuzzed_reverse, **kwarg_fuzzed_reverse)\n"
    else:
        code += f"output_reverse = oneflow.{api_name}(*arg_fuzzed_reverse, **kwarg_fuzzed_reverse)\n"
    code += f"output_reverse = oneflow.sum(output_reverse)\n\n"
    code += f"print(output_direct)\n"
    code += f"print(output_reverse)\n"
    code += f"print(output_direct==output_reverse)\n"
    return code