import oneflow
import copy
from utils.handle_output import output_handler
import sys

def calc_nd(api, api_obj, arg_fuzzed, kwarg_fuzzed, t, random_arg_idx, random_idx, eps=1e-4):
    eps_matrix = oneflow.zeros_like(t)
    eps_matrix[random_idx] = eps
    t_eps_r = t + eps_matrix
    t_eps_l = t - eps_matrix

    arg_fuzzed_eps_l = copy.deepcopy(list(arg_fuzzed))
    if type(random_arg_idx).__name__ == 'list':
        arg_fuzzed_eps_l[random_arg_idx[0]][random_arg_idx[1]] = t_eps_l
    else:
        arg_fuzzed_eps_l[random_arg_idx] = t_eps_l
    arg_fuzzed_eps_l = tuple(arg_fuzzed_eps_l)

    arg_fuzzed_eps_r = copy.deepcopy(list(arg_fuzzed))
    if type(random_arg_idx).__name__ == 'list':
        arg_fuzzed_eps_r[random_arg_idx[0]][random_arg_idx[1]] = t_eps_r
    else:
        arg_fuzzed_eps_r[random_arg_idx] = t_eps_r
    arg_fuzzed_eps_r = tuple(arg_fuzzed_eps_r)

    if type(api).__name__ == 'type':
        output = output_handler(api_obj(*arg_fuzzed,**kwarg_fuzzed))
        output_eps_l = output_handler(api_obj(*arg_fuzzed_eps_l,**kwarg_fuzzed))
        output_eps_r = output_handler(api_obj(*arg_fuzzed_eps_r,**kwarg_fuzzed))
    else:
        output = output_handler(api(*arg_fuzzed,**kwarg_fuzzed))
        output_eps_l = output_handler(api(*arg_fuzzed_eps_l,**kwarg_fuzzed))
        output_eps_r = output_handler(api(*arg_fuzzed_eps_r,**kwarg_fuzzed))
    n_gradient_l = (output-output_eps_l)/eps
    n_gradient_r = (output_eps_r-output)/eps
    n_gradient_c = (output_eps_r-output_eps_l)/(2*eps)

    return n_gradient_l, n_gradient_r, n_gradient_c