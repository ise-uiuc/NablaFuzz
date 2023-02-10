import oneflow
import random
import numpy as np
import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
from utils.numerical_grad import calc_nd
from utils.handle_output import output_handler

def grad_filter(api, api_obj, arg_fuzzed, kwarg_fuzzed, t, random_idx, random_arg_idx, n=5, sample_range=1e-4,atol=0.1, rtol=1e-3):
    _, _, grad_center = calc_nd(
        api,
        api_obj, 
        arg_fuzzed, 
        kwarg_fuzzed, 
        t, 
        random_arg_idx, 
        random_idx
    )
    for i in range(n): 
        new_t = oneflow.zeros_like(t)
        new_t[random_idx] += random.uniform(-sample_range, sample_range)
        _, _, grad_neighbor = calc_nd(
            api, 
            api_obj, 
            arg_fuzzed, 
            kwarg_fuzzed, 
            new_t, random_arg_idx, 
            random_idx
        )
        if not np.allclose(grad_neighbor.numpy(),
            grad_center.numpy(),
            atol=atol, 
            rtol=rtol,
            equal_nan=True):
            print('Not differentiable')
            return False
    return True

def grad_check(api, arg_fuzzed, kwarg_fuzzed, input_arg_fuzzed, input_kwarg_fuzzed, eps=1e-4, atol=1e-2, rtol=1e-2):
    api_obj = None
    if input_arg_fuzzed != () or input_kwarg_fuzzed != {}:
        random_arg_idx = 0
        t =  arg_fuzzed[random_arg_idx]
    else:
        random_arg_idx = 0
        t = arg_fuzzed[random_arg_idx]    
        if type(t).__name__ == 'list':
            random_arg_idx = [0,0]
            t = t[random_arg_idx[1]]
        elif type(arg_fuzzed[random_arg_idx]).__name__ not in ['Tensor','Parameter'] and len(arg_fuzzed[random_arg_idx])>1:
            random_arg_idx = 1
            t = arg_fuzzed[random_arg_idx]
    if hasattr(t,'shape'):
        shape_list = t.shape
        random_idx = []
        for dim in shape_list:
            random_idx.append(random.randint(0,dim-1))

        random_idx = tuple(random_idx)
    else:
        random_idx=0
    
    if type(api).__name__ == 'type':
        api_obj = api(*input_arg_fuzzed,**input_kwarg_fuzzed)
    
    #reverse-mode gradient
    if type(api).__name__ == 'type':
        output = output_handler(api_obj(*arg_fuzzed,**kwarg_fuzzed))
        output.backward()
        if type(random_arg_idx).__name__ == 'list':
            r_gradient = arg_fuzzed[random_arg_idx[0]][random_arg_idx[1]].grad[random_idx]
        else:
            r_gradient = arg_fuzzed[random_arg_idx].grad[random_idx]
    else:
        output = output_handler(api(*arg_fuzzed,**kwarg_fuzzed))
        output.backward()
        if type(random_arg_idx).__name__ == 'list':
            r_gradient = arg_fuzzed[random_arg_idx[0]][random_arg_idx[1]].grad[random_idx]
        else:
            r_gradient = arg_fuzzed[random_arg_idx].grad[random_idx]
    
    #numerical gradient
    n_gradient_l, n_gradient_r, n_gradient_c = calc_nd(api, 
        api_obj, 
        arg_fuzzed, 
        kwarg_fuzzed, 
        t, 
        random_arg_idx, 
        random_idx
    )
    print("numerical_gradient:", float(n_gradient_l.numpy()), float(n_gradient_r.numpy()), float(n_gradient_c.numpy()))
    print("reverse_mode_gradient:",float(r_gradient.numpy()))

    n_nan_some, n_nan_all, r_nan = False, False, False
    if np.isnan(n_gradient_c.numpy()) and np.isnan(n_gradient_l.numpy()) and np.isnan(n_gradient_c.numpy()):
        n_nan_all = True
    if np.isnan(n_gradient_c.numpy()) or np.isnan(n_gradient_l.numpy()) or np.isnan(n_gradient_c.numpy()):
        n_nan_some = True
    if np.isnan(r_gradient.numpy()):
        r_nan = True
    error_messages = []    

    if r_nan and not n_nan_some:
        error_messages = ['nan error']
    if not r_nan and n_nan_all:
        error_messages = ['nan error']
    abs_error_arr = np.array([
        abs(float((n_gradient_l-r_gradient).numpy())), 
        abs(float((n_gradient_r-r_gradient).numpy())), 
        abs(float((n_gradient_c-r_gradient).numpy()))
    ])
    rel_error_arr = np.array([
        abs(float(((n_gradient_l-r_gradient)/r_gradient).numpy())),
        abs(float(((n_gradient_r-r_gradient)/r_gradient).numpy())), 
        abs(float(((n_gradient_c-r_gradient)/r_gradient).numpy()))
    ])

    if (abs_error_arr >= atol).all():
        print("Absolute error exceeds maximum tolerance!",
            "abs_error:", abs_error_arr,
            "numerical_gradient:",float(n_gradient_l.numpy()),float(n_gradient_r.numpy()),float(n_gradient_c.numpy()),
            "reverse_mode_gradient:",float(r_gradient.numpy())
        )
        error_messages.append("abs_error")
    if (rel_error_arr >= rtol).all():
        print("Relative error exceeds maximum tolerance!",
            "rel_error:", rel_error_arr,
            "numerical_gradient:",float(n_gradient_l.numpy()),float(n_gradient_r.numpy()),float(n_gradient_c.numpy()),
            "reverse_mode_gradient:",float(r_gradient.numpy())
        )
        error_messages.append("rel_error")
    if error_messages == ["abs_error", "rel_error"] or error_messages == ['nan error']:
        if grad_filter(api, 
            api_obj, 
            arg_fuzzed, 
            kwarg_fuzzed, 
            t, 
            random_idx, 
            random_arg_idx
        ):
            return " ".join(error_messages)
        print('Filtered')
        return 'filtered'
    print("OK.")
    return None