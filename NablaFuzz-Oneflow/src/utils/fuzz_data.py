import oneflow
import random
import numpy as np
import sys

def restore_or_generate(arg, mode="reverse", dtype_for_value_check=oneflow.float64):
    arg_type = type(arg).__name__
    if arg_type == 'dict':
        if 'type' in arg.keys():
            if arg['type'] == 'Tensor':
                if arg['dtype'] =='oneflow.bool':
                    temp_t = np.random.randint(0, 2, tuple(arg['shape']))
                    t = oneflow.tensor(temp_t)
                    t = t.to(oneflow.bool)
                elif arg['dtype'].startswith('oneflow.int'):
                    random_dtype = np.random.choice([oneflow.int64, oneflow.int32])
                    temp_t = np.random.randint(0, 5, tuple(arg['shape']))
                    t = oneflow.tensor(temp_t,dtype=random_dtype)
                else:
                    if mode == 'reverse':
                        temp_t = np.random.uniform(tuple(arg['shape']))
                        t = oneflow.tensor(temp_t,
                            dtype=oneflow.float64,
                            requires_grad=True
                        )
                    elif mode == 'reverse_for_value_check':
                        temp_t = np.random.uniform(tuple(arg['shape']))
                        t = oneflow.tensor(temp_t,
                            dtype=dtype_for_value_check,
                            requires_grad=True
                        )
                    t.uniform_(-5,5)
                return t
            elif arg['type'] == 'Parameter':
                if mode == 'reverse':
                    temp_t = np.random.uniform(tuple(arg['shape']))
                    t = oneflow.tensor(temp_t,
                        dtype=oneflow.float64,
                        requires_grad=True
                    )
                elif mode == 'reverse_for_value_check':
                    temp_t = np.random.uniform(tuple(arg['shape']))
                    t = oneflow.tensor(temp_t, 
                        dtype= dtype_for_value_check, 
                        requires_grad=True
                    )
                t.uniform_(-5,5)
                p = oneflow.nn.Parameter(t)
                return p
            elif arg['type'] == 'ndarray':
                arr = np.random.rand(*tuple(arg['shape']))
                return arr
            elif arg['type'] == "NoneType":
                return None
            else:
                print('Tricky parameters!!!!')
        else:
            return arg
    
    else: 
        return arg


def fuzz(args,use_cuda=False):
    if type(args).__name__ == 'list':
        temp_list = []
        for arg in args:
            if type(arg).__name__ == 'list':
                temp_arg = list(fuzz(arg, use_cuda=use_cuda))
                temp_list.append(temp_arg)
                continue
            temp_arg = restore_or_generate(arg)
            if type(temp_arg).__name__ in ['Tensor', 'Parameter']:
                if use_cuda:
                    if temp_arg.requires_grad:
                        temp_list.append(temp_arg.detach().cuda().requires_grad_())
                    else:
                        temp_list.append(temp_arg.detach().cuda())
                else:
                    temp_list.append(temp_arg)
            else:
                temp_list.append(temp_arg)
        return tuple(temp_list)
    else:
        temp_dict = {}
        for item in args.items():
            temp_arg = restore_or_generate(item[1])
            if type(temp_arg).__name__ in ['Tensor', 'Parameter']:
                if use_cuda:
                    if temp_arg.requires_grad:
                        temp_dict[item[0]] = temp_arg.detach().cuda().requires_grad_()
                    else:
                        temp_dict[item[0]] = temp_arg.detach().cuda()
                else:
                    temp_dict[item[0]] = temp_arg
            else:
                temp_dict[item[0]] = temp_arg
        return temp_dict



def fuzz_for_value_check(args, use_cuda=False):
    dtype_list = [oneflow.float64, oneflow.float32]
    dtype = random.choice(dtype_list)
    if type(args).__name__ == 'list':
        temp_list_reverse = []
        temp_list_direct = []
        for arg in args:
            temp_arg = restore_or_generate(
                arg,mode='reverse_for_value_check',
                dtype_for_value_check=dtype
            )
            if not use_cuda:
                temp_list_reverse.append(temp_arg)
                if type(arg).__name__ == 'dict':
                    if arg['type'] in ['Parameter', 'Tensor']:
                        temp_arg = temp_arg.clone().detach()
                temp_list_direct.append(temp_arg)
            elif use_cuda:
                detach = False
                if type(arg).__name__ == 'dict':
                    if arg['type'] in ['Parameter', 'Tensor']:
                        temp_arg = temp_arg.clone().cuda()
                        temp_arg_detach = temp_arg.clone().detach()
                        detach = True
                temp_list_reverse.append(temp_arg)
                if detach:
                    temp_list_direct.append(temp_arg_detach)
                else:
                    temp_list_direct.append(temp_arg)
                
        return tuple(temp_list_reverse), tuple(temp_list_direct)
    else:
        temp_dict_reverse = {}
        temp_dict_direct = {}
        for item in args.items():
            temp_arg = restore_or_generate(
                item[1],
                mode='reverse_for_value_check',
                dtype_for_value_check=dtype
            )
            if not use_cuda:
                temp_dict_reverse[item[0]] = temp_arg
                if type(item[1]).__name__ == 'dict':
                    if item[1]['type'] in ['Parameter', 'Tensor']:
                        temp_arg = temp_arg.clone().detach()
                temp_dict_direct[item[0]] = temp_arg
            elif use_cuda:
                detach = False
                if type(item[1]).__name__ == 'dict':
                    if item[1]['type'] in ['Parameter', 'Tensor']:
                        temp_arg = temp_arg.clone().cuda()
                        temp_arg_detach = temp_arg.clone().detach()
                        detach = True
                temp_dict_reverse[item[0]] = temp_arg
                if detach:
                    temp_dict_direct[item[0]] = temp_arg_detach
                else:
                    temp_dict_direct[item[0]] = temp_arg
        return temp_dict_reverse, temp_dict_direct

