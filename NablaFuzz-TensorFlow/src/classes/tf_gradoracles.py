import os
import tensorflow as tf
import numpy as np
import json
from termcolor import colored

from classes.oracles import *
from constant.keys import *
from constant.parameters import ATOL, RTOL
from tf_utils import write_to_dir

ND_DELTA = 1e-6


def run_code_proc(code, results):
    import tensorflow as tf
    import numpy as np
    err = None
    try:
        exec(code, locals(), locals())
    except Exception as e:
        err = type(e).__name__ + "," + str(e).replace("\n", "")
    if err is not None:
        results[RUN_ERROR_KEY] = type(err).__name__ + "," + str(err)

def run_code(code, mode="exec", verbose=True, **kwargs):
    if mode == "exec":
        import tensorflow as tf
        import numpy as np
        try:
            results = dict()
            exec(code, locals(), locals())
        except Exception as e:
            results[RUN_ERROR_KEY] = type(err).__name__ + "," + str(err)
        if verbose: print(results)
        return results

    elif mode == "multiproc":
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        timeout = 10
        proc = multiprocessing.Process(target=run_code_proc, args=(code, return_dict))
        proc.start()
        proc.join(timeout)
        
        exitcode = proc.exitcode
        if exitcode != 0:
            return None, exitcode
        results = return_dict
        return results, exitcode

class IntegratedGradientOracleTF(OracleTF):
    mode = "multiproc"
    
    #region General arg to code functions. Refactored from oracles.py
    
    def _arg_to_code(self, api:TFAPI, skip_arg_to_code=False, split_assemble_code=False, verbose=False, enforce_double=False):
        arg_code = ""
        arg_str = ""
        arg_assemble_code = ""
        tensor_args = []

        for idx, arg_def in enumerate(api.arg_defs):
            arg = api.get_arg_by_names(arg_def.name, "parameter:"+str(idx))
            if arg_def.name == "**kwargs" and \
                isinstance(arg, dict) and len(arg) > 0:
                for key, argitem in arg.items():
                    if not skip_arg_to_code:
                        arg_code += argitem.to_code(key)
                    arg_str += f"{key}={key}, "
            elif arg!= None and arg_def.name != API_INVOCATION_INPUT_KEY:
                if arg.name == "**kwargs": continue
                arg_case = arg

                if not skip_arg_to_code:
                    if split_assemble_code:
                        arg_code_arg, arg_assemble_code_arg, tensor_args_arg = arg.to_code(
                            arg_case.var_name, split_assemble_code=split_assemble_code)
                        arg_code += arg_code_arg
                        arg_assemble_code += arg_assemble_code_arg
                        tensor_args.extend(tensor_args_arg)

                    else:
                        arg_code += arg_case.to_code(arg_case.var_name, split_assemble_code=split_assemble_code)

                if arg_def.name in [API_INVOCATION_INPUT_KEY, "name"]:
                    pass
                elif arg_def.is_optional or 'raw_ops' in api.api:
                    arg_str += f"{arg_def.name}={arg.var_name}, "
                else:
                    arg_str += f"{arg.var_name}, "
        if api.is_class:
            if API_INVOCATION_INPUT_KEY in api.args.keys():
                arg = api.args[ API_INVOCATION_INPUT_KEY ]
                arg_name = API_INVOCATION_INPUT_KEY
                arg_code += arg_assemble_code
                arg_assemble_code = ""
                tensor_args = []
                if not skip_arg_to_code:
                    if split_assemble_code:
                        arg_code_cls_input, arg_assemble_code_cls_input, tensor_args_cls_input = arg.to_code(arg_name, split_assemble_code=split_assemble_code)
                        arg_code += arg_code_cls_input
                        arg_assemble_code += arg_assemble_code_cls_input
                        tensor_args.extend(tensor_args_cls_input)
                    else:
                        arg_code += arg.to_code(arg_name, split_assemble_code=split_assemble_code)
                    api_name = api.api.split(".")[-1]
                    if enforce_double:
                        if not "dtype=" in arg_str:
                            arg_code += f"{api_name}_class = {api.api}({arg_str} dtype=tf.float64)\n"
                        else:
                            arg_code += f"{api_name}_class = {api.api}({arg_str})\n"
                    else:
                        arg_code += f"{api_name}_class = {api.api}({arg_str})\n"
                    
            else:
                raise ValueError(f"Input signature is not in the record of class {api.name}")
        if split_assemble_code:
            return arg_code, arg_assemble_code, arg_str, tensor_args
        return arg_code, arg_str, tensor_args

    def _call_to_code(self, api:TFAPI, arg_str:str, result_var:str) -> str:
        res_code = "tf.random.set_seed(42)\n"
        call_code = ""
        if api.is_class:
            api_name = api.api.split(".")[-1]

            if API_INVOCATION_INPUT_KEY in api.args.keys():
                arg_name = API_INVOCATION_INPUT_KEY
                res_code += f"{result_var} = {api_name}_class(*{arg_name})\n"
                call_code = f"{api_name}_class(*{arg_name})\n"
        else:
            res_code += f"{result_var} = {api.api}({arg_str})\n"
            call_code = f"{api.api}({arg_str})\n"
        return call_code, res_code
            

    #endregion

    @staticmethod
    def wrap_nd_grad(tensor_args:List[str], func_name=TEST_FUNC_EXEC_NAME): #TEST_FUNC_GRAD_NAME) -> str:
        """ Input is a list of argument names whose gradients will be computed. """ 
        args = ','.join(tensor_args)
        c = f"{GRAD_THEO_KEY}, {GRAD_NUME_KEY} = tf.test.compute_gradient({func_name}, [{args}], delta={ND_DELTA})\n"
        c += f"{RES_KEY}[\"{GRAD_THEO_KEY}\"] = {GRAD_THEO_KEY}\n"
        c += f"{RES_KEY}[\"{GRAD_NUME_KEY}\"] = {GRAD_NUME_KEY}\n"
        return c

    @staticmethod
    def wrap_function(call_code:str, args_assemble_code:str, args) -> str:

        c = ""
        # c += "@tf.function\n"
        c += f"def {TEST_FUNC_EXEC_NAME}({','.join(args)}):\n"
        c += API.indent_code(args_assemble_code)
        c += API.indent_code(f"return {call_code}")
        return c

    @staticmethod
    def wrap_function_with_args(args, arg_code, func_code, fn_name=TEST_FUNC_EXEC_NAME, wrapped_fn_name=WRAP_FUNC_EXEC_NAME) -> str:
        c = ""
        c += f"def {wrapped_fn_name}():\n"
        c += API.indent_code(arg_code)
        c += API.indent_code(func_code)
        c += API.indent_code(f"return {fn_name}")
        return c

    @staticmethod
    def wrap_function_for_gradient_computation(func_exec_def_code, args:List[str], func_exec_name=TEST_FUNC_EXEC_NAME, func_grad_name=TEST_FUNC_GRAD_NAME):
        arg_list = ','.join(args)
        tmp_var = f"result_{func_grad_name}"
    
        c = f"def {TEST_FUNC_GRAD_NAME}({','.join(args)}):\n" + \
            API.indent_code(func_exec_def_code) + \
            API.indent_code(f"{tmp_var} = {func_exec_name}({arg_list})") + \
            API.indent_code(f"return {tmp_var}")
        return c

    #region Gradient to code functions.
    @staticmethod
    def wrap_gradient_bwd(code:str, tensor_args, res, persistent=True, watch_accessed_variables=None, watch_tensor=True, result_ad_back_var="__res_ad_var") -> str:

        # Trace gradient.
        persistent_code = f"persistent={persistent}" if isinstance(persistent, bool) else ""
        watch_accessed_variables_code = f"watch_accessed_variables={watch_accessed_variables}" if isinstance(watch_accessed_variables, bool) else ""
        gcode = f"with tf.GradientTape({','.join([persistent_code, watch_accessed_variables_code])}) as g:\n"
        
        if watch_tensor:
            for argname in tensor_args:
                gcode += API.indent_code(f"g.watch({argname})\n")
            gcode += API.indent_code(code)
        
        gcode += f"{result_ad_back_var} = {res}\n"
        for argname in tensor_args:
            gcode += f"{RES_KEY}[\"gd_{argname}\"] = g.gradient({res}, {argname})\n"
        return gcode

    @staticmethod
    def wrap_gradient_fwd(code:str, tensor_args, res,result_ad_fwd_var)->str:
        gcode = ""
        for idx in range(len(tensor_args)):
            gcode += f"with tf.autodiff.ForwardAccumulator({tensor_args[idx]},tf.ones({tensor_args[idx]}.shape,{tensor_args[idx]}.dtype)) as acc_{idx}:\n"
            gcode += API.indent_code(code)
            gcode +=  API.indent_code(f"{RES_KEY}[\"gd_fwd_{tensor_args[idx]}\"] = acc_{idx}.jvp({res})\n")
        gcode += f"{result_ad_fwd_var} = {res}\n"
        return gcode
    #endregion


    def api_to_function_code(self, api:TFAPI, skip_arg_to_code=False, verbose=False):
        arg_code, args_assemble_code, arg_str, tensor_args = self._arg_to_code(
            api, skip_arg_to_code=skip_arg_to_code, split_assemble_code=True, verbose=verbose, enforce_double=True)
        
        result_exec_var = f"{RES_KEY}[\"{RES_EXEC_KEY}\"]"
        res_call_code, _ = self._call_to_code(api, arg_str, result_exec_var)
        res_func_code = self.wrap_function(res_call_code, args_assemble_code, tensor_args)
        res_func_code = self.wrap_function_with_args(tensor_args, arg_code, res_func_code)
        res_func_code += f"{TEST_FUNC_EXEC_NAME} = {WRAP_FUNC_EXEC_NAME}()\n"
        return arg_code, res_func_code, tensor_args

    def api_to_code(self, api:TFAPI, skip_arg_to_code=False, verbose=False):
        arg_code, func_exec_def_code, tensor_args = self.api_to_function_code(
            api, skip_arg_to_code, verbose)
        
        return arg_code, func_exec_def_code, tensor_args
        
        
    def check_cannot_back_ad(self, res_exec, err_ad:str): 

        # Check if result type cannot get gradient
        res_type = TFLibrary.get_type(res_exec)
        if res_type not in TFArgument._tensor_arg_dtypes:
            return True

        # Check if result dtype is differentiable.
        if res_exec.dtype not in [tf.float32, tf.float64, tf.complex64, tf.complex128]:
            return True
        
        # Check known error:
        if "LookupError,gradient registry has no entry for" in err_ad:
            return True

        return False

    def check_cannot_fwd_ad(self,res_exec):
        if res_exec.dtype == tf.float32:
            False
        return True
