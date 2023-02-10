from abc import abstractmethod
import multiprocessing
from pathlib import Path
from random import randint
from typing import List
from unittest import result

from tf_utils import ResultType
from classes.api import API
from classes.argument import ArgType, Argument
from classes.tf_api import TFAPI, TFArgument
from classes.tf_library import TFLibrary
from constant.keys import *
from enum import IntEnum
from classes.library_def import tf_lib_def

class OracleType(IntEnum):
    CRASH = 1
    CUDA = 2
    PRECISION = 3
    GRADIENT = 4

class Oracle():

    @abstractmethod
    def api_to_code(self, api:API):
        pass

    @abstractmethod
    def arg_to_code(self, api:Argument):
        pass
    
    @staticmethod
    def wrap_try(code:str, error_var) -> str:
        wrapped_code = "try:\n"
        if code.strip() == "":
            code = "pass"
        wrapped_code += API.indent_code(code)
        wrapped_code += f"except Exception as e:\n  {RES_KEY}[\"{error_var}\"] = type(e).__name__ + \",\" + str(e)\n"
        return wrapped_code

    @staticmethod
    def wrap_device(code:str, device) -> str:
        device_code = f"with tf.device('/{device}'):\n" + API.indent_code(code)
        return device_code

    @staticmethod
    def wrap_time(code:str, time_var) -> str:
        wrapped_code = "t_start = time.time()\n"
        wrapped_code += code
        wrapped_code += "t_end = time.time()\n"
        wrapped_code += f"{RES_KEY}[\"{time_var}\"] = t_end - t_start\n"
        return wrapped_code

    @staticmethod
    def get_status_res(result_dict:dict, error_key, res_key):
        """ Returns (str, None) if has error, else return (None, res). """
        if error_key in result_dict:
            errmsg = result_dict[error_key]
            errtyp = errmsg.split(',', 1)[0]
            return errtyp, None
        else:
            if res_key not in result_dict:
                return None, None
            return None, result_dict[res_key]


class OracleTF(Oracle):
    
    pretty_print_str = """def pretty_print(res, msg=""):
    def _pretty_print(v):
        if tf.is_tensor(v):
            avg = ""
            if v.dtype.is_integer or v.dtype.is_floating or v.dtype.is_complex:
                avg = np.average(v.numpy())
            return f"{v.shape}, {v.dtype}, {avg}"
        elif isinstance(v, np.ndarray):
            avg = ""
            if np.issubdtype(v.dtype, np.integer) or \
                np.isrealobj(v) or np.iscomplexobj(v):
                avg = np.average(v)
            return f"{v.shape}, {v.dtype}, {avg}"
        elif isinstance(v, list) or isinstance(v, tuple):
            return [_pretty_print(x) for x in v]
        elif isinstance(v, dict):
            pres = dict()
            for k, v in res.items():
                pres[k] = _pretty_print(v)
            print(pres)
        else:
            return str(v)
    print(msg, _pretty_print(res))
"""

    @staticmethod
    def wrap_compgrad(tensor_args:List[str]) -> str:
        """ Input is a list of argument names whose gradients will be computed. """ 
        args = ','.join(tensor_args)
        c = f"{GRAD_THEO_KEY}, {GRAD_NUME_KEY} = tf.test.compute_gradient(test_func, [{args}])\n"
        c += f"{RES_KEY}[\"{GRAD_THEO_KEY}\"] = {GRAD_THEO_KEY}\n"
        c += f"{RES_KEY}[\"{GRAD_NUME_KEY}\"] = {GRAD_NUME_KEY}\n"
        return c

    @staticmethod
    def wrap_function(code:str, tensor_args) -> str:
        c = "@tf.function\n"
        c += f"def test_func({','.join(tensor_args)}):\n"
        c += API.indent_code(f"return {code}")
        return c

    @staticmethod
    def wrap_gradient(code:str, tensor_args, res, persistent=True, watch_accessed_variables=None, watch_tensor=True) -> str:

        # Trace gradient.
        # Default statement is "with tf.GradientTape(persistent=True) as g:\n"
        persistent_code = f"persistent={persistent}" if isinstance(persistent, bool) else ""
        watch_accessed_variables_code = f"watch_accessed_variables={watch_accessed_variables}" if isinstance(watch_accessed_variables, bool) else ""
        gcode = f"with tf.GradientTape({','.join([persistent_code, watch_accessed_variables_code])}) as g:\n"
        
        if watch_tensor:
            for argname in tensor_args:
                gcode += API.indent_code(f"g.watch({argname})\n")
            gcode += API.indent_code(code)
        
        for argname in tensor_args:
            gcode += f"{RES_KEY}[\"gd_{6}\"] = g.gradient({res}, {argname})\n"
        return gcode
        
    def _api_to_code(self, api, res="res",
            skip_arg_to_code=False, **kwargs) -> str:
        arg_code = ""
        arg_str = ""
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
                    arg_code += arg_case.to_code(arg_case.var_name)

                # for the *shape, arg_def.name == *shape, arg_case.name == shape
                if arg_def.name in [API_INVOCATION_INPUT_KEY, "name"]:
                    pass
                elif arg_def.is_optional or 'raw_ops' in api.api:
                    arg_str += f"{arg_def.name}={arg.var_name}, "
                else:
                    arg_str += f"{arg.var_name}, "
                    if arg.is_tensorlike() and arg.trainable:
                        tensor_args.append(arg.var_name)

        res_code = "tf.random.set_seed(42)\n"
        call_code = ""
        result_var = f"{RES_KEY}[\"{res}\"]"
        if api.is_class:
            api_name = api.api.split(".")[-1]
            res_code += f"{api_name}_class = {api.api}({arg_str})\n"

            if API_INVOCATION_INPUT_KEY in api.args.keys():
                arg = api.args[ API_INVOCATION_INPUT_KEY ]
                arg_name = "__input__"
                if not skip_arg_to_code:
                    arg_code += arg.to_code(arg_name)
                    if arg.is_tensorlike():
                        tensor_args.append(arg_name)
                res_code += f"{result_var} = {api_name}_class(*{arg_name})\n"
                call_code = f"{api_name}_class(*{arg_name})\n"
        else:
            res_code += f"{result_var} = {api.api}({arg_str})\n"
            call_code = f"{api.api}({arg_str})\n"
        invocation = self._to_invocation_code(arg_code, res_code, res=result_var, tensor_args=tensor_args, call_code=call_code, **kwargs)
        return invocation


    def _to_invocation_code(self, arg_code, res_code, res=None, tensor_args=None, use_try=False, err_name="", 
        wrap_device=False, device_name="", time_it=False, time_var="", track_grad=False, grad_value=False, call_code=None)-> str:
        if track_grad and tensor_args != None:
            res_code = self.wrap_gradient(res_code, tensor_args, res)
        if track_grad and grad_value == True:
            res_func_code = self.wrap_function(call_code, tensor_args)
            res_code = res_code + res_func_code
            if tensor_args != []:
                comp_grad_code = self.wrap_compgrad(tensor_args)
                res_code += comp_grad_code

        if time_it:
            res_code = res_code + self.wrap_time(res_code, time_var)
        code = arg_code + res_code
        inv_code = code
        if wrap_device:
            inv_code = self.wrap_device(inv_code, device=device_name)
        if use_try:
            inv_code = self.wrap_try(inv_code, error_var=err_name)
        return inv_code

    def _arg_to_diff_code(self, arg:TFArgument, var_name, low_precision=False) -> str:
        if arg.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(arg.value)):
                code += self._arg_to_diff_code(arg.value[i], (f"{var_name}_{i}", low_precision))
                arg_name_list += f"{var_name}_{i},"
            if arg.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif arg.type == ArgType.TF_OBJECT:
            return "%s = None\n" % (var_name)
        elif arg.type == ArgType.TF_DTYPE:
            return "%s = tf.%s\n" % (var_name, arg.value.name)
        elif arg.type in arg._tensor_arg_dtypes:
            code = f"{var_name} = tf.identity({var_name}_tensor)\n"
            if not low_precision:
                code += f"{var_name} = tf.cast({var_name}, tf.{arg.dtype.name})\n"
            if arg.type == ArgType.TF_VARIABLE:
                code += "%s = tf.Variable(%s)\n" % (var_name, var_name)
            return code
        return ""

    def _to_diff_arg_code(self, api, prefix="arg", low_precision=False):
        args = []
        kwargs = {}
        for key in api.args.keys():
            if "parameter:" in key:
                args.append(api.args[key])
            elif key != "output_signature" and key != API_INVOCATION_INPUT_KEY:
                kwargs[key] = api.args[key]

        arg_code = ""
        arg_str = ""
        index = 0
        for arg in args:
            arg_code += self._arg_to_diff_code(arg, f"{prefix}_{index}", low_precision=low_precision)
            arg_str += f"{prefix}_{index},"
            index += 1
        for key, arg in kwargs.items():
            arg_code += self._arg_to_diff_code(arg, key, low_precision=low_precision)
            arg_str += "%s=%s," % (key, key)
        return arg_code, arg_str


    def _api_to_diff_code(self, api, prefix="arg", res="", low_precision=False, **kwargs) -> str:
        
        inputs = None
        input_name = ''
        if API_INVOCATION_INPUT_KEY in api.args:
            inputs = api.args[API_INVOCATION_INPUT_KEY]
        if inputs:
            input_name = f"{prefix}_input"

        arg_code, arg_str = self._to_diff_arg_code(api, prefix=prefix, low_precision=low_precision)
        res_code = ""
        if api.is_class:
            cls_name = f"{prefix}_class"
            res_code = f""
            if inputs:
                arg_code += self._arg_to_diff_code(inputs, input_name, low_precision=low_precision)
                res_code += f"{RES_KEY}[\"{res}\"] = {cls_name}(*{input_name})\n"
        else:
            res_code = f"{RES_KEY}[\"{res}\"] = {api.api}({arg_str})\n"
        
        invocation = self._to_invocation_code(arg_code, res_code, **kwargs)
        return invocation

class CrashOracleTF(OracleTF):

    def api_to_code(self, api:TFAPI):
        code = self._api_to_code(api, res=RESULT_KEY, use_try=True, err_name=ERROR_KEY)
        return code
        


    def run_and_check(self, code, success_dir, fail_dir, bug_dir, error_dir, crash_dir):
        
        # Randomize
        random_seed = randint(7, 100007)
        code = f"tf.random.set_seed({random_seed})\n" + code
        status = ResultType.FAIL

        write_code = "import tensorflow as tf\nimport numpy as np\nresults = dict()\n" + code + \
            self.pretty_print_str + "\npretty_print(results)\n"
        

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        timeout = 10
        proc = multiprocessing.Process(target=run_code_proc, args=[code, return_dict])
        proc.start()
        proc.join(timeout)
        
        exitcode = proc.exitcode
        if exitcode != 0:
            status = ResultType.CRASH
            tf_lib_def.write_to_dir(crash_dir, write_code)
            
        results = return_dict
        
        if RUN_ERROR_KEY in results:
            status = ResultType.ERROR
            tf_lib_def.write_to_dir(error_dir, write_code)
        else:
            if ERROR_KEY in results:
                status = ResultType.FAIL
                tf_lib_def.write_to_dir(fail_dir, write_code)
            else:
                status = ResultType.SUCCESS
                tf_lib_def.write_to_dir(success_dir, write_code)
        
        return status        


class CudaOracleTF(OracleTF):
    def api_to_code(self, api:TFAPI):
        cpu_code = self._api_to_code(api, res=RES_CPU_KEY, 
            use_try=True, err_name=ERR_CPU_KEY, wrap_device=True, device_name="CPU")
        gpu_code = self._api_to_diff_code(api, res=RES_GPU_KEY,
            use_try=True, err_name=ERR_GPU_KEY, wrap_device=True, device_name="GPU:0")
        
        code = cpu_code + gpu_code
        return self.wrap_try(code, ERROR_KEY)
        


    def run_and_check(self, code, success_dir, fail_dir, bug_dir, error_dir, crash_dir):
        
        # Randomize
        random_seed = randint(7, 100007)
        code = f"tf.random.set_seed({random_seed})\n" + code
        status = ResultType.FAIL

        write_code = "import tensorflow as tf\nimport numpy as np\nresults = dict()\n" + code + \
            self.pretty_print_str + "\npretty_print(results)\n"
        

        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        
        timeout = 10
        proc = multiprocessing.Process(target=run_code_proc, args=[code, return_dict])
        proc.start()
        proc.join(timeout)
        
        exitcode = proc.exitcode
        if exitcode != 0:
            status = ResultType.CRASH
            tf_lib_def.write_to_dir(crash_dir, write_code)
            
        results = return_dict
        
        if RUN_ERROR_KEY in results:
            status = ResultType.ERROR
            tf_lib_def.write_to_dir(error_dir, write_code)
        else:
            if ERROR_KEY in results:
                status = ResultType.FAIL
                tf_lib_def.write_to_dir(fail_dir, write_code)
            else:
                cpu_err = ""
                gpu_err = ""
                if ERR_CPU_KEY in results:
                    cpu_err = results[ERR_CPU_KEY].split(",")[0]
                if ERR_GPU_KEY in results:
                    gpu_err = results[ERR_GPU_KEY].split(",")[0]
                if cpu_err != gpu_err:
                    status = ResultType.BUG
                    tf_lib_def.write_to_dir(bug_dir, write_code)
                if ERR_CPU_KEY in results and \
                    ERR_GPU_KEY in results: 
                        status = ResultType.FAIL
                else:
                    print(results)
                    res_cpu = results[RES_CPU_KEY]
                    res_gpu = results[RES_GPU_KEY]
                    if TFLibrary.is_equal(res_cpu, res_gpu):
                        status = ResultType.SUCCESS
                        tf_lib_def.write_to_dir(success_dir, write_code)
                    else:
                        status = ResultType.BUG
                        tf_lib_def.write_to_dir(bug_dir, write_code)
        
        return status        


def run_code_proc(code, results):
    import tensorflow as tf
    import numpy as np
    err = None
    try:
        exec(code)
    except Exception as e:
        err = type(e).__name__ + "," + str(e).replace("\n", "")
    if err is not None:
        results[RUN_ERROR_KEY] = err

class PrecisionOracleTF(OracleTF):
    def api_to_code(self, api:TFAPI):
        low_code = self._api_to_code(api, res=RES_LOW_KEY, low_precision=True,
            use_try=True, err_name=ERR_LOW_KEY, time_it=True, time_var=TIME_LOW_KEY)
        high_code = self._api_to_diff_code(api, res=RES_HIGH_KEY,
            use_try=True, err_name=ERR_HIGH_KEY, time_it=True, time_var=TIME_HIGH_KEY)
        code = low_code + high_code
        return self.wrap_try(code, ERROR_KEY)
