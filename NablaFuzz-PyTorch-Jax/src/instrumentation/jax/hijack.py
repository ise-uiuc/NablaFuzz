import jax
from jax.instrumentation.decorators import (
    dump_signature_of_class,
    dump_signature_of_function,
)
import inspect
from jax.instrumentation.utils import dump_data, load_data


def hijack_api(obj, func_name_str, mode=""):
    func_name_list = func_name_str.split(".")
    func_name = func_name_list[-1]

    module_obj = obj
    # print(func_name_str)
    if len(func_name_list) > 2:
        for module_name in func_name_list[1:-1]:
            module_obj = getattr(module_obj, module_name)
    orig_func = getattr(module_obj, func_name)

    def is_class(x):
        return inspect.isclass(x)

    def is_callable(x):
        return callable(x)

    if mode == "function":
        wrapped_func = dump_signature_of_function(orig_func, func_name_str)
    elif mode == "class":
        wrapped_func = dump_signature_of_class(orig_func, func_name_str)
    else:
        if is_class(orig_func):
            wrapped_func = dump_signature_of_class(orig_func, func_name_str)
        elif is_callable(orig_func):
            wrapped_func = dump_signature_of_function(orig_func, func_name_str)
        else:
            wrapped_func = orig_func
    setattr(module_obj, func_name, wrapped_func)


def hijack():
    instrument_error = "instrument-error.log"
    instrument_apis = "instrument-apis.log"
    dump_data("", instrument_error)
    dump_data("", instrument_apis)

    lines = load_data(__file__.replace("hijack.py", "jax-apis.txt"), True)
    skipped = []
    for l in lines:
        l = l.strip()
        if l not in skipped:
            try:
                hijack_api(jax, l)
            except Exception as e:
                dump_data(f"{l}: {str(e)}\n", instrument_error, "a")
                # print(e)
            else:
                dump_data(f"{l}\n", instrument_apis, "a")
