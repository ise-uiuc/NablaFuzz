from jax.instrumentation.signature_handler import SignatureHandler
from jax.instrumentation.write_tools import write_fn
from jax.instrumentation.utils import dump_data

sighdl = SignatureHandler()
class_error_file = "error-class.log"
dump_data("", class_error_file, "w")


def build_param_dict(*args, **kwargs):
    param_dict = dict()
    for ind, arg in enumerate(args):
        param_dict["parameter:%d" % ind] = sighdl.get_var_signature(arg)
    for key, value in kwargs.items():
        if key == "name":
            continue
        param_dict[key] = sighdl.get_var_signature(value)
    param_dict = dict(param_dict)
    return param_dict


def get_signature_for_tensors(t):
    return sighdl.get_var_signature(t)


def dump_signature_of_class(klass, class_name):
    if not hasattr(klass, "__call__"):
        return klass
    if klass.__init__ == object.__init__:
        dump_data(f"{class_name}: No Init\n", class_error_file, "a")
        return klass
    old_init = klass.__init__
    old_call = klass.__call__
    init_params = dict()

    def new_init(self, *args, **kwargs):
        nonlocal init_params
        try:
            init_params = build_param_dict(*args, **kwargs)
        except Exception as e:
            print(e.message)
        try:
            old_init(self, *args, **kwargs)
        except Exception as e:
            dump_data(f"{class_name}: {str(e)}\n", class_error_file, "a")
            import json

            dump_data(
                json.dumps(init_params, indent=2) + "\n", class_error_file, "a"
            )
            raise Exception("Class Wrapper Init Error")

    def new_call(self, *inputs, **kwargs):
        nonlocal init_params

        input_signature = build_param_dict(*inputs, **kwargs)
        outputs = old_call(self, *inputs, **kwargs)
        write_fn(
            self.__class__.__module__ + "." + self.__class__.__name__,
            init_params,
            input_signature,
        )
        return outputs

    klass.__init__ = new_init
    klass.__call__ = new_call
    return klass


from functools import wraps


def dump_signature_of_function(func, hint):
    @wraps(func)
    def wrapper(*args, **kwargs):
        import json
        import os

        outputs = func(*args, **kwargs)
        param_dict = build_param_dict(*args, **kwargs)
        write_fn(hint, param_dict, None)
        return outputs

    if not callable(func):
        return func

    return wrapper
