from jax.instrumentation.utils import dump_data
import numpy as np
import jax
import json

# from jax._src import test_util as jtu


def is_iterable(v):
    return isinstance(v, list) or isinstance(v, tuple)


def get_var_class_full_name(v):
    return v.__class__.__module__ + "." + v.__class__.__name__


def json_serialize_value(v):
    """Return the json serializable value of v."""
    try:
        return json.dumps(v)
    except Exception as e:
        return str(v)


def json_deserialize_value(v):
    """Return the json serializable value of v."""
    try:
        return json.loads(v)
    except Exception as e:
        return v


class SignatureHandler:
    python_built_in_types = [
        str,
        int,
        float,
        complex,
        list,
        tuple,
        range,
        dict,
        set,
        frozenset,
        bool,
        bytes,
        bytearray,
        memoryview,
    ]
    np_dtypes = [
        np.bool_,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.complex64,
        np.complex128,
    ]
    _error_log = "error-tracer.log"
    _unhandle_log = "unhandle.log"

    def __init__(self) -> None:
        dump_data("", self._error_log, "w")
        dump_data("", self._unhandle_log, "w")

    def get_var_signature(self, v):
        if self.check_var_nptype(v):
            return self.get_nptype_signature(v)

        if self.check_var_npraw(v):
            return self.get_npraw_signature(v)

        if self.check_var_nparray(v):
            return self.get_nparray_signature(v)

        if self.check_var_list(v):
            return self.get_list_signature(v)

        if self.check_var_raw(v):
            return self.get_raw_signature(v)

        if self.check_var_jaxarray(v):
            return self.get_jaxarray_signature(v)

        if self.check_var_jaxtracer(v):
            return self.get_jaxtracer_signature(v)

        return self.get_other_signature(v)

    def check_var_raw(self, v):
        """Check if a variable is a python built-in object."""
        if type(v) in self.python_built_in_types:
            return True
        else:
            return False

    def get_raw_signature(self, v):
        s = dict()
        s["Label"] = "raw"
        s["value"] = json_serialize_value(v)
        return s

    def check_var_list(self, v):
        """Check if a variable is a list."""
        return isinstance(v, list) or isinstance(v, tuple)

    def get_list_signature(self, v):
        s = dict()
        s["Label"] = "list"
        s["value"] = [self.get_var_signature(e) for e in v]
        return s

    def check_var_tuple(self, v):
        """Check if a variable is a list."""
        return isinstance(v, list)

    def get_tuple_signature(self, v):
        s = dict()
        s["Label"] = "tuple"
        s["value"] = (self.get_var_signature(e) for e in v)
        return s

    def check_var_npraw(self, v):
        if isinstance(
            v, (np.integer, np.floating, np.complexfloating, np.bool_)
        ):
            return True
        else:
            return False

    def get_npraw_signature(self, v):
        s = dict()
        s["Label"] = "npraw"
        s["value"] = str(v)
        s["dtype"] = self.get_dtype(v.dtype)
        return s

    def check_var_nptype(self, v):
        if isinstance(v, (type, np.dtype)):
            return True
        else:
            return False

    def get_nptype_signature(self, v):
        s = dict()
        s["Label"] = "nptype"
        s["value"] = self.get_dtype(v)
        return s

    def check_var_nparray(self, v):
        return isinstance(v, np.ndarray)

    def get_nparray_signature(self, v):
        shape = v.shape
        dtype = self.get_dtype(v.dtype)

        scalar = self.try_get_scalar(v, shape, dtype)
        if scalar != None:
            return scalar

        s = dict()
        s["Label"] = "nparray"
        s["shape"] = shape
        s["dtype"] = dtype
        return s

    def check_var_jaxarray(self, v):
        if isinstance(v, jax.numpy.DeviceArray):
            return True
        else:
            return False

    def get_jaxarray_signature(self, v):
        shape = v.shape
        dtype = self.get_dtype(v.dtype)

        scalar = self.try_get_scalar(v, shape, dtype)
        if scalar != None:
            return scalar

        s = dict()
        s["Label"] = "jaxarray"
        s["shape"] = shape
        s["dtype"] = dtype
        return s

    def check_var_jaxtracer(self, v):
        # if isinstance(v, jax.interpreters.partial_eval.DynamicJaxprTracer):
        if "Tracer" in str(type(v)):
            return True
        else:
            return False

    def get_jaxtracer_signature(self, v):
        s = dict()
        s["Label"] = "tracer"
        try:
            s["shape"] = v.aval.shape
            s["dtype"] = self.get_dtype(v.aval.dtype)
            # dump_data(f"Type: {str(type(v.aval))}, Value: {str(v.aval)}\n", 'info-tracer.log', 'a')
        except Exception as e:
            s["shape"] = []
            s["dtype"] = "float64"
            print(e)
            dump_data(
                f"Tracer Error: {str(v)}, {str(e)}\n", self._error_log, "a"
            )
        return s

    def get_other_signature(self, v):
        s = dict()
        s["Label"] = "other"
        s["type"] = str(type(v))
        dump_data(f"{type(v)}\n", self._unhandle_log, "a")
        return s

    def try_get_scalar(self, v, shape, dtype):
        if len(shape) == 0 or (len(shape) == 1 and shape[0] == 1):
            s = dict()
            try:
                s["Label"] = "scalar"
                s["dtype"] = dtype
                s["value"] = json_serialize_value(v.item())
            except Exception:
                return None
            else:
                return s
        else:
            return None

    def get_dtype(self, v):
        if not isinstance(v, (type, np.dtype)):
            dump_data("Dtype Error: Not a dtype\n", self._error_log, "a")
            return "NOT-A-DTYPE"
        elif hasattr(v, "name"):
            return v.name
        else:
            return v.__name__
