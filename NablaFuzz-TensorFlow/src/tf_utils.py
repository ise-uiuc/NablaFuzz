import os
import numpy as np
import tensorflow as tf

from pathlib import Path
from enum import Enum, IntEnum, auto
ATOL = 1e-1
RTOL = 1e-3

class ResultType(Enum):
    RANDOM = auto()
    SUCCESS = auto() 
    FAIL = auto() 
    CRASH = auto()
    BUG = auto()
    ERROR = auto() 
    STATUS_MISMATCH = auto()
    VALUE_MISMATCH = auto()
    IMPLEMENTATION_ERROR = auto()

    ARGS_FAIL = auto()
    EXEC_FAIL = auto()
    GRAD_FAIL = auto()
    GRAD_NOT_COMPUTED = auto()
    GRAD_VALUE_MISMATCH = auto()
    
    DIRECT_NAN = auto()
    DIRECT_FAIL = auto()

    DIRECT_CRASH = auto()
    REV_CRASH = auto()
    FWD_CRASH = auto()

    REV_GRAD_FAIL = auto()
    REV_STATUS_MISMATCH = auto()
    
    FWD_GRAD_FAIL = auto()
    FWD_STATUS_MISMATCH = auto()
    
    ND_FAIL = auto()
    
BUGGY_RESULT_TYPES = [
    ResultType.STATUS_MISMATCH,
    ResultType.VALUE_MISMATCH,
    ResultType.GRAD_VALUE_MISMATCH,
    ResultType.DIRECT_CRASH,
    ResultType.REV_CRASH,
    ResultType.FWD_CRASH,
    ResultType.REV_GRAD_FAIL,
    ResultType.FWD_GRAD_FAIL,
    ResultType.REV_STATUS_MISMATCH,
    ResultType.FWD_STATUS_MISMATCH,
    ResultType.ND_FAIL,
]

def is_complex_tensor_or_nparray(t):
    if t is None: return False
    return tf.convert_to_tensor(t).dtype.is_complex
def is_gradient_tensor_equal(g1, g2, atol=ATOL, rtol=RTOL, equal_nan=True):
    if (g1 is None) and (g2 is None):
        return True
    if (g1 is None) or (g2 is None):
        return False
    if tf.size(g1) > 0 or tf.size(g2) > 0:
        if is_complex_tensor_or_nparray(g1):
            if not is_complex_tensor_or_nparray(g2): 
                return False
            r1 = tf.math.real(g1)
            r2 = tf.math.real(g2)
            i1 = tf.math.imag(g1)
            i2 = tf.math.imag(g2)
            return is_gradient_tensor_equal(i1, i2, rtol=rtol, atol=atol, equal_nan=equal_nan) \
                and is_gradient_tensor_equal(r1, r2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        mask = tf.math.is_finite(g1)
        g1 = tf.boolean_mask(g1, mask)
        g2 = tf.boolean_mask(g2, mask)
        if g1.dtype == tf.bfloat16:
            g1 = tf.cast(g1, tf.float16)
        if g2.dtype == tf.bfloat16:
            g2 = tf.cast(g2, tf.float16)
        status = np.allclose(g1, g2, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if not status:
            return False
    return True

def is_gradient_equal(grad1, grad2, atol=ATOL, rtol=RTOL, equal_nan=True):
    """ Check if the two computed gradient list is equal. """
    error = 0
    if (grad1 is None) and (grad2 is None):
        return True
    if (grad1 is None) or (grad2 is None):
        return False
    
    if len(grad1) != len(grad2):
        return False
        
    for j_t, j_n in zip(grad1, grad2):
        if not is_gradient_tensor_equal(j_t, j_n):
            return False
    return True
 

def is_equal(x, y, rtol=ATOL, atol=RTOL, equal_nan=True):
    from classes.tf_library import TFLibrary
    return TFLibrary.is_equal(x, y, rtol, atol, equal_nan)

def allow_error(err):
    _allow_errors = [
        "not supported",
        "LookupError,gradient registry has no entry for",
        "was expected to be a double tensor but is a float tensor",
        "gradient_tape/UnsortedSegmentSum_1/pfor/UnsortedSegmentSum",
        "was expected to be a float tensor but is a double ",
        "Empty tensor with nonzero gradients",
        "double is not in the list of allowed values"
    ]

    if any([allowed_e.lower() in err.lower() for allowed_e in _allow_errors]):
        return True

    _allowed_templates = [
        ["tensor conversion requested", "dtype", "float32", "float64" ],
        ["LookupError", "gradient registry has no entry for"],
        ["Expected",  "dtype", "float32", "got", "tf.float64"],
        ["UFuncTypeError", "Cannot cast ufunc 'add' output from dtype", "complex", "to", "float64", "with casting rule 'same_kind'"]
    ]
    for template in _allowed_templates:
        if all([wd.lower() in err.lower() for wd in template]): return True
    return False

def pretty_print(res, msg=""):
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


def write_to_dir(dir, code, maxcnt=None):
    if not os.path.exists(dir):
        os.makedirs(dir)
    filenames = os.listdir(dir)
    max_name = 0
    for name in filenames:
        max_name = max(max_name, int(name.replace(".py", "")))
    if maxcnt != None:
        if max_name > maxcnt:
            return ""
    new_name = str(max_name + 1) + ".py"
    with open(os.path.join(dir, new_name), "w") as f:
        f.write(code)
    return new_name


def load_result_file(fp):
    """ Load result files.

    Each line can be one of:
    tf.abs
    tf.abs-1
    tf.abs-1 ResultType.GRAD_VALUE_MISMATCH
    tf.abs-1 Non-Diff    
    """
    if not os.path.exists(fp):
        return {}
    results = {}
    with open(fp, "r") as f:
        data = f.readlines()
    for line in data:
        if " " in line:
            api_key, result = line.strip().split(" ", 1)
        else:
            api_key = line.strip()
            result = "unknown"
        if "-" in api_key:
            api_name, key = api_key.split("-")
        else:
            api_name = api_key.strip()
            key = "empty"
        if api_name not in results:
            results[api_name] = {}
        results[api_name][key] = result
    return results

def dump_res(result, fn):
    if fn is None: return
    with open(fn, 'a') as f:
        f.write(result + '\n')

skip_api = [
    "tf.broadcast_to",
    "tf.compat.v1.broadcast_to",
    "tf.compat.v1.keras.layers.Convolution1DTranspose",
    "tf.compat.v1.keras.layers.SeparableConv2D",
    "tf.compat.v1.keras.layers.SeparableConvolution2D",
    "tf.compat.v1.keras.layers.experimental.RandomFourierFeatures",
    "tf.compat.v1.signal.frame", 
    "tf.experimental.numpy.broadcast_to",
    "tf.image.resize",
    "tf.keras.layers.LSTM",
    "tf.keras.layers.LocallyConnected1D",
    "tf.compat.v1.keras.layers.Convolution1D",
    "tf.compat.v1.keras.layers.Conv2D",
    "tf.signal.frame",
    "tf.tile",
    "tf.nn.conv2d",
    "tf.compat.v1.nn.conv2d",
    "tf.compat.v1.debugging.assert_less",
    "tf.compat.v1.mixed_precision.FixedLossScale",
    "tf.compat.v1.less_equal",
    "tf.compat.v1.matrix_triangular_solve",
    "tf.compat.v1.minimum"
]

def if_skip(api_name, key=None, run_mode="run", mode="mutant"):
    """ Check if skips api.
        run_mode is one of: run, Ccov, Pythoncov
    """
    if api_name in skip_api: return True
    return False
