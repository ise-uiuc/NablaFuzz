from classes.jax_api import *
from classes.library import Library
from constant.keys import *
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes as _dtypes


_default_tolerance = {
    _dtypes.float0: 0,
    np.dtype(np.bool_): 0,
    np.dtype(np.int8): 0,
    np.dtype(np.int16): 0,
    np.dtype(np.int32): 0,
    np.dtype(np.int64): 0,
    np.dtype(np.uint8): 0,
    np.dtype(np.uint16): 0,
    np.dtype(np.uint32): 0,
    np.dtype(np.uint64): 0,
    np.dtype(_dtypes.bfloat16): 1e-2,
    np.dtype(np.float16): 1e-3,
    np.dtype(np.float32): 1e-6,
    np.dtype(np.float64): 1e-15,
    np.dtype(np.complex64): 1e-6,
    np.dtype(np.complex128): 1e-15,
}


def default_tolerance():
    return _default_tolerance


default_gradient_tolerance = {
    np.dtype(_dtypes.bfloat16): 1e-1,
    np.dtype(np.float16): 1e-2,
    np.dtype(np.float32): 2e-3,
    np.dtype(np.float64): 1e-5,
    np.dtype(np.complex64): 1e-3,
    np.dtype(np.complex128): 1e-5,
}


def tolerance(dtype, tol=None):
    tol = {} if tol is None else tol
    if not isinstance(tol, dict):
        return tol
    tol = {np.dtype(key): value for key, value in tol.items()}
    dtype = _dtypes.canonicalize_dtype(np.dtype(dtype))
    return tol.get(dtype, default_tolerance()[dtype])


def _assert_numpy_allclose(a, b, atol=None, rtol=None, err_msg=""):
    if a.dtype == b.dtype == _dtypes.float0:
        np.testing.assert_array_equal(a, b, err_msg=err_msg)
        return
    a = a.astype(np.float32) if a.dtype == _dtypes.bfloat16 else a
    b = b.astype(np.float32) if b.dtype == _dtypes.bfloat16 else b
    kw = {}
    if atol:
        kw["atol"] = atol
    if rtol:
        kw["rtol"] = rtol
    with np.errstate(invalid="ignore"):
        # TODO(phawkins): surprisingly, assert_allclose sometimes reports invalid
        # value errors. It should not do that.
        np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


class JaxLibrary(Library):
    def __init__(self, directory) -> None:
        super().__init__(directory)

    # TODO:
    @staticmethod
    def is_equal(a, b, atol=1e-1, rtol=1e-3, ignore_nan=False):
        type_a = JaxArgument.get_type(a)
        type_b = JaxArgument.get_type(b)

        if type_a != type_b:
            return False

        if type_a == ArgType.JAX_ARRAY:
            a, b = np.asarray(a), np.asarray(b)
            if not a.shape == b.shape:
                return False

            any_nan_a = jnp.any(jnp.isnan(a))
            any_nan_b = jnp.any(jnp.isnan(b))
            if any_nan_a or any_nan_b:
                if any_nan_a != any_nan_b:
                    return False
                elif ignore_nan:
                    return True

            # atol = max(tolerance(a.dtype, atol), tolerance(b.dtype, atol))
            # rtol = max(tolerance(a.dtype, rtol), tolerance(b.dtype, rtol))
            try:
                _assert_numpy_allclose(a, b, atol=atol, rtol=rtol)
            except Exception:
                return False
            else:
                return True
        elif type_a in [ArgType.TUPLE, ArgType.LIST]:
            if len(a) != len(b):
                return False
            equal = True
            for i in range(len(a)):
                equal = equal and JaxLibrary.is_equal(
                    a[i], b[i], atol, rtol, ignore_nan
                )
            return equal
        elif type_a == ArgType.NULL:
            return True
        else:
            return a == b

    @staticmethod
    def run_code(code):
        results = dict()
        results[ERROR_KEY] = None
        results[ERR_CPU_KEY] = None
        results[ERR_GPU_KEY] = None
        results[ERR_1] = None
        results[ERR_2] = None
        results[GRAD_ERR_1] = None
        results[GRAD_ERR_2] = None
        results[ERR_FN] = None
        results[ERR_CHECK] = None
        error = None
        try:
            exec(code)
        except Exception as e:
            error = str(e)
        return results, error
