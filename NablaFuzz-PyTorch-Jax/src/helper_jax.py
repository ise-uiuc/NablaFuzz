from time import time
import jax
import jax.numpy as jnp
import numpy as np
from jax._src import dtypes as _dtypes
from itertools import product
import json

NEIGHBOR_NUM = 5
NEIGHBOR_STEP = 1e-4
EPS = 1e-6
ATOL = 1e-1
RTOL = 1e-3


def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)


def get_tensors(a):
    if isinstance(a, (tuple, list)):
        ret = []
        for t in a:
            ret += get_tensors(t)
        return ret
    else:
        return [a]


def debug_equal(a, b):
    a_list = get_tensors(a)
    b_list = get_tensors(b)
    a_len = len(a_list)
    b_len = len(b_list)
    if a_len != b_len:
        print(a_list)
        print(b_list)
        print("Length is not equal")
        return
    for i in range(a_len):
        print(i)
        print(a_list[i])
        print(b_list[i])
        _assert_numpy_allclose(a, b, 1e-1, 1e-3)


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
        np.testing.assert_allclose(a, b, **kw, err_msg=err_msg)


def allow_error(err):
    _allow_errors = [
        "grad requires real-valued outputs (output dtype that is a sub-dtype of np.floating)",
        "Non-hashable static arguments are not supported, as this can lead to unexpected cache-misses",
        "not implemented",
        "not supported",
        "requires real-valued outputs",
        "Unimplemented case",
        "not a valid JAX type",
        "bytes-like object is require",
        "at least one array or dtype is require",
        "Unsupported",
        "differentiation does not work",
    ]
    for a_err in _allow_errors:
        if a_err in err:
            return True
    return False


def is_nan_output(res):
    try:
        return jax.numpy.any(jax.numpy.isnan(res))
    except Exception:
        return False


def any_complex_output(res):
    if isinstance(res, (list, tuple)):
        has_complex = False
        for v in res:
            has_complex = has_complex or any_complex_output(v)
        return has_complex
    else:
        try:
            return jax.numpy.iscomplex(res)
        except Exception:
            return False


def _iter_arg(x):
    for d_idx, x_idx in enumerate(product(*[range(m) for m in x.shape])):
        yield x, x_idx, d_idx


def _get_numerical_jacobian_wrt_inp(fn, inputs, inp_idx, eps=EPS, fast=False):
    def get_wrap_fn(fn, inp_idx):
        def new_fn(inp):
            arg_list = [
                inp if idx == inp_idx else arg
                for idx, arg in enumerate(inputs)
            ]
            return fn(*arg_list)

        return new_fn

    def compute(outa, outb):
        if isinstance(outa, (tuple, list)):
            ret = []
            for i in range(len(outa)):
                ret += [((outa[i] - outb[i]) / (2 * eps)).reshape(-1)]
            return tuple(ret)
        else:
            ret = (outa - outb) / (2 * eps)
            return ret.reshape(-1)

    wrap_fn = get_wrap_fn(fn, inp_idx)
    inp = inputs[inp_idx]
    jacnum = []

    if fast:
        rng = jax.random.PRNGKey(0)
        d = 1
        for i in inp.shape:
            d *= i
        delta = jax.random.ball(rng, d).reshape(inp.shape) * eps
        # rng = np.random.RandomState(1)
        # delta = rng.uniform(0.5 * eps, 1 * eps, inp.shape)
        # print(delta)

        a = inp + delta
        outa = wrap_fn(a)

        b = inp - delta
        outb = wrap_fn(b)
        jacnum.append(compute(outa, outb))
    else:
        for x, x_idx, d_idx in _iter_arg(inp):
            a = x.at[x_idx].set(x[x_idx] + eps)
            outa = wrap_fn(a)

            b = x.at[x_idx].set(x[x_idx] - eps)
            outb = wrap_fn(b)
            jacnum.append(compute(outa, outb))
    return jax.numpy.array(jacnum)


def get_numerical_jacobian(fn, inputs, eps=EPS, fast=False):
    inputs = _as_tuple(inputs)
    jacobians = []
    for inp_idx, inp in enumerate(inputs):
        jacobians.append(
            _get_numerical_jacobian_wrt_inp(
                fn, inputs, inp_idx, eps=eps, fast=fast
            )
        )
    return jacobians


def gradcheck(
    fn, inputs, eps=EPS, atol=ATOL, rtol=RTOL, mode="rev", output_err=True
):
    inputs = _as_tuple(inputs)
    jac_num = get_numerical_jacobian(fn, inputs, eps)
    if mode == "rev":
        jac = jax.jacrev(fn, tuple(range(len(inputs))))(*inputs)
    elif mode == "fwd":
        jac = jax.jacfwd(fn, tuple(range(len(inputs))))(*inputs)
    else:
        assert 0, "Not Support: " + mode

    for idx, jac_num_i in enumerate(jac_num):
        if output_err:
            _assert_numpy_allclose(
                jac_num_i.reshape(-1),
                jac[idx].reshape(-1),
                atol=atol,
                rtol=rtol,
                err_msg=f"Jacobian mismatch for {idx} argument",
            )
        else:
            assert jax.numpy.allclose(
                jac_num_i.reshape(-1),
                jac[idx].reshape(-1),
                atol=atol,
                rtol=rtol,
            ), "Not equal to tolerance"


def is_differentiable(
    fn,
    inputs,
    neighbor_step=NEIGHBOR_STEP,
    eps=EPS,
    atol=ATOL,
    rtol=RTOL,
    fast=False,
):
    def get_neighbor(inputs, type="left"):
        res = []
        for inp in inputs:
            if type == "left":
                temp = inp - neighbor_step
            elif type == "right":
                temp = inp + neighbor_step
            else:
                # sign = jax.numpy.sign(np.random.randn(*inp.shape))
                # temp = inp + sign * neighbor_step
                #
                delta = np.random.uniform(-eps, eps, inp.shape)
                temp = inp + delta
            res.append(temp)
        return res

    inputs = _as_tuple(inputs)
    out = fn(*inputs)
    jac = get_numerical_jacobian(fn, inputs, eps, fast=fast)
    for _ in range(NEIGHBOR_NUM):
        rand_inputs = get_neighbor(inputs, "rand")
        rand_out = fn(*rand_inputs)
        if not jax.numpy.allclose(out, rand_out, atol=atol, rtol=rtol):
            return False

        rand_jac = get_numerical_jacobian(fn, rand_inputs, eps, fast=fast)
        for idx, inp_jac in enumerate(jac):
            if not jax.numpy.allclose(
                inp_jac, rand_jac[idx], atol=atol, rtol=rtol
            ):
                return False
    return True


def get_res_info(res):
    # FIXME: it is not good enough
    if isinstance(res, (jax.numpy.DeviceArray,)):
        return json.dumps(
            {"shape": [int(i) for i in res.shape], "dtype": str(res.dtype)}
        )
    elif isinstance(res, (tuple, list)):
        res = []
        for x in res:
            res.append(get_res_info(x))
        return str(res)
    else:
        return "None"


def DirectInv(fn, inputs):
    value = None
    err_msg = ""
    try:
        value = fn(*inputs)
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, value, err_msg


def RevInv(fn, inputs):
    value = None
    err_msg = ""
    gradient = None
    try:
        value, _ = jax.vjp(fn, *inputs)
        argnums = (i for i in range(len(inputs)))
        gradient = jax.jacrev(fn, argnums)(*inputs)
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, value, gradient, err_msg


def FwdInv(fn, inputs):
    value = None
    err_msg = ""
    gradient = None
    try:
        value, _ = jax.linearize(fn, *inputs)
        argnums = (i for i in range(len(inputs)))
        gradient = jax.jacfwd(fn, argnums)(*inputs)
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, value, gradient, err_msg


def get_precise_inputs(inputs):
    precise_inputs = []
    for inp in inputs:
        if "float" in str(inp.dtype):
            precise_inputs.append(
                jax.lax.convert_element_type(inp, jax.numpy.float64)
            )
        else:
            precise_inputs.append(
                jax.lax.convert_element_type(inp, jax.numpy.complex128)
            )
    return tuple(precise_inputs)


def NDCheck(fn, inputs, mode="rev"):
    err_msg = ""
    precise_inputs = get_precise_inputs(inputs)
    try:
        from jax.test_util import check_grads

        check_grads(
            fn,
            precise_inputs,
            order=1,
            modes=(mode,),
            atol=ATOL,
            rtol=RTOL,
            eps=EPS,
        )
    except Exception as e:
        status = "fail"
        err_msg = str(e)
    else:
        status = "success"
    return status, err_msg


def Grad(fn, inputs):
    def wrapper(*args):
        return jax.numpy.sum(fn(*args))

    nums = range(len(inputs))
    return jax.grad(wrapper, nums)


dtype_precision_dict = {
    "bfloat16": 1,
    "float16": 2,
    "float32": 3,
    "complex64": 3,
    "float64": 4,
    "complex128": 4,
}


def dtype_precision(dtype: str):
    dtype = str(dtype)
    if "int" in dtype or "bool" in dtype:
        return 0
    for dt, precision in dtype_precision_dict.items():
        if dt in dtype:
            return precision
    assert 0, f"No such dtype: {dtype}"


def is_high_to_low_precision(inputs, output):
    def _check(output):
        for input in inputs:
            if dtype_precision(input.dtype) > dtype_precision(output.dtype):
                return True
        return False

    if isinstance(output, (tuple, list)):
        for o in output:
            if _check(o):
                return True
        return False
    elif isinstance(output, jnp.DeviceArray):
        return _check(output)
    else:
        print(type(output))
        return False


def Filter(fn, inputs, is_ND=False) -> str:
    def is_nan_grad(grad: list):
        ret = False
        for t in grad:
            if isinstance(t, (tuple, list)):
                for tt in t:
                    ret = ret or jnp.any(jnp.isnan(tt))
            else:
                ret = ret or jnp.any(jnp.isnan(t))
        return ret

    # NaN Check
    _, rev_value, rev_grad, rev_msg = RevInv(fn, inputs)
    _, fwd_value, fwd_grad, fwd_msg = FwdInv(fn, inputs)
    has_rev_grad = rev_grad is not None
    has_fwd_grad = fwd_grad is not None
    if has_rev_grad:
        rev_nan = is_nan_grad(rev_grad)
    if has_fwd_grad:
        fwd_nan = is_nan_grad(fwd_grad)
    if (not has_rev_grad or rev_nan) and (not has_fwd_grad or fwd_nan):
        # print(has_rev_grad, has_fwd_grad)
        # print(rev_msg, fwd_msg)
        return "NaN"
    elif has_rev_grad and has_fwd_grad and rev_nan != fwd_nan:
        return "NaN-error"

    # Presicion
    if is_ND:
        inputs = get_precise_inputs(inputs)
        _, output_value, _ = DirectInv(fn, inputs)
    else:
        output_value = rev_value if rev_value is not None else fwd_value
    if is_high_to_low_precision(inputs, output_value):
        return "Precision"

    # Differentiability
    if not is_differentiable(fn, inputs, fast=False):
        return "Non-Diff"
    else:
        return "Pass"


"""
Below are some test cases
"""


def test_iter():
    a = jax.numpy.array([[0.0, 0.0]])
    for x, x_idx, d_idx in _iter_arg(a):
        print(x, x_idx, d_idx)


def test_num1():
    # a = jax.numpy.array([[1., 1.], [1., 1.]])
    a = jax.numpy.array(1.0 + 1.0j, dtype=jax.numpy.complex128)
    print(jax.numpy.abs(a))
    jac1 = _get_numerical_jacobian_wrt_inp(jax.numpy.abs, (a,), 0).reshape(-1)
    jac2 = jax.jacfwd(jax.numpy.abs)(a).reshape(-1)
    print(jac1)
    print(jac2)
    # print(jax.numpy.allclose(jac1, jac2, atol=1e-2, rtol=1e-2))


def test_check():
    # a = jax.numpy.array([[1.0, 1.0], [1.0, 1.0]])
    b = jax.numpy.array([[1.0, 0.0], [1.0, 1.0]])
    gradcheck(
        jax.numpy.abs,
        (b,),
    )


def test_is_diff():
    b = jax.numpy.array([[1.0, 0.0], [1.0, 1.0]])
    print(is_differentiable(jax.numpy.abs, (b,), fast=True))


def test_check_time():
    def f(output_err):
        try:
            a = jax.numpy.array([[1.0, 0.0], [1.0, 1.0]])
            gradcheck(jax.numpy.abs, (a,), output_err=output_err)
        except Exception:
            pass

    def get_time(output_err):
        st = time()
        for _ in range(100):
            f(output_err)
        print(time() - st)

    get_time(False)
    get_time(True)


def test_inv():
    fn = jax.numpy.sum
    input = jax.numpy.array([1.0, 2.0])
    print(DirectInv(fn, (input,)))
    print(RevInv(fn, (input,)))
    print(FwdInv(fn, (input,)))
    print(NDCheck(fn, (input,)))


# test_num1()
# test_check()
# test_is_diff()
# test_check_time()
# test_inv()
