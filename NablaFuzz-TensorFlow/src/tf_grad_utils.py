import json
import random
import numpy as np
import tensorflow as tf
from tf_utils import *
# from constant.keys import *

TEST_FUNC_EXEC_NAME = "test_func_exec"
ND_DELTA = 1e-6
NUM_DIRECT_REP = 9


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def CopyInputs(inputs, precise=False):
    """ Make a copy of inputs.
    Args:
        inputs: a list of floating/complex tensorflow tensors.
        precise: if set to true, convert to high-precision dtype for ND.
    """
    new_inputs = []
    for x in inputs:
        if precise:
            if x.dtype.is_complex:
                new_inputs.append(tf.identity(tf.cast(x, tf.complex128)))
            elif x.dtype.is_floating:
                new_inputs.append(tf.identity(tf.cast(x, tf.float64)))
            else:
                new_inputs.append(tf.identity(x))
        else:
            new_inputs.append(tf.identity(x))
    return tuple(new_inputs)

def DirectInv(fn, inputs, device='cpu'):
    status = "success"
    value = None
    errmsg = ""
    with tf.device(device):
        try:
            new_inputs = CopyInputs(inputs)
            value = fn(*new_inputs)
        except Exception as e:
            status = "fail"
            errmsg = type(e).__name__ + "," + str(e).replace("\n", "")
    return status, value, errmsg

def RevInvGrad(fn, inputs, device="cpu"):
    status = "success"
    value = None
    gradient = None
    errmsg = ""
    try:
        with tf.device(device):
            new_inputs = CopyInputs(inputs)
            with tf.GradientTape() as tape:
                for x in new_inputs:
                    tape.watch(x)
                value = fn(*new_inputs)
            gradient = tape.gradient(value, new_inputs)
    except Exception as e:
        status = "fail"
        errmsg = type(e).__name__ + "," + str(e).replace("\n", "")
        
    return status, value, gradient, errmsg
def _product(t):
  if isinstance(t, int):
    return t
  else:
    y = 1
    for x in t:
      y *= x
    return y

def RevInvJacobian(fn, inputs, device="cpu"):
    status = "success"
    value = None
    jacobian = None
    errmsg = ""
    try:
        with tf.device(device):
            new_inputs = CopyInputs(inputs)
            with tf.GradientTape() as tape:
                for x in new_inputs:
                    tape.watch(x)
                value = fn(*new_inputs)
            jacobian = tape.jacobian(value, new_inputs)

            y_size = _product(value.shape)
            new_jacobian = []
            for i, x in enumerate(new_inputs):
                if jacobian[i] == None:
                    new_jacobian.append(None)
                    continue
                x_size = _product(x.shape)
                new_jacobian.append(tf.reshape(jacobian[i], (y_size, x_size)).numpy())
            jacobian = new_jacobian
    except Exception as e:
        status = "fail"
        errmsg = type(e).__name__ + "," + str(e).replace("\n", "")
        
    
    return status, value, jacobian, errmsg

def RevInv(fn, inputs, device="cpu", use_jacobian=True):
    if not use_jacobian:
        return RevInvGrad(fn, inputs, device)
    else:
        return RevInvJacobian(fn, inputs, device)

def FwdInvGrad(fn, inputs, device="cpu"):
    status = "success"
    value = None
    gradient = None
    errmsg = ""
    try:
        with tf.device(device):
            new_inputs = CopyInputs(inputs)
            value = fn(*new_inputs)
            m = tf.size(value)
            gradient = []
            for x in inputs:
                n = tf.size(x)
                jac_fwd_x = []
                for i in range(tf.size(x)):
                    tmp = tf.TensorArray(x.dtype,tf.size(x))
                    tmp = tmp.write(i,tf.cast(tf.constant(1),x.dtype))
                    tangents = tf.reshape(tmp.stack(),x.shape)
                    with tf.autodiff.ForwardAccumulator(x,tangents) as acc:
                        value = tf.reduce_sum(fn(*inputs))
                    jvp_i = acc.jvp(value)
                    jac_fwd_x.append(jvp_i)
                jac_fwd_x = tf.convert_to_tensor(jac_fwd_x)
                jac_fwd_x = tf.reshape(jac_fwd_x, x.shape)
                gradient.append(gradient)
            
    except Exception as e:
        status = "fail"
        errmsg = type(e).__name__ + "," + str(e).replace("\n", "")
        
        
    return status, value, gradient, errmsg

def FwdInvJacobian(fn, inputs, device="cpu"):
    status = "success"
    value = None
    jacobian = None
    errmsg = ""
    try:
        with tf.device(device):
            new_inputs = CopyInputs(inputs)
            
            value = fn(*new_inputs)
            m = tf.size(value)

            value = None
            
            jacobian = []
            for x in new_inputs:
                n = tf.size(x)
                jac_fwd_x = []
                for i in range(tf.size(x)):                    
                    tmp = tf.TensorArray(x.dtype,tf.size(x))
                    tmp = tmp.write(i,tf.cast(tf.constant(1),x.dtype))
                    tangents = tf.reshape(tmp.stack(),x.shape)
                    with tf.autodiff.ForwardAccumulator(x,tangents) as acc:
                        value = fn(*new_inputs)
                    jvp_i = acc.jvp(value)
                    if jvp_i is not None:
                        jac_fwd_x.append(jvp_i)
                jac_fwd_x = tf.convert_to_tensor(jac_fwd_x)
                try:
                    jac_fwd_x = tf.transpose(tf.reshape(jac_fwd_x, (n, m)))
                    jac_fwd_x = jac_fwd_x.numpy()
                except:
                    jac_fwd_x = None

                jacobian.append(jac_fwd_x)
            
    except Exception as e:
        status = "fail"
        errmsg = type(e).__name__ + "," + str(e).replace("\n", "")
        
    return status, value, jacobian, errmsg
    

def FwdInv(fn, inputs, device="cpu", use_jacobian=True):
    if not use_jacobian:
        return FwdInvGrad(fn, inputs, device)
    else:
        return FwdInvJacobian(fn, inputs, device)

def NDINV(fn, inputs, device="cpu", delta=ND_DELTA):
    status = "success"
    errmsg = ""
    theoretical = None
    numerical = None
    try:
        with tf.device(device):
            new_inputs = CopyInputs(inputs, precise=True)
            theoretical, numerical = tf.test.compute_gradient(fn, new_inputs, delta=delta)
            
    except Exception as e:
        status = "fail"
        errmsg = type(e).__name__ + "," + str(e).replace("\n", "")
    return status, theoretical, numerical, errmsg

def _as_tuple(x):
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)

def Grad(fn):

    def new_func(*input_args):
        with tf.GradientTape(persistent=True) as g1:
            for x in input_args:
                g1.watch(x)
            outputs = fn(*input_args)
            outputs = _as_tuple(outputs)
        grad_inputs = []
        for output in outputs:
            grad_i = g1.gradient(output, input_args)
            grad_inputs.extend([g for g in grad_i if g is not None])
        return grad_inputs[0]

    return new_func

def check_crash(errmsg):
    crash_msgs = [
        "Internal error"
    ]
    for crash_msg in crash_msgs:
        if crash_msg.lower() in errmsg.lower(): return True
    return False

def testAD(fn, inputs, device, verbose=True):
    # Direct Invoke
    direct_status, direct_value, direct_err = DirectInv(fn, inputs)
    if check_crash(direct_err):
        return ResultType.DIRECT_CRASH

    for _ in range(NUM_DIRECT_REP):
        direct_status_, direct_value_, direct_err_ = DirectInv(fn, inputs)
        if direct_status != direct_status_ or not is_equal(
                direct_value, direct_value_, equal_nan=True):
            return ResultType.RANDOM
        elif not is_equal(direct_value, direct_value_, equal_nan=False):
            return ResultType.DIRECT_NAN
    if len(inputs) == 0:
        return ResultType.GRAD_NOT_COMPUTED
    use_jacobian=True
    rev_status, rev_value, rev_grad, rev_err = RevInv(fn, inputs, use_jacobian=use_jacobian, device=device)
    fwd_status, fwd_value, fwd_grad, fwd_err = FwdInv(fn, inputs, use_jacobian=use_jacobian, device=device)
    
    if check_crash(rev_err):
        return ResultType.REV_CRASH
    if check_crash(fwd_err):
        return ResultType.FWD_CRASH
    
    if rev_status != direct_status:
        if rev_status == "fail":
            if not is_equal(direct_value, rev_value):
                return ResultType.VALUE_MISMATCH
            if not allow_error(rev_err):
                return ResultType.REV_GRAD_FAIL
        else:
            return ResultType.REV_STATUS_MISMATCH
    elif direct_status == "success":
        if not is_equal(direct_value, rev_value):
            return ResultType.VALUE_MISMATCH
            
    if fwd_status != direct_status:
        if fwd_status == "fail":
            if not is_equal(direct_value, fwd_value):
                return ResultType.VALUE_MISMATCH
            if not allow_error(fwd_err):
                return ResultType.FWD_GRAD_FAIL
        else:
            return ResultType.FWD_STATUS_MISMATCH
    elif direct_status == "success":
        if not is_equal(direct_value, fwd_value):
            return ResultType.VALUE_MISMATCH
    
    if rev_status == "success" and fwd_status == "success" and \
        not is_gradient_equal(rev_grad, fwd_grad):
        return ResultType.GRAD_VALUE_MISMATCH
    
    nd_status = "fail"
    if direct_status == "success":
        nd_status, nd_theoretical, nd_numerical, nd_err = NDINV(fn, inputs, device=device)
    
        if nd_status == "fail" and not allow_error(nd_err):
            return ResultType.ND_FAIL

        if nd_status == "success" and \
            not is_gradient_equal(nd_theoretical, nd_numerical, atol=ATOL, rtol=RTOL):
            return ResultType.GRAD_VALUE_MISMATCH
    
    
    if "success" in [rev_status, fwd_status, nd_status]:
        grad_flag = False
        if rev_grad is not None:
            for x in rev_grad: 
                if x is not None: grad_flag = True
        if fwd_grad is not None:
            for x in fwd_grad: 
                if x is not None: grad_flag = True
        if grad_flag:
            return ResultType.SUCCESS
        else:
            return ResultType.GRAD_NOT_COMPUTED
    else:
        if direct_status == "fail":
            return ResultType.DIRECT_FAIL
        else:
            return ResultType.GRAD_NOT_COMPUTED

def run_and_check(arg_func_def_code, random_seed, tensor_args, use_grad=False,
    device="cpu") -> (ResultType, list):
    set_seed(random_seed)
    
    try:
        exec(arg_func_def_code)
        exec("inputs = []")
        for tensor_name in tensor_args:
            eval(f"inputs.append({tensor_name})")
    except Exception as e:
        return ResultType.ARGS_FAIL
    else:
        with tf.device(device):
            if use_grad:
                result = eval(f"testAD(Grad({TEST_FUNC_EXEC_NAME}), inputs, device=\"{device}\")")
            else:
                result = eval(f"testAD({TEST_FUNC_EXEC_NAME}, inputs, device=\"{device}\")")
    return result
    