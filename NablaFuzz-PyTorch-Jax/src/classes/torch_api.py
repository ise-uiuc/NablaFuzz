import torch
from classes.argdef import ArgDef
from classes.argument import *
from classes.api import *
from classes.database import TorchDatabase
import json
import random


class TorchArgument(Argument):
    _supported_types = [
        ArgType.TORCH_DTYPE,
        ArgType.TORCH_OBJECT,
        ArgType.TORCH_TENSOR,
    ]
    _dtypes = [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex64,
        torch.complex128,
        torch.bool,
    ]
    _memory_format = [
        torch.contiguous_format,
        torch.channels_last,
        torch.preserve_format,
    ]
    _float_complex_dtypes = [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
        torch.complex64,
        torch.complex128,
    ]
    _min_values = [0] + [-(1 << i) for i in range(0, 8)]
    _max_values = [(1 << i) - 1 for i in range(0, 8)]
    _tensor_size_limit = 1e7

    def __init__(
        self,
        value,
        type: ArgType,
        shape=None,
        dtype=None,
        max_value=1,
        min_value=0,
    ):
        super().__init__(value, type)
        self.shape = shape
        self.dtype = dtype
        self.max_value = max_value
        self.min_value = min_value
        self.grad = False
        self.new_tensor = True

    def to_code(
        self,
        var_name,
        device="cpu",
        low_precision=False,
        is_sparse=False,
        use_old_tensor=False,
    ) -> str:
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                self.value[i].grad = self.grad
                self.value[i].new_tensor = self.new_tensor
                code += self.value[i].to_code(
                    f"{var_name}_{i}",
                    device=device,
                    low_precision=low_precision,
                    use_old_tensor=use_old_tensor,
                )
                arg_name_list += f"{var_name}_{i},"

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
            return code
        elif self.type == ArgType.TORCH_TENSOR:
            dtype = self.dtype
            max_value = self.max_value
            min_value = self.min_value
            if low_precision:
                dtype = self.low_precision_dtype(dtype)
                max_value, min_value = self.random_tensor_value(dtype)

            # FIXME: shape tune
            shape = self.shape
            size = 1
            for i in range(len(shape) - 1, -1, -1):
                if size * shape[i] > TorchArgument._tensor_size_limit:
                    shape[i] = 1
                else:
                    size *= shape[i]

            suffix = ""
            if is_sparse:
                suffix += ".to_sparse()"
            if self.grad and self.is_float_or_complex_tensor():
                suffix += ".requires_grad_()"

            code = ""
            if self.new_tensor and not use_old_tensor:
                default_arg = f"{shape}, dtype={dtype}"

                if dtype.is_floating_point:
                    code += f"{var_name}_tensor = torch.empty({default_arg})\n"
                    code += f"{var_name}_tensor.uniform_({min_value}, {max_value})\n"
                elif dtype.is_complex:
                    code += f"{var_name}_tensor = torch.empty({default_arg})\n"
                    code += f"{var_name}_tensor.uniform_({min_value}, {max_value})\n"
                else:
                    code += f"{var_name}_tensor = torch.randint({min_value}, {max_value+1}, {default_arg})\n"
            code += f"{var_name} = {var_name}_tensor.clone().detach().to('{device}'){suffix}\n"
            return code
        elif self.type == ArgType.TORCH_OBJECT:
            return f"{var_name} = {self.value}\n"
        elif self.type == ArgType.TORCH_DTYPE:
            return f"{var_name} = {self.value}\n"
        return super().to_code(var_name)

    def to_diff_code(self, var_name, oracle):
        """differential testing with oracle"""
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            code = ""
            arg_name_list = ""
            for i in range(len(self.value)):
                code += self.value[i].to_diff_code(f"{var_name}_{i}", oracle)
                arg_name_list += f"{var_name}_{i},"
            if self.type == ArgType.LIST:
                code += f"{var_name} = [{arg_name_list}]\n"
            else:
                code += f"{var_name} = ({arg_name_list})\n"
        elif self.type == ArgType.TORCH_TENSOR:
            if oracle == OracleType.CUDA:
                code += f"{var_name} = {var_name}_tensor.clone().cuda()\n"
            elif oracle == OracleType.PRECISION:
                code += f"{var_name} = {var_name}_tensor.clone().type({self.dtype})\n"
        return code

    def mutate_value(self) -> None:
        if self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.TORCH_TENSOR:
            self.max_value, self.min_value = self.random_tensor_value(
                self.dtype
            )
        elif self.type in super()._support_types:
            super().mutate_value()
        else:
            print(self.type, self.value)
            assert 0

    def mutate_type(self) -> None:
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    TorchArgument(2, ArgType.INT),
                    TorchArgument(3, ArgType.INT),
                ]
            elif new_type == ArgType.TORCH_TENSOR:
                self.shape = [2, 2]
                self.dtype = torch.float32
            elif new_type == ArgType.TORCH_DTYPE:
                self.value = choice(self._dtypes)
            elif new_type == ArgType.TORCH_OBJECT:
                self.value = choice(self._memory_format)
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.TORCH_TENSOR:
            new_size = list(self.shape)
            # change the dimension of tensor
            if change_tensor_dimension():
                if add_tensor_dimension():
                    new_size.append(1)
                elif len(new_size) > 0:
                    new_size.pop()
            # change the shape
            for i in range(len(new_size)):
                if change_tensor_shape():
                    new_size[i] = self.mutate_int_value(new_size[i], _min=0)
            self.shape = new_size
            # change dtype
            if change_tensor_dtype():
                self.dtype = choice(self._dtypes)
                self.max_value, self.min_value = self.random_tensor_value(
                    self.dtype
                )
        elif self.type == ArgType.TORCH_OBJECT:
            pass
        elif self.type == ArgType.TORCH_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert 0

    def to_record(self):
        if self.type == ArgType.TORCH_TENSOR:
            shape = []
            for i in self.shape:
                shape.append(int(i))
            return {"shape": shape, "dtype": str(self.dtype)}
        elif self.type == ArgType.TORCH_DTYPE:
            return str(self.value)
        elif self.type == ArgType.TORCH_OBJECT:
            return str(self.value)
        elif self.type in [ArgType.LIST, ArgType.TUPLE]:
            temp = []
            for arg in self.value:
                temp.append(arg.to_record())
            return temp
        elif self.type == ArgType.INT:
            return int(self.value)
        else:
            return self.value

    def is_float_or_complex_tensor(self):
        return (
            self.type == ArgType.TORCH_TENSOR
            and self.dtype in self._float_complex_dtypes
        )

    @staticmethod
    def random_tensor_value(dtype):
        max_value = 1
        min_value = 0
        if dtype == torch.bool:
            min_value = choice([0, 1])
            max_value = max(min_value, choice([0, 1]))
        elif dtype == torch.uint8:
            min_value = 0
            max_value = choice(TorchArgument._max_values)
        else:
            min_value = choice(TorchArgument._min_values)
            max_value = choice(TorchArgument._max_values)
        return max_value, min_value

    @staticmethod
    def generate_arg_from_signature(signature):
        """Generate a Torch argument from the signature"""
        if signature == "torchTensor":
            return TorchArgument(
                None, ArgType.TORCH_TENSOR, shape=[2, 2], dtype=torch.float32
            )
        if signature == "torchdtype":
            return TorchArgument(
                choice(TorchArgument._dtypes), ArgType.TORCH_DTYPE
            )
        if isinstance(signature, str) and signature == "torchdevice":
            value = torch.device("cpu")
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torchmemory_format":
            value = choice(TorchArgument._memory_format)
            return TorchArgument(value, ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature == "torch.strided":
            return TorchArgument("torch.strided", ArgType.TORCH_OBJECT)
        if isinstance(signature, str) and signature.startswith("torch."):
            value = eval(signature)
            if isinstance(value, torch.dtype):
                return TorchArgument(value, ArgType.TORCH_DTYPE)
            elif isinstance(value, torch.memory_format):
                return TorchArgument(value, ArgType.TORCH_OBJECT)
            print(signature)
            # TODO:
            assert 0
        if isinstance(signature, bool):
            return TorchArgument(signature, ArgType.BOOL)
        if isinstance(signature, int):
            return TorchArgument(signature, ArgType.INT)
        if isinstance(signature, str):
            return TorchArgument(signature, ArgType.STR)
        if isinstance(signature, float):
            return TorchArgument(signature, ArgType.FLOAT)
        if isinstance(signature, tuple):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.TUPLE)
        if isinstance(signature, list):
            value = []
            for elem in signature:
                value.append(TorchArgument.generate_arg_from_signature(elem))
            return TorchArgument(value, ArgType.LIST)
        # signature is a dictionary
        if isinstance(signature, dict):
            if not (
                "shape" in signature.keys() and "dtype" in signature.keys()
            ):
                raise Exception("Wrong signature {0}".format(signature))
            shape = signature["shape"]
            dtype = signature["dtype"]
            # signature is a ndarray or tensor.
            if isinstance(shape, (list, tuple)):
                if not dtype.startswith("torch."):
                    dtype = f"torch.{dtype}"
                dtype = eval(dtype)
                max_value, min_value = TorchArgument.random_tensor_value(dtype)
                return TorchArgument(
                    None,
                    ArgType.TORCH_TENSOR,
                    shape,
                    dtype=dtype,
                    max_value=max_value,
                    min_value=min_value,
                )
            else:
                return TorchArgument(
                    None,
                    ArgType.TORCH_TENSOR,
                    shape=[2, 2],
                    dtype=torch.float32,
                )
        return TorchArgument(None, ArgType.NULL)

    @staticmethod
    def low_precision_dtype(dtype):
        if dtype in [torch.int16, torch.int32, torch.int64]:
            return torch.int8
        elif dtype in [torch.float32, torch.float64]:
            return torch.float16
        elif dtype in [torch.complex64, torch.complex128]:
            return torch.complex32
        return dtype

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res is not None:
            return res
        if isinstance(x, torch.Tensor):
            return ArgType.TORCH_TENSOR
        elif isinstance(x, torch.dtype):
            return ArgType.TORCH_DTYPE
        else:
            return ArgType.TORCH_OBJECT


class TorchAPI(API):
    # indices of sparse tensor for sparse API
    _sparse_API = {
        "torch.sspaddmm": [0, 1],
        "torch.Tensor.sspaddmm": [0, 1],
        "torch.sparse.sum": [0],
        "torch.sparse.addmm": [1],
        "torch.sparse.mm": [0],
        "torch.hspmm": [0],
        "torch.smm": [0],
        "torch.Tensor.smm": [0],
        "torch.sparse.softmax": [0],
        "torch.sparse.log_softmax": [0],
    }

    def __init__(self, api_name):
        super().__init__(api_name)
        arg_defs = TorchDatabase.get_argdef(api_name)
        self.arg_defs = list(map(lambda x: ArgDef.new(x), arg_defs))
        self.is_class = inspect.isclass(eval(self.api))

    def get_invocation(self, record=None):
        for arg_def in self.arg_defs:
            arg_def.case = None
        if record is None:
            record = TorchDatabase.get_rand_record(self.api)
        self.args = self.generate_args_from_record(record)
        no_name_args = []
        for key, arg in self.args.items():
            if key.startswith("parameter:"):
                assert int(key[10:]) == len(no_name_args)
                no_name_args.append(arg)

        idx = 0
        is_Tensor_api = self.api.startswith("torch.Tensor.")
        while idx < len(no_name_args):
            idx_def = idx + is_Tensor_api
            current_name = self.arg_defs[idx_def].name
            if current_name[0] == "*":
                current_name = current_name[1:]
                temp_list = []
                for _ in range(idx, len(no_name_args)):
                    temp_list.append(no_name_args[idx])
                new_case = TorchArgument(temp_list, ArgType.LIST)
                new_case.set_name(current_name)
                self.arg_defs[idx_def].case = new_case
                break
            else:
                no_name_args[idx].set_name(current_name)
                self.arg_defs[idx_def].case = no_name_args[idx]
                idx += 1
        for key, arg in self.args.items():
            if key.startswith("parameter") or key == "input_signature":
                continue
            arg_def = self.find_arg_with_name(key)
            if arg_def is None:
                print("NO SUCH ARG:", self.api, key)
                continue
            if arg_def.ignore is True:
                continue
            arg.set_name(key)
            arg_def.case = arg

    def to_record(self):
        record = {}
        for key, arg in self.args.items():
            assert key != "_id"
            record[key] = arg.to_record()
        return record

    def find_arg_with_name(self, arg_name):
        for arg_def in self.arg_defs:
            if arg_def.name == arg_name:
                return arg_def
        return None

    def mutate(self, enable_value=True, enable_type=True, enable_db=True):
        num_arg = len(self.args)
        if num_arg == 0:
            return
        num_Mutation = randint(1, num_arg + 1)
        for _ in range(num_Mutation):
            arg_name = choice(list(self.args.keys()))
            arg = self.args[arg_name]

            if enable_type and do_type_mutation():
                arg.mutate_type()
            do_value_mutation = True
            if enable_db and do_select_from_db():
                new_arg, success = TorchDatabase.select_rand_over_db(
                    self.api, arg_name
                )
                if success:
                    new_arg = TorchArgument.generate_arg_from_signature(
                        new_arg
                    )
                    self.args[arg_name] = new_arg
                    do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def get_arglist(self):
        arglist = []
        for arg_def in self.arg_defs:
            if arg_def.case is not None:
                # for the *shape, arg_def.name == *shape, arg_case.name == shape
                if arg_def.is_optional:
                    arglist.append(f"{arg_def.name}={arg_def.name}")
                else:
                    arglist.append(f"{arg_def.name}")
        return arglist

    def to_code(
        self,
        prefix="arg",
        res="res",
        device="cpu",
        use_try=False,
        error_res=None,
        low_precision=False,
        use_old_tensor=False,
        use_backward=False,
        grad_res="grad_res",
        grad_err_res="grad_err_res",
        tensor_arg_names=[],
    ) -> str:
        code = ""
        arg_str = ""

        for idx, arg_def in enumerate(self.arg_defs):
            if arg_def.name == "device":
                arg_str += f"device='{device}', "
            elif (
                arg_def.case is not None
                and arg_def.name != "input_signature"
                and not arg_def.ignore
            ):
                arg_case = arg_def.case
                code += arg_case.to_code(
                    arg_case.name,
                    is_sparse=self.is_sparse_tensor(self.api, idx),
                    device=device,
                    use_old_tensor=use_old_tensor,
                )
                # for the *shape, arg_def.name == *shape, arg_case.name == shape
                if arg_def.name in ["_input_tensor"]:
                    pass
                elif arg_def.is_optional:
                    arg_str += f"{arg_def.name}={arg_def.name}, "
                else:
                    arg_str += f"{arg_def.name}, "

        res_code = ""
        if self.is_class:
            if not use_old_tensor:
                res_code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                arg_name = "input_signature"
                code += self.args["input_signature"].to_code(
                    arg_name,
                    device=device,
                    low_precision=low_precision,
                    use_old_tensor=use_old_tensor,
                )
                res_code += f"{res} = {prefix}_class(*{arg_name})\n"
        elif self.api.startswith("torch.Tensor."):
            tensor_op = self.api.replace("torch.Tensor.", "")
            res_code = f"{res} = _input_tensor.{tensor_op}({arg_str})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(
            res,
            error_res,
            res_code,
            use_try,
            low_precision,
            use_backward=use_backward,
            grad_res=grad_res,
            grad_err_res=grad_err_res,
            tensor_arg_names=tensor_arg_names,
        )

    def to_forward_pass_code(
        self,
        prefix="arg",
        res1="res1",
        res2="res2",
        err1="err1",
        err2="err2",
        device="cpu",
    ):
        fn_res_name = "res"

        (
            tensor_arg_names,
            tensor_code,
            wrapper_code,
            res_code,
            _,
            tensor_info_code,
        ) = self._to_ad_code_helper(
            TorchArgument._float_complex_dtypes,
            prefix=prefix,
            device=device,
            fn_res_name=fn_res_name,
        )
        code += tensor_info_code

        if len(tensor_arg_names) == 0:
            # There isn't any differentiable tensor
            return ""

        tensor_arg_str = ", ".join(tensor_arg_names)
        generate_code, detach_code = self.split_tensor_code(tensor_code)

        normal_code = detach_code.replace(".requires_grad_()", "")
        normal_code += f"{res1} = fn({tensor_arg_str})\n"

        grad_code = detach_code
        grad_code += f"{res2} = fn({tensor_arg_str})\n"

        code = self._get_wrapper_fn_code(
            tensor_arg_str, res_code, fn_res_name, wrapper_code
        )
        code += generate_code
        code += self.try_except_code(normal_code, err1)
        code += self.try_except_code(grad_code, err2)

        return code

    def to_gradcheck_code(
        self,
        prefix="arg",
        device="cpu",
        error_res_fn="err_fn",
        error_res_check="err_check",
        use_old_tensor=False,
        atol=1e-2,
        rtol=1e-2,
        check_forward_ad=False,
        check_backward_ad=True,
        check_batched_grad=False,
    ) -> str:
        # the returned code
        code = ""
        # the code in the wrapper function
        wrapper_code = ""
        # the tensor argument code
        tensor_code = ""
        # the return name in the wrapper function
        fn_res_name = "fn_res"
        # the name list of all tensor argument
        tensor_arg_names = []
        # the list of dtypes that are supported by gradcheck
        allowed_dtypes = [torch.float64, torch.complex128]

        gradcheck_kwargs = {
            "check_sparse_nnz": False,
            "atol": atol,
            "rtol": rtol,
            "check_forward_ad": check_forward_ad,
            "check_backward_ad": check_backward_ad,
            "check_batched_grad": check_batched_grad,
        }

        (
            tensor_arg_names,
            tensor_code,
            wrapper_code,
            res_code,
            has_sparse_tensor,
            tensor_info_code,
        ) = self._to_ad_code_helper(
            allowed_dtypes,
            prefix=prefix,
            use_old_tensor=use_old_tensor,
            device=device,
            fn_res_name=fn_res_name,
        )
        code += tensor_info_code
        gradcheck_kwargs["check_sparse_nnz"] = has_sparse_tensor

        if len(tensor_arg_names) == 0:
            # There are no input tensors
            return ""

        tensor_arg_str = ", ".join(tensor_arg_names)

        generate_code, detach_code = self.split_tensor_code(tensor_code)
        gradcheck_arg_list = []
        for key, value in gradcheck_kwargs.items():
            gradcheck_arg_list.append(f"{key}={value}")
        check_code = (
            detach_code
            + f"torch.autograd.gradcheck(fn, ({tensor_arg_str}), {', '.join(gradcheck_arg_list)})"
        )
        # error_filter_code = "from helper import is_differentiable\n"
        # error_filter_code += f"if not is_differentiable(fn, ({tensor_arg_str})): {error_res_fn} = 'Non-diff'\n"

        code += self._get_wrapper_fn_code(
            tensor_arg_str, res_code, fn_res_name, wrapper_code
        )
        code += generate_code
        test_code = detach_code
        test_code += "from helper import is_backward_res\n"
        test_code += f"assert is_backward_res(fn({tensor_arg_str})), 'Cannot Backward'\n"
        else_code = self.try_except_code(check_code, error_res_check)
        code += self.try_except_code(
            test_code, error_res_fn, else_code=else_code
        )
        return code

    def to_gradgradcheck_code(
        self,
        prefix="arg",
        device="cpu",
        error_res_fn="err_fn",
        error_res_check="err_check",
        use_old_tensor=False,
        atol=1e-2,
        rtol=1e-2,
        check_fwd_over_rev=False,
    ) -> str:
        # the returned code
        code = ""
        # the code in the wrapper function
        wrapper_code = ""
        # the tensor argument code
        tensor_code = ""
        # the return name in the wrapper function
        fn_res_name = "fn_res"
        # the name list of all tensor argument
        tensor_arg_names = []
        # the list of dtypes that are supported by gradcheck
        allowed_dtypes = [torch.float64, torch.complex128]

        gradcheck_kwargs = {
            # "check_sparse_nnz": False,
            "atol": atol,
            "rtol": rtol,
            "check_fwd_over_rev": check_fwd_over_rev,
            "check_rev_over_rev": not check_fwd_over_rev,
            "check_undefined_grad": not check_fwd_over_rev,
        }

        (
            tensor_arg_names,
            tensor_code,
            wrapper_code,
            res_code,
            has_sparse_tensor,
            tensor_info_code,
        ) = self._to_ad_code_helper(
            allowed_dtypes,
            prefix=prefix,
            use_old_tensor=use_old_tensor,
            device=device,
            fn_res_name=fn_res_name,
        )
        code += tensor_info_code
        # gradcheck_kwargs["check_sparse_nnz"] = has_sparse_tensor

        if len(tensor_arg_names) == 0:
            # There are no input tensors
            return ""

        tensor_arg_str = ", ".join(tensor_arg_names)

        generate_code, detach_code = self.split_tensor_code(tensor_code)
        gradcheck_arg_list = []
        for key, value in gradcheck_kwargs.items():
            gradcheck_arg_list.append(f"{key}={value}")
        check_code = (
            detach_code
            + f"torch.autograd.gradgradcheck(fn, ({tensor_arg_str}), {', '.join(gradcheck_arg_list)})"
        )

        code += self._get_wrapper_fn_code(
            tensor_arg_str, res_code, fn_res_name, wrapper_code
        )
        code += generate_code
        test_code = detach_code
        test_code += "from helper import is_backward_res\n"
        test_code += f"assert is_backward_res(fn({tensor_arg_str})), 'Cannot Backward'\n"
        else_code = self.try_except_code(check_code, error_res_check)
        code += self.try_except_code(
            test_code, error_res_fn, else_code=else_code
        )
        return code

    def to_fb_compare_code(
        self,
        prefix="arg",
        device="cpu",
        res1="res1",
        res2="res2",
        err1="err1",
        err2="err2",
        use_old_tensor=False,
    ) -> str:
        """
        Compare the Jacobians of forward- and reverse-mode
        """
        # the returned code
        code = ""
        # the code in the wrapper function
        wrapper_code = ""
        # the tensor argument code
        tensor_code = ""
        # the return name in the wrapper function
        fn_res_name = "fn_res"
        # the name list of all tensor argument
        tensor_arg_names = []
        # the list of dtypes that are supported by AD
        allowed_dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]

        (
            tensor_arg_names,
            tensor_code,
            wrapper_code,
            res_code,
            _,
            tensor_info_code,
        ) = self._to_ad_code_helper(
            allowed_dtypes,
            prefix=prefix,
            use_old_tensor=use_old_tensor,
            device=device,
            fn_res_name=fn_res_name,
        )
        code += tensor_info_code

        if len(tensor_arg_names) == 0:
            # There are no input tensors
            return ""

        tensor_arg_str = ", ".join(tensor_arg_names)
        generate_code, detach_code = self.split_tensor_code(tensor_code)

        bwd_code = detach_code
        bwd_code += f"{res1} = get_jacobians(fn, ({tensor_arg_str},), True)"

        fwd_code = detach_code
        fwd_code += f"{res2} = get_jacobians(fn, ({tensor_arg_str},), False)"

        code += self._get_wrapper_fn_code(
            tensor_arg_str, res_code, fn_res_name, wrapper_code
        )
        code += generate_code
        code += self.try_except_code(bwd_code, err1)
        code += self.try_except_code(fwd_code, err2)

        code = "from gradcheck import get_jacobians\n" + code
        return code

    def to_differential_fn_code(self):
        (
            tensor_arg_names,
            tensor_code,
            wrapper_code,
            res_code,
            _,
            tensor_info_code,
        ) = self._to_ad_code_helper(
            TorchArgument._float_complex_dtypes,
        )
        tensor_arg_str = ", ".join(tensor_arg_names)
        fn_code = tensor_info_code
        fn_code += f"torch.random.manual_seed({random.randint(0, 1 << 16)})\n"
        fn_code += self._get_wrapper_fn_code(
            tensor_arg_str, res_code, "fn_res", wrapper_code
        )
        inv_code, _ = self.split_tensor_code(tensor_code)
        input_list = [f"{arg}_tensor" for arg in tensor_arg_names]
        return fn_code, inv_code, input_list

    def _to_ad_code_helper(
        self,
        allowed_dtypes: list,
        prefix="arg",
        use_old_tensor=False,
        device="cpu",
        fn_res_name="fn_res",
    ):
        """
        return: (tensor_arg_names, tensor_code, wrapper_code, res_code, has_sparse_tensor)
            tensor_arg_names: the name list of the differentiable tensors (with dtype in allowed_dtypes)
            tensor_code: the code generate the above tensors
            wrapper code: the code in the wrapper function
            res_code: the wrapper funtion "return" code
            has_sparse: if any the differentiable tensor is sparse
        """
        tensor_arg_names = []
        tensor_code = ""
        wrapper_code = ""
        tensor_info_code = "# Tensor Info\n"
        # for gradcheck
        has_sparse_tensor = False

        arg_str = ""

        for idx, arg_def in enumerate(self.arg_defs):
            if arg_def.name == "device":
                arg_str += f"device='{device}', "
            elif (
                arg_def.case is not None
                and arg_def.name != "input_signature"
                and not arg_def.ignore
            ):
                arg_case = arg_def.case

                if arg_case.is_float_or_complex_tensor() and not self.is_class:
                    # set the grad flag
                    arg_case.grad = True
                    # set the dtype
                    if arg_case.dtype not in allowed_dtypes:
                        arg_case.dtype = choice(allowed_dtypes)

                is_sparse = self.is_sparse_tensor(self.api, idx)
                has_sparse_tensor = has_sparse_tensor or is_sparse
                temp_code = arg_case.to_code(
                    arg_case.name,
                    is_sparse=is_sparse,
                    device=device,
                    use_old_tensor=use_old_tensor,
                )
                if self.is_class:
                    # If it is a class API, skip the setting arguments
                    # NOTE: The API is not differentiable with respect to this kind of argument
                    wrapper_code += temp_code
                else:
                    if arg_case.is_float_or_complex_tensor():
                        tensor_code += temp_code
                        tensor_arg_names.append(arg_def.name)
                        info = json.dumps(
                            {
                                "name": arg_def.name,
                                "shape": [int(i) for i in arg_case.shape],
                                "dtype": str(arg_case.dtype),
                            }
                        )
                        tensor_info_code += f"# {info}\n"
                    else:
                        wrapper_code += temp_code

                if arg_def.name in ["_input_tensor"]:
                    pass
                elif arg_def.is_optional:
                    arg_str += f"{arg_def.name}={arg_def.name}, "
                else:
                    arg_str += f"{arg_def.name}, "

        res_code = ""
        if self.is_class:
            if not use_old_tensor:
                wrapper_code += f"{prefix}_class = {self.api}({arg_str})\n"

            if "input_signature" in self.args.keys():
                input_args = self.args["input_signature"]
                assert input_args.type in [ArgType.TUPLE, ArgType.LIST]

                arg_name_list = []
                for idx, arg_case in enumerate(input_args.value):
                    arg_name = f"input_{idx}"

                    if arg_case.is_float_or_complex_tensor():
                        # set the grad flag
                        arg_case.grad = True
                        # set the dtype
                        if arg_case.dtype not in allowed_dtypes:
                            arg_case.dtype = choice(allowed_dtypes)

                    is_sparse = self.is_sparse_tensor(self.api, idx)
                    has_sparse_tensor = has_sparse_tensor or is_sparse
                    temp_code = arg_case.to_code(
                        arg_name,
                        is_sparse=is_sparse,
                        device=device,
                        use_old_tensor=use_old_tensor,
                    )
                    if arg_case.is_float_or_complex_tensor():
                        tensor_code += temp_code
                        tensor_arg_names.append(arg_name)
                        info = json.dumps(
                            {
                                "name": arg_name,
                                "shape": [int(i) for i in arg_case.shape],
                                "dtype": str(arg_case.dtype),
                            }
                        )
                        tensor_info_code += f"# {info}\n"
                    else:
                        wrapper_code += temp_code
                    arg_name_list.append(arg_name)

                res_code += f"{fn_res_name} = {prefix}_class({', '.join(arg_name_list)})\n"
        elif self.api.startswith("torch.Tensor."):
            tensor_op = self.api.replace("torch.Tensor.", "")
            res_code = (
                f"{fn_res_name} = _input_tensor.{tensor_op}({arg_str})\n"
            )
        else:
            res_code = f"{fn_res_name} = {self.api}({arg_str})\n"

        return (
            tensor_arg_names,
            tensor_code,
            wrapper_code,
            res_code,
            has_sparse_tensor,
            tensor_info_code,
        )

    @staticmethod
    def _get_wrapper_fn_code(
        tensor_arg_str, res_code, fn_res_name, wrapper_code
    ):
        wrapper_function = f"def fn({tensor_arg_str}):\n"
        wrapper_function += API.indent_code(res_code)
        wrapper_function += API.indent_code(f"return {fn_res_name}\n")

        code = "def get_fn():\n"
        code += API.indent_code(wrapper_code)
        code += API.indent_code(wrapper_function)
        code += API.indent_code("return fn\n")
        code += "fn = get_fn()\n"
        return code

    def to_diff_code(
        self,
        oracle: OracleType,
        prefix="arg",
        res="res",
        *,
        error_res=None,
        use_try=False,
    ) -> str:
        """Generate code for the oracle"""
        code = ""
        arg_str = ""
        count = 1

        for key, arg in self.args.items():
            if key == "input_signature":
                continue
            arg_name = f"{prefix}_{count}"
            code += arg.to_diff_code(arg_name, oracle)
            if key.startswith("parameter:"):
                arg_str += f"{arg_name},"
            else:
                arg_str += f"{key}={arg_name},"
            count += 1

        res_code = ""
        if self.is_class:
            if oracle == OracleType.CUDA:
                code = f"{prefix}_class = {prefix}_class.cuda()\n"
            if "input_signature" in self.args.keys():
                arg_name = f"{prefix}_{count}"
                code += self.args["input_signature"].to_diff_code(
                    arg_name, oracle
                )
                res_code = f"{res} = {prefix}_class(*{arg_name})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"

        return code + self.invocation_code(
            res, error_res, res_code, use_try, oracle == OracleType.PRECISION
        )

    @staticmethod
    def split_tensor_code(code):
        """Split the tensor code to get the tensor generation code and tensor detach code"""
        if code.strip() == "":
            return "", ""

        code_list = code.split("\n")
        l1, l2 = [], []
        for c in code_list:
            if c.strip() == "":
                continue
            if (
                "_tensor = torch.randint(" in c
                or "_tensor = torch.empty(" in c
                or "_tensor.uniform_(" in c
            ):
                l1.append(c)
            else:
                l2.append(c)
        generate_code = "\n".join(l1) + "\n"
        detach_code = "\n".join(l2) + "\n"
        return generate_code, detach_code

    @staticmethod
    def backward_code(
        res,
        grad_res="grad_res",
        grad_err_res="grad_err_res",
        tensor_arg_names=[],
    ):
        bk_code = f"{res}.sum().backward()\n"
        error_code = f"{grad_err_res} = str(e)\n"
        if len(tensor_arg_names) > 0:
            # get the grad of all input tensors
            grad_code = ""
            for name in tensor_arg_names:
                grad_code += f"{name}.grad, "
            grad_code = f"{grad_res} = [{grad_code}]"
        else:
            grad_code = f"{grad_res} = None"

        code = "try:\n"
        code += API.indent_code(bk_code)
        code += "except Exception as e:\n"
        code += API.indent_code(error_code)
        code += "else:\n"
        code += API.indent_code(grad_code)
        return code

    @staticmethod
    def invocation_code(
        res,
        error_res,
        res_code,
        use_try,
        low_precision,
        use_backward=False,
        grad_res="grad_res",
        grad_err_res="grad_err_res",
        tensor_arg_names=[],
    ):
        code = ""
        if use_try:
            # specified with run_and_check function in relation_tools.py
            if error_res is None:
                error_res = res
            temp_code = "try:\n"
            temp_code += API.indent_code(res_code)
            temp_code += f"except Exception as e:\n"
            temp_code += API.indent_code(f'{error_res} = "ERROR:"+str(e)\n')
            if use_backward:
                temp_code += "else:\n"
                temp_code += API.indent_code(
                    TorchAPI.backward_code(
                        res,
                        grad_res=grad_res,
                        grad_err_res=grad_err_res,
                        tensor_arg_names=tensor_arg_names,
                    )
                )
            res_code = temp_code

        if low_precision:
            code += "start = time.time()\n"
            code += res_code
            code += f"{res} = time.time() - start\n"
        else:
            code += res_code
        return code

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key in record.keys():
            if key != "output_signature":
                args[key] = TorchArgument.generate_arg_from_signature(
                    record[key]
                )
        return args

    @staticmethod
    def is_sparse_tensor(api_name, index):
        return (
            api_name in TorchAPI._sparse_API.keys()
            and index in TorchAPI._sparse_API[api_name]
        )
