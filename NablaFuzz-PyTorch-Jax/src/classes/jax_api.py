from classes.argument import *
from classes.api import *
from classes.database import JaxDatabase
import json
import jax
import jax.numpy as jnp
from random import randint, choice
from utils.loader import load_data

from utils.printer import dump_data


class JaxArgument(Argument):
    _dtypes = [
        jnp.bfloat16,
        jnp.float16,
        jnp.float32,
        jnp.float64,
        jnp.complex64,
        jnp.complex128,
        jnp.int8,
        jnp.int16,
        jnp.int32,
        jnp.int64,
        jnp.uint8,
        jnp.uint16,
        jnp.uint32,
        jnp.uint64,
        jnp.bool_,
    ]
    _inexact_float_dtypes = [
        jnp.bfloat16,
        jnp.float16,
        jnp.float32,
        jnp.float64,
    ]
    _inexact_precise_float_dtypes = [
        jnp.float32,
        jnp.float64,
    ]
    _inexact_dtypes = [
        jnp.bfloat16,
        jnp.float16,
        jnp.float32,
        jnp.float64,
        jnp.complex64,
        jnp.complex128,
    ]
    _precise_dtype_dict = {
        jnp.bfloat16: jnp.float64,
        jnp.float16: jnp.float64,
        jnp.float32: jnp.float64,
        jnp.float64: jnp.float64,
        jnp.complex64: jnp.complex128,
        jnp.complex128: jnp.complex128,
    }
    _support_types = [ArgType.JAX_ARRAY, ArgType.JAX_DTYPE]
    _min_values = [0] + [-(1 << i) for i in range(0, 8)]
    _max_values = [(1 << i) - 1 for i in range(0, 8)]
    _shape_limit = 1e3
    _seed_range = 1e8

    def __init__(
        self, value, type: ArgType, minv=0, maxv=0, shape=None, dtype=None
    ):
        super().__init__(value, type)

        self.minv = minv
        self.maxv = maxv
        self.shape = shape
        self.dtype = dtype
        self.var_name = ""

        self.changable = True

    def set_var_name(self, var_name: str):
        self.var_name = var_name

    def get_var_name(self):
        return self.var_name

    def to_code(self, var_name: str, use_old_tensor=False) -> str:
        """Generate the code of argument"""
        self.set_var_name(var_name)
        code = ""
        if self.type in [ArgType.LIST, ArgType.TUPLE]:
            arg_name_list = []
            for idx, sub_value in enumerate(self.value):
                sub_var_name = f"{var_name}_{idx}"
                code += sub_value.to_code(
                    sub_var_name, use_old_tensor=use_old_tensor
                )
                arg_name_list.append(sub_var_name)

            if self.type == ArgType.LIST:
                code += f"{var_name} = [{', '.join(arg_name_list)}]\n"
            else:
                code += f"{var_name} = ({', '.join(arg_name_list)})\n"
        elif self.type == ArgType.JAX_ARRAY:
            shape = self.shape
            dtype = f"jax.numpy.{self.dtype.__name__}"

            shape = JaxArgument.shape_tune(shape)

            if not use_old_tensor:
                seed = randint(0, 1e8)
                code += f"mykey = jax.random.PRNGKey({seed})\n"
                if "float" in dtype:
                    temp_arg = f"mykey, {shape}, {dtype}, minval={self.minv}, maxval={self.maxv}"
                    code += (
                        f"{var_name}_array = jax.random.uniform({temp_arg})\n"
                    )
                elif "complex" in dtype:
                    if "complex64" in dtype:
                        float_dtype = "jax.numpy.float32"
                    else:
                        float_dtype = "jax.numpy.float64"

                    seed_ = randint(0, JaxArgument._seed_range)
                    code += f"mykey_ = jax.random.PRNGKey({seed_})\n"

                    temp_arg_1 = f"mykey, {shape}, {float_dtype}, minval={self.minv}, maxval={self.maxv}"
                    temp_arg_2 = f"mykey_, {shape}, {float_dtype}, minval={self.minv}, maxval={self.maxv}"

                    code += f"{var_name}_array = jax.lax.complex(jax.random.uniform({temp_arg_1}), jax.random.uniform({temp_arg_2}))\n"
                elif "bool" in dtype:
                    temp_arg = f"mykey, {shape}, {self.minv}, {self.maxv+1}"
                    code += f"{var_name}_array = jax.numpy.bool_(jax.random.randint({temp_arg}))\n"
                else:
                    temp_arg = (
                        f"mykey, {shape}, {self.minv}, {self.maxv+1}, {dtype}"
                    )
                    code += (
                        f"{var_name}_array = jax.random.randint({temp_arg})\n"
                    )
            code += f"{var_name} = {var_name}_array.clone()\n"
        elif self.type == ArgType.JAX_DTYPE:
            return f"{var_name} = jax.numpy.{self.value.__name__}\n"
        elif self.type == ArgType.JAX_SCALAR:
            dtype = f"jax.numpy.{self.dtype.__name__}"
            if "complex" in str(self.dtype):
                value = f"complex('{self.value}')"
            elif str(self.value) in ["inf", "-inf", "nan"]:
                value = f"float('{self.value}')"
            else:
                value = self.value
            return f"{var_name} = jax.numpy.array({value}, dtype={dtype})\n"
        else:
            code = super().to_code(var_name)
        return code

    def mutate_value(self) -> None:
        self.mutate_value_random()

    def mutate_value_random(self) -> None:
        if not self.changable:
            return
        """ Apply random value mutation. """
        if self.type == ArgType.INT:
            self.value = self.mutate_int_value(self.value)
        elif self.type == ArgType.STR:
            self.value = self.mutate_str_value(self.value)
        elif self.type == ArgType.FLOAT:
            self.value = self.mutate_float_value(self.value)
        elif self.type == ArgType.BOOL:
            self.value = choice([True, False])
        elif self.type == ArgType.TUPLE or self.type == ArgType.LIST:
            for arg in self.value:
                arg.mutate_value_random()
        elif self.type == ArgType.JAX_ARRAY:
            self.minv, self.maxv = self.random_value_range(self.dtype)
        elif self.type == ArgType.JAX_DTYPE:
            self.value = JaxArgument.mutate_dtype()
        elif self.type == ArgType.JAX_SCALAR:
            value = self.value
            dtype = f"jax.numpy.{self.dtype.__name__}"
            if "complex" in dtype:
                real = self.mutate_float_value(value.real)
                imag = self.mutate_float_value(value.imag)
                self.value = complex(real, imag)
            elif "float" in dtype:
                self.value = self.mutate_float_value(value)
            elif "int" in dtype:
                self.value = self.mutate_int_value(value)
            elif "bool" in dtype:
                self.value = choice([False, True])
            else:
                raise TypeError(f"No such dtype in scalar: {dtype}")
        elif self.type == ArgType.NULL:
            pass
        else:
            raise ValueError(self.type)

    def mutate_type(self) -> None:
        if not self.changable:
            return
        if self.type == ArgType.NULL:
            # choose from all types
            new_type = choice(self._support_types + super()._support_types)
            self.type = new_type
            if new_type == ArgType.LIST or new_type == ArgType.TUPLE:
                self.value = [
                    JaxArgument(2, ArgType.INT),
                    JaxArgument(3, ArgType.INT),
                ]
            elif new_type == ArgType.JAX_ARRAY:
                self.shape = [2, 2]
                self.dtype = jnp.float32
            elif new_type == ArgType.JAX_DTYPE:
                self.value = choice(self._dtypes)
            else:
                self.value = super().initial_value(new_type)
        elif self.type == ArgType.JAX_ARRAY:
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
                self.minv, self.maxv = self.random_value_range(self.dtype)
        elif self.type == ArgType.JAX_DTYPE:
            self.value = choice(self._dtypes)
        elif self.type == ArgType.JAX_SCALAR:
            self.dtype = choice(self._dtypes)

            dtype = f"jax.numpy.{self.dtype.__name__}"
            if "complex" in dtype:
                self.value = complex(1.0, 1.0)
            elif "float" in dtype:
                self.value = self.mutate_float_value(1.0)
            elif "int" in dtype:
                self.value = self.mutate_int_value(1)
            elif "bool" in dtype:
                self.value = choice([False, True])
            else:
                raise TypeError(f"No such dtype in scalar: {dtype}")
            self.mutate_value_random()
        elif self.type in super()._support_types:
            super().mutate_type()
        else:
            print(self.type, self.value)
            assert 0

    def is_differentiable(
        self, dtypes=[], convert_to_precise=True, limit_max_value=None
    ) -> bool:
        """
        Check whether this argument is differentiable
        NOTE: It may change the value or dtype of this argument

        TODO: limit its range of value (this is for numerical differentiation)
        """

        def limit(value):
            sign = self.value >= 0
            max_abs = max(abs(self.value), abs(limit_max_value))
            # NOTE: change the value of this argument!
            return max_abs if sign else -max_abs

        allowed_float = False
        for dt in dtypes:
            if "float" in str(dt):
                allowed_float = True
                break

        is_diff = False
        if self.type in [ArgType.JAX_ARRAY, ArgType.JAX_SCALAR]:
            if self.dtype in dtypes:
                is_diff = True
            elif (
                convert_to_precise
                and JaxArgument.get_more_precise_dtype(self.dtype) in dtypes
            ):
                # NOTE: It will change the dtype of this argument!
                self.dtype = JaxArgument.get_more_precise_dtype(self.dtype)
                is_diff = True
            else:
                is_diff = False
        elif allowed_float and self.type == ArgType.FLOAT:
            # TODO: now disable
            is_diff = False
        else:
            is_diff = False

        # FIXME:
        if is_diff and limit_max_value is not None:
            if self.type == ArgType.JAX_SCALAR:
                if self.dtype in [jnp.complex128, jnp.complex64]:
                    value = self.value
                    self.value = complex(limit(value.real), limit(value.imag))
                else:
                    self.value = limit(self.value)
        return is_diff

    @staticmethod
    def get_more_precise_dtype(dtype):
        precise_dict = JaxArgument._precise_dtype_dict
        if dtype in precise_dict.keys():
            return precise_dict[dtype]
        else:
            return dtype

    @staticmethod
    def str_to_dtype(dt: str):
        for dtype in JaxArgument._dtypes:
            if dt in dtype.__name__:
                return dtype
        return jax.numpy.float32
        # raise Exception(f"No such dtype: {dt}")

    @staticmethod
    def random_value_range(dtype):
        assert (
            dtype in JaxArgument._dtypes
        ), f"random_value_range doesn't support such dtype: {dtype}"
        minv = 0
        maxv = 1
        dtype_name = dtype.__name__

        if "bool" in dtype_name:
            minv = choice([0, 1])
            maxv = max(minv, choice([0, 1]))
        else:
            minv = (
                0 if "uint" in dtype_name else choice(JaxArgument._min_values)
            )
            maxv = choice(JaxArgument._max_values)
        return minv, maxv

    @staticmethod
    def mutate_dtype():
        return choice(JaxArgument._dtypes)

    @staticmethod
    def generate_arg_from_signature(signature):
        def generate_arg_basic(value):
            if isinstance(value, int):
                return JaxArgument(value, ArgType.INT)
            if isinstance(value, str):
                return JaxArgument(value, ArgType.STR)
            if isinstance(value, float):
                return JaxArgument(value, ArgType.FLOAT)
            if isinstance(value, tuple):
                tuple_value = []
                for elem in value:
                    tuple_value.append(
                        JaxArgument.generate_arg_from_signature(elem)
                    )
                return JaxArgument(tuple_value, ArgType.TUPLE)
            if isinstance(value, list):
                list_value = []
                for elem in value:
                    list_value.append(
                        JaxArgument.generate_arg_from_signature(elem)
                    )
                return JaxArgument(list_value, ArgType.LIST)
            return None

        temp_arg = generate_arg_basic(signature)
        if temp_arg is not None:
            return temp_arg

        if (not isinstance(signature, dict)) or ("Label" not in signature):
            return JaxArgument(None, ArgType.NULL)

        label = signature["Label"]

        if label == "raw":
            try:
                value = json.loads(signature["value"])
            except Exception:
                value = signature["value"]
            temp_arg = generate_arg_basic(value)
            if temp_arg is not None:
                return temp_arg

        if label == "tuple":
            value = json.loads(signature["value"])
            tuple_value = []
            for elem in value:
                tuple_value.append(
                    JaxArgument.generate_arg_from_signature(elem)
                )
            return JaxArgument(tuple_value, ArgType.TUPLE)
        if label == "list":
            try:
                value = json.loads(signature["value"])
            except Exception:
                value = signature["value"]
            list_value = []
            for elem in value:
                list_value.append(
                    JaxArgument.generate_arg_from_signature(elem)
                )
            return JaxArgument(list_value, ArgType.LIST)

        if label in ["nparray", "jaxarray", "tracer"]:
            if not (
                "shape" in signature.keys() and "dtype" in signature.keys()
            ):
                raise Exception("Wrong signature {0}".format(signature))
            shape = signature["shape"]
            dtype = signature["dtype"]
            dtype = JaxArgument.str_to_dtype(dtype)

            if isinstance(shape, (list, tuple)):
                minv, maxv = JaxArgument.random_value_range(dtype)
                return JaxArgument(
                    None, ArgType.JAX_ARRAY, minv, maxv, shape, dtype
                )
            else:
                minv, maxv = 0, 1
                shape = [
                    1,
                ]
                return JaxArgument(
                    None, ArgType.JAX_ARRAY, minv, maxv, shape, dtype
                )

        if label in ["scalar", "npraw"]:
            dtype = JaxArgument.str_to_dtype(signature["dtype"])
            if "complex" in signature["dtype"]:
                value = complex(signature["value"])
            else:
                value = JaxArgument.load_from_str(
                    signature["value"], signature["dtype"]
                )
            return JaxArgument(value, ArgType.JAX_SCALAR, dtype=dtype)

        # TODO:
        if label == "other":
            dump_data(json.dumps(signature), "unhandle.log", "a")
            pass
        return JaxArgument(None, ArgType.NULL)

    @staticmethod
    def load_from_str(val, dtype):
        try:
            return json.loads(val)
        except Exception:
            if dtype == "bool":
                if val == "False":
                    return False
                elif val == "True":
                    return True
                return False
            return 0.0

    @staticmethod
    def shape_tune(shape) -> list:
        size = 1
        for i in range(len(shape) - 1, -1, -1):
            if size * shape[i] > JaxArgument._shape_limit:
                shape[i] = 1
            else:
                size *= shape[i]
        return shape

    @staticmethod
    def get_type(x):
        res = Argument.get_type(x)
        if res is not None:
            return res
        if hasattr(res, "shape") or "Array" in str(type(x)):
            return ArgType.JAX_ARRAY
        if res in JaxArgument._dtypes:
            return ArgType.JAX_DTYPE
        else:
            return ArgType.NULL
            # assert 0, f"Cannot get the type: {x}, whose type is: {type(x)}"


class JaxAPI(API):
    def __init__(self, api_name, record=None):
        super().__init__(api_name)
        self.record = (
            JaxDatabase.get_rand_record(api_name) if record is None else record
        )
        self.args = JaxAPI.generate_args_from_record(self.record)
        self.is_class = inspect.isclass(eval(self.api))

    def new_record(self, record):
        self.record = record
        self.args = JaxAPI.generate_args_from_record(record)

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
            # if enable_db and do_select_from_db():
            #     new_arg, success = TFDatabase.select_rand_over_db(self.api, arg_name)
            #     if success:
            #         new_arg = TFArgument.generate_arg_from_signature(new_arg)
            #         self.args[arg_name] = new_arg
            #         do_value_mutation = False
            if enable_value and do_value_mutation:
                arg.mutate_value()

    def to_code(
        self,
        prefix="arg",
        res="res",
        use_try=False,
        error_res="",
        use_old_tensor=False,
    ) -> str:
        args, kwargs = self._get_args()
        arg_code, arg_str = self._to_arg_code(
            args, kwargs, prefix=prefix, use_old_tensor=use_old_tensor
        )
        res_code = ""
        if self.is_class:
            # FIXME: I change the instrumentation of input of class
            cls_name = f"{prefix}_class"
            arg_code += f"{cls_name} = {self.api}({arg_str})\n"

            input_args, input_kwargs = self._get_input_args()
            input_arg_code, input_arg_str = self._to_arg_code(
                input_args,
                input_kwargs,
                f"{prefix}_input",
                use_old_tensor=use_old_tensor,
            )

            arg_code += input_arg_code
            res_code += f"{res} = {cls_name}({input_arg_str})\n"
        else:
            res_code = f"{res} = {self.api}({arg_str})\n"
        invocation = JaxAPI._to_invocation_code(
            arg_code, res_code, use_try=use_try, error_res=error_res
        )
        return invocation

    def to_differential_fn_code(
        self,
        prefix="arg",
        sum=False,
        dtypes=JaxArgument._inexact_float_dtypes,
        limit_max_value=None,
    ):
        """Return (wrapper_function_code, global_inv_code, diff_arg_list)"""
        wrapper_fn_code = ""
        global_invo_code = ""
        tensor_info_code = "# Tensor Info\n"

        args, kwargs = self._get_args()

        arg_str_list = []
        diff_arg_str_list = []
        index = 0
        for arg in args:
            arg_name = f"{prefix}_{index}"
            if arg.is_differentiable(dtypes, limit_max_value=limit_max_value):
                global_invo_code += arg.to_code(arg_name)
                diff_arg_str_list.append(arg_name)
                info = json.dumps(
                    {
                        "name": arg_name,
                        "type": str(arg.type),
                        "dtype": str(arg.dtype),
                    }
                )
                tensor_info_code += f"# {info}\n"
            else:
                wrapper_fn_code += arg.to_code(arg_name)

            arg_str_list.append(arg_name)
            index += 1
        for key, arg in kwargs.items():
            arg_name = key
            if arg.is_differentiable(dtypes, limit_max_value=limit_max_value):
                global_invo_code += arg.to_code(arg_name)
                diff_arg_str_list.append(arg_name)
                info = json.dumps(
                    {
                        "name": arg_name,
                        "type": str(arg.type),
                        "dtype": str(arg.dtype),
                    }
                )
                tensor_info_code += f"# {info}\n"
            else:
                wrapper_fn_code += arg.to_code(arg_name)
            arg_str_list.append(f"{key}={key}")

        fn_code = f"def fn({', '.join(diff_arg_str_list)}):\n"
        fn_code += API.indent_code(wrapper_fn_code)
        # TODO: for class API
        if sum:
            fn_code += API.indent_code(
                f"return {self.api}({', '.join(arg_str_list)}).sum()\n"
            )
        else:
            fn_code += API.indent_code(
                f"return {self.api}({', '.join(arg_str_list)})\n"
            )
        fn_code = tensor_info_code + fn_code
        return fn_code, global_invo_code, diff_arg_str_list

    def to_grad_code(self, res1="res1", res2="res2", err1="err1", err2="err2"):
        fn_code, inv_code, diff_arg_list = self.to_differential_fn_code(
            sum=True
        )
        if len(diff_arg_list) == 0:
            return ""

        code = fn_code

        test_code = inv_code
        test_code += f"{res1} = fn({', '.join(diff_arg_list)})\n"
        code += self.try_except_code(test_code, err1)

        test_code = inv_code
        indices = [str(i) for i in range(len(diff_arg_list))]
        test_code += f"{res2}, _ = jax.value_and_grad(fn, ({','.join(indices)}))({', '.join(diff_arg_list)})\n"
        code += self.try_except_code(test_code, err2)

        return code

    def to_gradcheck_code(self, order=1, err1="err1", err2="err2", mode="rev"):
        """
        Compare the status of normal and check_grads
        """
        fn_code, inv_code, diff_arg_list = self.to_differential_fn_code(
            dtypes=[jnp.float64],
            limit_max_value=100,
        )
        if len(diff_arg_list) == 0:
            return ""

        code = fn_code

        test_code = inv_code
        test_code += f"fn({', '.join(diff_arg_list)})\n"
        code += self.try_except_code(test_code, err1)

        test_code = inv_code
        test_code += "from jax._src.public_test_util import check_grads\n"
        test_code += f"check_grads(fn, ({', '.join(diff_arg_list)},), order={order}, modes=('{mode}'), atol=1e-1, rtol=1e-3)\n"
        # test_code += "from helper_jax import gradient_tolerance, gradcheck\n"
        # test_code += (
        #     f"gradcheck(fn, ({', '.join(diff_arg_list)},), mode='{mode}')\n"
        # )
        code += self.try_except_code(test_code, err2)

        return code

    def to_rev_fwd_code(
        self,
        res1="res1",
        res2="res2",
        err1="err1",
        err2="err2",
        filter_complex=True,
    ):
        # use filter_complex to filter complex input for forward mode
        fn_code, inv_code, diff_arg_list = self.to_differential_fn_code(
            dtypes=JaxArgument._inexact_float_dtypes,
        )
        if len(diff_arg_list) == 0:
            return ""

        indices = [str(i) for i in range(len(diff_arg_list))]
        argnums = f"({','.join(indices)})"

        code = ""
        code += fn_code

        # forward pass of computation
        code += inv_code
        code += "try:\n"
        code += API.indent_code(f"res = fn({', '.join(diff_arg_list)})\n")
        code += "except Exception:\n"
        code += API.indent_code(f"{err1} = 'forward pass error'\n")
        code += API.indent_code(f"{err2} = 'forward pass error'\n")

        temp_code = ""
        # it needs to check whether the output has nan first
        temp_code += "from helper_jax import *\n"
        temp_code += f"if is_nan_output(res):\n"
        temp_code += API.indent_code(f"{err1} = 'is nan'\n")
        temp_code += API.indent_code(f"{err2} = 'is nan'\n")

        # use any_complex_output to filter complex output for reverse mode
        temp_code += f"elif any_complex_output(res):\n"
        temp_code += API.indent_code(f"{err1} = 'is complex'\n")
        temp_code += API.indent_code(f"{err2} = 'is comple'\n")
        temp_code += "else:\n"

        else_code = ""
        rev_code = inv_code
        rev_code += (
            f"{res1} = jax.jacrev(fn, {argnums})({', '.join(diff_arg_list)})\n"
        )
        else_code += self.try_except_code(rev_code, err1)

        fwd_code = inv_code
        fwd_code += (
            f"{res2} = jax.jacfwd(fn, {argnums})({', '.join(diff_arg_list)})\n"
        )
        else_code += self.try_except_code(fwd_code, err2)

        temp_code += API.indent_code(else_code)

        code += "else:\n"
        code += API.indent_code(temp_code)
        return code

    def _to_arg_code(self, args, kwargs, prefix="arg", use_old_tensor=False):
        arg_code = ""
        arg_str_list = []
        index = 0
        for arg in args:
            arg_code += arg.to_code(
                f"{prefix}_{index}", use_old_tensor=use_old_tensor
            )
            arg_str_list.append(f"{prefix}_{index}")
            index += 1
        for key, arg in kwargs.items():
            arg_code += arg.to_code(key, use_old_tensor=use_old_tensor)
            arg_str_list.append(f"{key}={key}")
        return arg_code, ", ".join(arg_str_list)

    def _get_input_args(self):
        """
        Return the args, kwargs of input_signature for class API
        """
        args = []
        kwargs = {}
        feature = "input_signature_"

        for key, value in self.args.items():
            if not key.startswith(feature):
                continue
            key = key.replace(feature, "")
            if key.startswith("parameter:"):
                args.append(value)
            else:
                kwargs[key] = value
        return args, kwargs

    def _get_args(self):
        """
        Return the args, kwargs of API (not including input_signature for class API)
        """
        args = []
        kwargs = {}
        for key, value in self.args.items():
            if key.startswith("parameter:"):
                args.append(value)
            elif key.startswith("input_signature_"):
                pass
            else:
                kwargs[key] = value
        return args, kwargs

    # def get_differentiable_arg_names(self, filter_complex=True) -> list[str]:
    #     diff_arg_names = []
    #     for arg in self.args.values():
    #         if arg.is_differentiable(filter_complex):
    #             diff_arg_names.append(arg.get_var_name())
    #     return diff_arg_names

    @staticmethod
    def _to_invocation_code(
        arg_code, res_code, use_try=False, error_res="", else_code=""
    ):
        if use_try:
            code = arg_code
            t_code = API.try_except_code(res_code, error_res, else_code)
            code += t_code
        else:
            code = arg_code + res_code
        return code

    @staticmethod
    def generate_args_from_record(record: dict) -> dict:
        args = {}
        for key, value in record.items():
            if key == "input_signature":
                # NOTE: the input_signature is a dict, which is different than pytorch
                input_args = JaxAPI.generate_args_from_record(value)
                for k, v in input_args.items():
                    args[f"input_signature_{k}"] = v
            else:
                args[key] = JaxArgument.generate_arg_from_signature(value)
        return args
