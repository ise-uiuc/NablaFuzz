from classes.torch_api import *
from classes.library import Library
from classes.argument import *
from classes.api import *
from os.path import join
from constant.keys import *


class TorchLibrary(Library):
    def __init__(
        self, output_dir, diff_bound=1e-5, time_bound=10, time_thresold=1e-3
    ) -> None:
        super().__init__(output_dir)
        self.diff_bound = diff_bound
        self.time_bound = time_bound
        self.time_thresold = time_thresold

    def test_with_oracle(self, api: TorchAPI, oracle: OracleType):
        if oracle == OracleType.CRASH:
            # We need call another process to catch the crash error
            code = "import torch\n"
            code += self.generate_code(api, oracle)
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(code)
            _, error = self.run_code(code)
            if error is None:
                self.write_to_dir(join(self.output[oracle], "success"), code)
            elif "INTERNAL ASSERT" in error:
                self.write_to_dir(
                    join(self.output[oracle], "potential-bug"), code
                )
            else:
                self.write_to_dir(join(self.output[oracle], "fail"), code)
        elif oracle == OracleType.CUDA:
            code = "import torch\n"
            code += api.to_code(
                res=f'{RES_KEY}["{RES_CPU_KEY}"]',
                use_try=True,
                error_res=f'{RES_KEY}["{ERR_CPU_KEY}"]',
            )
            code += api.to_diff_code(
                oracle,
                res=f'{RES_KEY}["{RES_GPU_KEY}"]',
                use_try=True,
                error_res=f'{RES_KEY}["{ERR_GPU_KEY}"]',
            )

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)

            write_dir = ""
            if error is None:
                # first check the correctness
                if (
                    results[ERR_CPU_KEY] is None
                    and results[ERR_GPU_KEY] is None
                ):
                    if self.is_equal(
                        results[RES_CPU_KEY],
                        results[RES_GPU_KEY],
                        self.diff_bound,
                    ):
                        write_dir = join(self.output[oracle], "success")
                    else:
                        write_dir = join(self.output[oracle], "potential-bug")
                elif (
                    results[ERR_CPU_KEY] is None
                    or results[ERR_GPU_KEY] is None
                ):
                    write_dir = join(self.output[oracle], "potential-bug")
                elif (
                    "INTERNAL ASSERT" in results[ERR_CPU_KEY]
                    or "INTERNAL ASSERT" in results[ERR_GPU_KEY]
                ):
                    write_dir = join(self.output[oracle], "potential-bug")
                else:
                    write_dir = join(self.output[oracle], "success")
            elif "INTERNAL ASSERT" in error:
                write_dir = join(self.output[oracle], "potential-bug")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, write_code)
        elif oracle == OracleType.PRECISION:
            code = "import torch\n"
            code += "import time\n"
            code += api.to_code(
                res=f'results["{TIME_LOW_KEY}"]', low_precision=True
            )
            code += api.to_diff_code(oracle, res=f'results["{TIME_HIGH_KEY}"]')

            write_code = "results = dict()\n" + code + "\nprint(results)\n"
            with open(join(self.directory, "temp.py"), "w") as f:
                f.write(write_code)

            results, error = self.run_code(code)
            if error is None:
                if isinstance(results[TIME_LOW_KEY], float) and isinstance(
                    results[TIME_HIGH_KEY], float
                ):
                    if (
                        results[TIME_LOW_KEY]
                        > self.time_bound * results[TIME_HIGH_KEY]
                        and results[TIME_HIGH_KEY] > self.time_thresold
                    ):
                        write_dir = join(self.output[oracle], "potential-bug")
                    else:
                        write_dir = join(self.output[oracle], "success")
                else:
                    write_dir = join(self.output[oracle], "fail")
            else:
                write_dir = join(self.output[oracle], "fail")
            self.write_to_dir(write_dir, write_code)

    @staticmethod
    def generate_code(api: TorchAPI, oracle: OracleType) -> str:
        if oracle == OracleType.CRASH:
            return api.to_code()
        elif oracle == OracleType.CUDA:
            code = api.to_code(res="cpu_res", use_try=True)
            code += api.to_diff_code(oracle, res="cuda_res", use_try=True)
            return code
        elif oracle == OracleType.PRECISION:
            code = api.to_code(res="low_res", low_precision=True)
            code += api.to_diff_code(oracle, res="high_res")
            return code
        else:
            assert 0

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

    @staticmethod
    def is_equal(x, y, atol=1e-1, rtol=1e-3, promoted=True, equal_nan=False):
        try:
            return TorchLibrary.is_equal_unsafe(
                x,
                y,
                atol=atol,
                rtol=rtol,
                promoted=promoted,
                equal_nan=equal_nan,
            )
        except Exception:
            return False

    @staticmethod
    def is_equal_unsafe(
        x, y, atol=1e-1, rtol=1e-3, promoted=True, equal_nan=False
    ):
        def eq_float_tensor(x, y):
            # not strictly equal
            return torch.allclose(
                x, y, atol=atol, rtol=rtol, equal_nan=equal_nan
            )

        x_type = TorchArgument.get_type(x)
        y_type = TorchArgument.get_type(y)
        if x_type != y_type:
            if x_type == ArgType.TORCH_TENSOR and y_type in [
                ArgType.LIST,
                ArgType.TUPLE,
            ]:
                flag = False
                for temp in y:
                    flag = flag or TorchLibrary.is_equal(
                        x, temp, atol, rtol, promoted
                    )
                return flag
            elif y_type == ArgType.TORCH_TENSOR and x_type in [
                ArgType.LIST,
                ArgType.TUPLE,
            ]:
                flag = False
                for temp in x:
                    flag = flag or TorchLibrary.is_equal(
                        y, temp, atol, rtol, promoted
                    )
                return flag
            else:
                return False
        if x_type == ArgType.TORCH_TENSOR:
            if x.shape != y.shape:
                return False
            # compare the tensor at cpu
            x = x.clone().cpu()
            y = y.clone().cpu()
            if x.dtype != y.dtype:
                if not promoted:
                    return False
                else:
                    # we need to cast the high-precision one into low-precision
                    promoted_type = torch.promote_types(x.dtype, y.dtype)
                    low_precision_dtype = (
                        x.dtype if x.dtype != promoted_type else y.dtype
                    )
                    x = x.to(dtype=low_precision_dtype)
                    y = y.to(dtype=low_precision_dtype)

            if x.is_sparse:
                x = x.to_dense()
            if y.is_sparse:
                y = y.to_dense()
            if x.is_complex():
                if not y.is_complex():
                    return False
                return eq_float_tensor(x.real, y.real) and eq_float_tensor(
                    x.imag, y.imag
                )
            if not x.dtype.is_floating_point:
                return torch.equal(x.cpu(), y.cpu())
            return eq_float_tensor(x, y)
        elif x_type == ArgType.FLOAT:
            return abs(x - y) < atol + rtol * abs(x)
        elif x_type in [ArgType.LIST, ArgType.TUPLE]:
            if len(x) != len(y):
                return False
            for i in range(len(x)):
                if (
                    TorchLibrary.is_equal(
                        x[i], y[i], atol, rtol, promoted, equal_nan=equal_nan
                    )
                    is False
                ):
                    return False
            return True
        else:
            return x == y


def test():
    api_name = "torch.nn.Conv2d"
    api = TorchAPI(api_name)
    MyPytorch = TorchLibrary("torch-output")
    print(MyPytorch.generate_code(api, OracleType.CRASH))
    print(MyPytorch.generate_code(api, OracleType.CUDA))
    print(MyPytorch.generate_code(api, OracleType.PRECISION))
    MyPytorch.test_with_oracle(api, OracleType.CRASH)
    MyPytorch.test_with_oracle(api, OracleType.CUDA)
    MyPytorch.test_with_oracle(api, OracleType.PRECISION)
    # print(TorchArgument.get_type(1))
