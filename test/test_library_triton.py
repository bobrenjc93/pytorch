# Owner(s): ["module: unknown"]

import sys
import types
from contextlib import contextmanager
from functools import lru_cache
from unittest import mock

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class FakeJITFunction:
    def __init__(self, name):
        self.__name__ = name


class FakeAutotuner:
    def __init__(self, name):
        self.__name__ = name


class FakeKernelWrapper:
    def __init__(self, fn):
        self.fn = fn


def capture_triton(kernel):
    raise AssertionError("test should inspect this function, not execute it")


@contextmanager
def fake_triton_runtime():
    triton_mod = types.ModuleType("triton")
    runtime_mod = types.ModuleType("triton.runtime")
    autotuner_mod = types.ModuleType("triton.runtime.autotuner")
    jit_mod = types.ModuleType("triton.runtime.jit")

    autotuner_mod.Autotuner = FakeAutotuner
    jit_mod.JITFunction = FakeJITFunction
    runtime_mod.autotuner = autotuner_mod
    runtime_mod.jit = jit_mod
    triton_mod.runtime = runtime_mod

    with mock.patch.dict(
        sys.modules,
        {
            "triton": triton_mod,
            "triton.runtime": runtime_mod,
            "triton.runtime.autotuner": autotuner_mod,
            "triton.runtime.jit": jit_mod,
        },
    ):
        yield


class TestLibraryTriton(TestCase):
    def assertKernelNames(self, kernels, names):
        self.assertEqual([kernel.__name__ for kernel in kernels], names)

    def test_get_inner_triton_kernels_public_torch_library_wrapper(self):
        from torch._library.triton import get_inner_triton_kernels

        kernel = FakeJITFunction("kernel")
        wrapped_kernel = FakeKernelWrapper(kernel)

        def identity_wrapper(fn):
            return fn

        def triton_op_body():
            wrapped = identity_wrapper(wrapped_kernel)
            torch.library.wrap_triton(wrapped)[None]()

        with fake_triton_runtime():
            self.assertKernelNames(get_inner_triton_kernels(triton_op_body), ["kernel"])

    def test_get_inner_triton_kernels_helper_and_lru_cache_factory(self):
        from torch._library.triton import get_inner_triton_kernels

        kernel = FakeJITFunction("factory_kernel")

        @lru_cache
        def get_kernel():
            return kernel

        def helper():
            local_kernel = get_kernel()
            capture_triton(local_kernel)[None]()

        def triton_op_body():
            helper()
            return get_kernel()

        with fake_triton_runtime():
            self.assertKernelNames(
                get_inner_triton_kernels(triton_op_body), ["factory_kernel"]
            )

    def test_get_inner_triton_kernels_registered_torch_op(self):
        import torch._library.custom_ops
        from torch._library.triton import get_inner_triton_kernels

        kernel = FakeJITFunction("custom_op_kernel")

        def fake_custom_op_impl():
            torch._library.capture_triton(kernel)[None]()

        opdef = types.SimpleNamespace(_abstract_fn=fake_custom_op_impl)

        def triton_op_body():
            torch.ops.fake_ns.fake_op()

        with (
            fake_triton_runtime(),
            mock.patch.dict(
                torch._library.custom_ops.OPDEFS, {"fake_ns::fake_op": opdef}
            ),
        ):
            self.assertKernelNames(
                get_inner_triton_kernels(triton_op_body), ["custom_op_kernel"]
            )


if __name__ == "__main__":
    run_tests()
