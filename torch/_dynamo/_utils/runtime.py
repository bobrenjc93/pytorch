from __future__ import annotations

import contextlib
import copy
import inspect
import re
import warnings
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Any

import torch
from torch._C import (
    _len_torch_function_stack,
    _pop_torch_function_stack,
    _push_on_torch_function_stack,
)


if TYPE_CHECKING:
    from types import CodeType

    from collections.abc import Generator


class Lit:
    def __init__(self, s: str) -> None:
        self.s = s

    def __repr__(self) -> str:
        return self.s


warn_once_cache: set[str] = set()


def warn_once(msg: str, stacklevel: int = 1) -> None:
    # Dynamo causes all warnings.warn (in user code and in Dynamo code) to print all the time.
    # https://github.com/pytorch/pytorch/issues/128427.
    # warn_once is a workaround: if the msg has been warned on before, then we will not
    # warn again.
    # NB: it's totally ok to store a cache of all the strings: this is what warnings.warn does as well.
    if msg in warn_once_cache:
        return
    warn_once_cache.add(msg)
    warnings.warn(msg, stacklevel=stacklevel + 1)


def strip_color_from_string(text: str) -> str:
    # This regular expression matches ANSI escape codes
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", text)


@contextlib.contextmanager
def _disable_saved_tensors_hooks_during_tracing() -> Generator[None, None, None]:
    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    try:
        prior = torch._C._autograd._saved_tensors_hooks_set_tracing(True)
        yield
    finally:
        torch._C._autograd._saved_tensors_hooks_set_tracing(prior)


def is_parameter_freezing() -> bool:
    return torch._inductor.config.freezing and not torch.is_grad_enabled()


def get_torch_function_mode_stack() -> list[Any]:
    return [
        get_torch_function_mode_stack_at(i) for i in range(_len_torch_function_stack())
    ]


def get_torch_function_mode_stack_at(ind: int) -> Any:
    assert ind < _len_torch_function_stack() and ind >= 0
    return torch._C._get_function_stack_at(ind)


def set_torch_function_mode_stack(stack: list[Any]) -> None:
    for _ in range(_len_torch_function_stack()):
        _pop_torch_function_stack()

    for mode in stack:
        _push_on_torch_function_stack(mode)


def clear_torch_function_mode_stack() -> None:
    for _ in range(_len_torch_function_stack()):
        _pop_torch_function_stack()


def get_current_stream(device: torch.device) -> torch.Stream:
    return torch.accelerator.current_stream(device)


# call from C dynamo in order to inspect values in pdb
def _breakpoint_for_c_dynamo(*args: Any) -> None:
    breakpoint()


def verify_guard_fn_signature(value: Any) -> None:
    fn = value.__metadata_guard__
    sig = inspect.signature(fn)
    if len(sig.parameters) != 2:
        from ..exc import InternalTorchDynamoError

        raise InternalTorchDynamoError(
            "Tensor subclass method __metadata_guard__ must take exactly two subclass metadata arguments"
        )
    if fn.__self__ != value.__class__:
        from ..exc import InternalTorchDynamoError

        raise InternalTorchDynamoError(
            "Tensor subclass method __metadata_guard__ must be a classmethod"
        )


# Helper functions below are to prevent TorchDynamo to prevent tracing of
# __torch_function__ calls triggered on tensor properties in the pre graph
# bytecode.
@torch._disable_dynamo
def call_size(x: Any, i: int) -> int:
    return x.size(i)


@torch._disable_dynamo
def call_stride(x: Any, i: int) -> int:
    return x.stride(i)


@torch._disable_dynamo
def call_storage_offset(x: Any) -> int:
    return x.storage_offset()


def _extract_tensor_dict(t: torch.Tensor) -> dict[str, Any]:
    KEYS_TO_COPY = [
        "_dynamo_static_input_type",
        "tag",
    ]

    tensor_dict = {
        key: copy.copy(t.__dict__[key]) for key in KEYS_TO_COPY if key in t.__dict__
    }

    return tensor_dict


def build_stream(args: tuple[Any], kwargs: dict[Any, Any]) -> torch.Stream:
    return torch._C.Stream(*args, **kwargs)


def build_event(args: tuple[Any], kwargs: dict[Any, Any]) -> torch.Event:
    return torch._C.Event(*args, **kwargs)


@torch._disable_dynamo
def record_pregraph_bytecode_enter() -> AbstractContextManager[None]:
    cm: AbstractContextManager[None] = (
        torch._C._profiler._RecordFunctionFast("Pregraph bytecode")
        if torch.autograd.profiler._is_profiler_enabled
        else contextlib.nullcontext()
    )
    cm.__enter__()
    return cm


@torch._disable_dynamo
def record_pregraph_bytecode_exit(cm: AbstractContextManager[None]) -> None:
    cm.__exit__(None, None, None)


def get_traced_code() -> list[CodeType] | None:
    from torch._guards import TracingContext

    return TracingContext.get_traced_code()
