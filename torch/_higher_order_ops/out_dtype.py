# mypy: allow-untyped-defs

import contextlib

import torch
import torch.utils._pytree as pytree
from torch._C import DispatchKey
from torch._higher_order_ops.utils import autograd_not_implemented
from torch._ops import HigherOrderOperator
from torch._prims_common import elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND
from torch._subclasses.fake_tensor import (
    fake_tensor_tls,
    FakeTensorMode,
    maybe_get_fake_mode,
)
from torch._subclasses.functional_tensor import FunctionalTensor, FunctionalTensorMode
from torch.fx.experimental.proxy_tensor import (
    disable_proxy_modes_tracing,
    get_proxy_slot,
    maybe_handle_decomp,
    ProxyTorchDispatchMode,
    set_proxy_slot,
    track_tensor_tree,
)
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    transform_subclass,
)


# TODO to figure out a more generic approach
ALLOWABLE_OPS = [
    torch.ops.aten.linear.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.conv2d.default,
    torch.ops.aten.convolution.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.div.Tensor,
    torch.ops.aten.div.Scalar,
]


class OutDtypeOperator(HigherOrderOperator):
    """
    The out_dtype operator takes an existing ATen functional operator, an
    `out_dtype` argument, and arguments to the original operator, and executes
    the original operator and returns a Tensor with the `out_dtype` precision.
    This operator does not mandate a compute precision so it allows the
    representation to not be opinionated about the exact implementation.

    The general implementation for all operators will be the following:
        1. Promote inputs dtypes based on default PyTorch dtype promotion rules,
            using the dtypes of all input Tensors/Scalars and the `out_dtype`
            arugument.
        2. Execute the operator
        3. Cast the output to `out_dtype`
    """

    def __init__(self) -> None:
        super().__init__("out_dtype")

    def __call__(self, op, output_dtype, *args):
        validate_out_dtype_args(op)

        # pyrefly: ignore [missing-attribute]
        res = super().__call__(op, output_dtype, *args)

        return res


out_dtype = OutDtypeOperator()


def validate_out_dtype_args(op):
    if not isinstance(op, torch._ops.OpOverload):
        raise ValueError("out_dtype's first argument must be an OpOverload")
    if op._schema.is_mutable:
        raise ValueError(
            "out_dtype's first argument needs to be a functional operator"
        )
    if not (
        len(op._schema.returns) == 1
        and isinstance(op._schema.returns[0].type, torch.TensorType)
    ):
        raise ValueError(
            "out_dtype's can only apply to ops that return a single tensor"
            f"Instead got {[r.type for r in op._schema.returns]}"
        )

    if op not in ALLOWABLE_OPS:
        raise ValueError(
            f"out_dtype only allows the following operators: {ALLOWABLE_OPS}."
        )


def _detect_functional_mode():
    return torch.utils._python_dispatch._detect_infra_mode(
        torch._C._TorchDispatchModeKey.FUNCTIONAL
    )


def _unwrap_temporary_functional_result(out):
    if isinstance(out, torch.Tensor) and is_traceable_wrapper_subclass(out):
        unwrapped = transform_subclass(
            out, lambda _, inner_t: _unwrap_temporary_functional_result(inner_t)
        )
        torch._mirror_autograd_meta_to(out, unwrapped)  # type: ignore[attr-defined]
        return unwrapped

    if not isinstance(out, FunctionalTensor):
        if isinstance(out, torch.Tensor) and torch._is_functional_tensor(  # type: ignore[attr-defined]
            out
        ):
            raise AssertionError("expected non-functional tensor")
        return out

    torch._sync(out)
    return torch._from_functional_tensor(out.elem)


def _copy_tracked_tensor_tree(src, dst, tracer):
    if isinstance(src, torch.Tensor):
        if not isinstance(dst, torch.Tensor):
            raise AssertionError(f"expected tensor output, got {type(dst)}")
        set_proxy_slot(dst, tracer, get_proxy_slot(src, tracer))
        return

    if isinstance(src, tuple):
        if not isinstance(dst, tuple) or len(src) != len(dst):
            raise AssertionError("expected matching tuple outputs")
        for src_item, dst_item in zip(src, dst):
            _copy_tracked_tensor_tree(src_item, dst_item, tracer)
        return

    if isinstance(src, list):
        if not isinstance(dst, list) or len(src) != len(dst):
            raise AssertionError("expected matching list outputs")
        for src_item, dst_item in zip(src, dst):
            _copy_tracked_tensor_tree(src_item, dst_item, tracer)
        return

    if isinstance(src, dict):
        if not isinstance(dst, dict) or src.keys() != dst.keys():
            raise AssertionError("expected matching dict outputs")
        for key in src:
            _copy_tracked_tensor_tree(src[key], dst[key], tracer)


def _copy_wrapper_subclass_inner_proxy_slots(src, dst, tracer):
    if not is_traceable_wrapper_subclass(src):
        return
    if not is_traceable_wrapper_subclass(dst):
        raise AssertionError(f"expected subclass output, got {type(dst)}")

    src_attrs, _ = src.__tensor_flatten__()  # type: ignore[attr-defined]
    dst_attrs, _ = dst.__tensor_flatten__()  # type: ignore[attr-defined]
    if src_attrs != dst_attrs:
        raise AssertionError("expected matching subclass attrs")

    tensor_attrs = [
        attr for attr in src_attrs if isinstance(getattr(src, attr), torch.Tensor)
    ]
    fallback_proxy = (
        get_proxy_slot(src, tracer, None) if len(tensor_attrs) == 1 else None
    )

    for attr in src_attrs:
        src_inner = getattr(src, attr)
        dst_inner = getattr(dst, attr)
        if not isinstance(dst_inner, torch.Tensor):
            continue

        proxy = get_proxy_slot(src_inner, tracer, None)
        if proxy is None:
            proxy = fallback_proxy
        if proxy is not None:
            set_proxy_slot(dst_inner, tracer, proxy)

        _copy_wrapper_subclass_inner_proxy_slots(src_inner, dst_inner, tracer)


def _dispatch_out_dtype_dense(
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
    fake_mode: FakeTensorMode | None = None,
    unwrap_temporary_output: bool = False,
):
    functional_mode = _detect_functional_mode()
    if fake_mode is None:
        for arg in pytree.arg_tree_leaves(*args):
            fake_mode = maybe_get_fake_mode(arg)
            if fake_mode is not None:
                break
    owns_functional_mode = functional_mode is None
    functional_mode_ctx = (
        functional_mode if functional_mode is not None else FunctionalTensorMode()
    )
    fake_mode_ctx = fake_mode if fake_mode is not None else contextlib.nullcontext()
    old_allow_non_fake_inputs = fake_tensor_tls.allow_non_fake_inputs_override
    fake_tensor_tls.allow_non_fake_inputs_override = True
    try:
        with functional_mode_ctx, fake_mode_ctx:
            out = out_dtype_dense(op, output_dtype, *args)
            if unwrap_temporary_output and owns_functional_mode:
                out = pytree.tree_map(_unwrap_temporary_functional_result, out)
            return out
    finally:
        fake_tensor_tls.allow_non_fake_inputs_override = old_allow_non_fake_inputs


def traceable_out_dtype_dense(
    op: torch._ops.OpOverload, output_dtype: torch.dtype, *args
):
    return _dispatch_out_dtype_dense(
        op,
        output_dtype,
        *args,
        unwrap_temporary_output=True,
    )


def trace_out_dtype(proxy_mode, func_overload, op, output_dtype, *args):
    # NB: Long-term we should put the decomposition logic into
    # ProxyTorchDispatchMode so that people do not need to call maybe_handle_decomp
    # in all HigherOrderOp proxy implementations.
    r = maybe_handle_decomp(proxy_mode, func_overload, (op, output_dtype, *args), {})
    if r is not NotImplemented:
        return r

    has_wrapper_subclass_arg = any(
        isinstance(arg, torch.Tensor) and is_traceable_wrapper_subclass(arg)
        for arg in pytree.arg_tree_leaves(*args)
    )
    if has_wrapper_subclass_arg:
        functional_mode = _detect_functional_mode()
        out = _dispatch_out_dtype_dense(op, output_dtype, *args)
        if functional_mode is not None:
            return out

        safe_out = pytree.tree_map(_unwrap_temporary_functional_result, out)
        _copy_tracked_tensor_tree(out, safe_out, proxy_mode.tracer)
        _copy_wrapper_subclass_inner_proxy_slots(out, safe_out, proxy_mode.tracer)
        return safe_out

    with disable_proxy_modes_tracing():
        out = traceable_out_dtype_dense(op, output_dtype, *args)

    node_args = (op, output_dtype, *args)
    proxy_args = pytree.tree_map(proxy_mode.tracer.unwrap_proxy, node_args)
    out_proxy = proxy_mode.tracer.create_proxy(
        "call_function", func_overload, proxy_args, {}, name="out_dtype"
    )
    return track_tensor_tree(out, out_proxy, constant=None, tracer=proxy_mode.tracer)


@out_dtype.py_impl(DispatchKey.CompositeExplicitAutograd)
def out_dtype_dense(op: torch._ops.OpOverload, output_dtype: torch.dtype, *args):
    if is_int_mm(op, output_dtype, args):
        return torch._int_mm(*args)
    return out_dtype_fallback(op, output_dtype, *args)


def is_int_mm(op, output_dtype, args):
    return (
        op is torch.ops.aten.mm.default
        and output_dtype == torch.int32
        and len(args) == 2
        and args[0].dtype == torch.int8
        and args[1].dtype == torch.int8
        and (args[0].is_cuda or args[0].is_xpu)
        and (args[1].is_cuda or args[1].is_xpu)
    )


def out_dtype_fallback(op, output_dtype, *args):
    flat_inputs = pytree.arg_tree_leaves(*args) + [torch.ones(1, dtype=output_dtype)]
    promote_dtype: torch.dtype = elementwise_dtypes(
        *flat_inputs,
        type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    )[0]

    casted_args = pytree.tree_map_only(
        torch.Tensor, lambda arg: arg.to(dtype=promote_dtype), args
    )
    res = op(*casted_args).to(dtype=output_dtype)
    return res


out_dtype.py_autograd_impl(autograd_not_implemented(out_dtype, deferred_error=True))


@out_dtype.py_impl(ProxyTorchDispatchMode)
def out_dtype_proxy(
    mode: ProxyTorchDispatchMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    return trace_out_dtype(mode, out_dtype, op, output_dtype, *args)


@out_dtype.py_impl(FakeTensorMode)
def out_dtype_fake_tensor_mode(
    mode: FakeTensorMode,
    op: torch._ops.OpOverload,
    output_dtype: torch.dtype,
    *args,
):
    return _dispatch_out_dtype_dense(
        op,
        output_dtype,
        *args,
        fake_mode=mode,
        unwrap_temporary_output=True,
    )


@out_dtype.py_functionalize_impl
def out_dtype_func(ctx, op, output_dtype, *args):
    unwrapped_args = tuple(ctx.unwrap_tensors(arg) for arg in args)

    with ctx.redispatch_to_next():
        res = out_dtype(op, output_dtype, *unwrapped_args)
    return ctx.wrap_tensors(res)


# Generated Dynamo graphs reference these helpers through the exported operator
# object at `torch._higher_order_ops.out_dtype`, not this module.
out_dtype._dispatch_out_dtype_dense = _dispatch_out_dtype_dense  # type: ignore[attr-defined]
out_dtype._unwrap_temporary_functional_result = (  # type: ignore[attr-defined]
    _unwrap_temporary_functional_result
)
out_dtype.elementwise_dtypes = elementwise_dtypes  # type: ignore[attr-defined]
out_dtype.is_int_mm = is_int_mm  # type: ignore[attr-defined]
out_dtype.out_dtype_dense = out_dtype_dense  # type: ignore[attr-defined]
out_dtype.out_dtype_fake_tensor_mode = out_dtype_fake_tensor_mode  # type: ignore[attr-defined]
out_dtype.out_dtype_fallback = out_dtype_fallback  # type: ignore[attr-defined]
out_dtype.out_dtype_func = out_dtype_func  # type: ignore[attr-defined]
out_dtype.out_dtype_proxy = out_dtype_proxy  # type: ignore[attr-defined]
out_dtype.trace_out_dtype = trace_out_dtype  # type: ignore[attr-defined]
out_dtype.traceable_out_dtype_dense = traceable_out_dtype_dense  # type: ignore[attr-defined]
