import math
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch._prims_common import is_float_dtype

from ..pattern_matcher import get_arg_value


aten = torch.ops.aten
prims = torch.ops.prims

_VALUE_PRESERVING_OPS = {
    aten.alias.default,
    aten.clone.default,
    aten.detach.default,
    aten.expand.default,
    aten.permute.default,
    aten.reshape.default,
    aten.squeeze.default,
    aten.squeeze.dim,
    aten.unsqueeze.default,
    aten.view.default,
}

_IOTA_COMPARISONS = {
    aten.eq.Scalar,
    aten.ge.Scalar,
    aten.gt.Scalar,
    aten.le.Scalar,
    aten.lt.Scalar,
    aten.ne.Scalar,
}


def _unwrap_value_preserving_ops(value: Any) -> Any:
    seen: set[torch.fx.Node] = set()
    while (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target in _VALUE_PRESERVING_OPS
    ):
        if value in seen:
            return None
        seen.add(value)
        if not value.args:
            return None
        source = value.args[0]
        if not isinstance(source, torch.fx.Node) or any(
            node is not source for node in value.all_input_nodes
        ):
            return None
        value = source
    return value


def _static_iota_bounds(value: Any) -> tuple[int, int, torch.dtype] | None:
    value = _unwrap_value_preserving_ops(value)
    if (
        not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target is not prims.iota.default
        or value.all_input_nodes
    ):
        return None

    length = get_arg_value(value, 0, "length")
    start = get_arg_value(value, 1, "start")
    step = get_arg_value(value, 2, "step")
    dtype = get_arg_value(value, 3, "dtype")
    if (
        not all(isinstance(arg, int) for arg in (length, start, step))
        or length <= 0
        or step == 0
        or dtype not in (torch.int32, torch.int64)
    ):
        return None

    last = start + (length - 1) * step
    dtype_limits = torch.iinfo(dtype)
    if not (
        dtype_limits.min <= start <= dtype_limits.max
        and dtype_limits.min <= last <= dtype_limits.max
    ):
        return None
    return min(start, last), max(start, last), dtype


def _is_all_true_iota_predicate(value: Any) -> bool:
    value = _unwrap_value_preserving_ops(value)
    if (
        not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target not in _IOTA_COMPARISONS
    ):
        return False

    iota = get_arg_value(value, 0, "self")
    if not isinstance(iota, torch.fx.Node) or any(
        node is not iota for node in value.all_input_nodes
    ):
        return False
    bounds = _static_iota_bounds(iota)
    other = get_arg_value(value, 1, "other")
    if bounds is None or not isinstance(other, int) or isinstance(other, bool):
        return False

    lower, upper, dtype = bounds
    dtype_limits = torch.iinfo(dtype)
    if not dtype_limits.min <= other <= dtype_limits.max:
        return False
    if value.target is aten.eq.Scalar:
        return lower == upper == other
    if value.target is aten.ge.Scalar:
        return lower >= other
    if value.target is aten.gt.Scalar:
        return lower > other
    if value.target is aten.le.Scalar:
        return upper <= other
    if value.target is aten.lt.Scalar:
        return upper < other
    if value.target is aten.ne.Scalar:
        return other < lower or other > upper
    return False


def _scalar_full(value: Any) -> tuple[Any, torch.dtype] | None:
    if (
        not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target is not aten.full.default
        or value.all_input_nodes
        or get_arg_value(value, 0, "size") not in ([], ())
    ):
        return None
    dtype = get_arg_value(value, 2, "dtype")
    if not isinstance(dtype, torch.dtype) or not is_float_dtype(dtype):
        return None
    return get_arg_value(value, 1, "fill_value"), dtype


def _is_all_zero_iota_bias(value: Any) -> bool:
    value = _unwrap_value_preserving_ops(value)
    if (
        not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target is not aten.where.self
    ):
        return False

    condition = get_arg_value(value, 0, "condition")
    zero = _scalar_full(get_arg_value(value, 1, "self"))
    neg_inf = _scalar_full(get_arg_value(value, 2, "other"))
    if zero is None or neg_inf is None or zero[1] is not neg_inf[1]:
        return False
    zero_value, _ = zero
    neg_inf_value, _ = neg_inf
    return (
        isinstance(zero_value, (int, float))
        and not isinstance(zero_value, bool)
        and zero_value == 0
        and isinstance(neg_inf_value, float)
        and math.isinf(neg_inf_value)
        and neg_inf_value < 0
        and _is_all_true_iota_predicate(condition)
    )


def _collect_ancestors(root: torch.fx.Node, result: set[torch.fx.Node]) -> None:
    stack = [root]
    while stack:
        node = stack.pop()
        if node in result:
            continue
        result.add(node)
        stack.extend(node.all_input_nodes)


def remove_all_zero_sdpa_biases(graph: torch.fx.Graph) -> int:
    """Drop efficient-attention biases proven zero by static iota bounds."""
    dead_candidates: set[torch.fx.Node] = set()
    replacements = 0

    for node in graph.nodes:
        if (
            node.op != "call_function"
            or node.target is not aten._scaled_dot_product_efficient_attention.default
        ):
            continue

        bias = get_arg_value(node, 3, "attn_bias")
        is_causal = get_arg_value(node, 6, "is_causal")
        if (
            not isinstance(bias, torch.fx.Node)
            or is_causal not in (None, False)
            or not _is_all_zero_iota_bias(bias)
        ):
            continue

        _collect_ancestors(bias, dead_candidates)
        node.update_arg(3, None)
        if len(node.args) > 6:
            node.update_arg(6, False)
        else:
            node.update_kwarg("is_causal", False)
        replacements += 1

    if replacements:
        for node in reversed(graph.nodes):
            if (
                node in dead_candidates
                and not node.users
                and node.op not in ("placeholder", "output")
                and not node.is_impure()
            ):
                graph.erase_node(node)
        counters["inductor"]["remove_all_zero_sdpa_biases"] += replacements

    return replacements
