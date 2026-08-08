import math
from typing import Any

import torch
from torch._dynamo.utils import counters
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

from ..pattern_matcher import get_arg_value


aten = torch.ops.aten
prims = torch.ops.prims

_AND_OPS = {
    aten.bitwise_and.Tensor,
    aten.logical_and.default,
}

_ALL_TRUE_PASSTHROUGH_OPS = {
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

_INDEX_PASSTHROUGH_OPS = {
    aten.alias.default,
    aten.clone.default,
    aten.detach.default,
    aten.expand.default,
    aten.reshape.default,
    aten.unsqueeze.default,
    aten.view.default,
}

_NOT_UNIFORM = object()


def _known_equal(lhs: Any, rhs: Any) -> bool:
    return statically_known_true(sym_eq(lhs, rhs))


def _tensor_meta(node: torch.fx.Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _unwrap_expands(value: Any) -> Any:
    while (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target is aten.expand.default
    ):
        value = value.args[0]
    return value


def _uniform_value(value: Any) -> Any:
    if not isinstance(value, torch.fx.Node):
        return value
    if value.op != "call_function" or value.target is not aten.full.default:
        return _NOT_UNIFORM
    return get_arg_value(value, 1, "fill_value")


def _is_all_true(value: Any, seen: set[torch.fx.Node] | None = None) -> bool:
    if value is True:
        return True
    if not isinstance(value, torch.fx.Node) or value.op != "call_function":
        return False

    seen = set() if seen is None else seen
    if value in seen:
        return False
    seen.add(value)

    if value.target is aten.full.default:
        meta = _tensor_meta(value)
        fill_value = get_arg_value(value, 1, "fill_value")
        return meta is not None and meta.dtype is torch.bool and fill_value is True
    if value.target in _AND_OPS:
        return all(_is_all_true(arg, seen.copy()) for arg in value.args[:2])
    if value.target is aten.index.Tensor:
        source = value.args[0]
        if not isinstance(source, torch.fx.Node):
            return False
        source_meta = _tensor_meta(source)
        indices = value.args[1]
        if (
            source_meta is None
            or len(indices) > source_meta.ndim
            or not _is_all_true(source, seen)
        ):
            return False
        return all(
            index is None or _is_zero_based_iota(index, source_meta.shape[dim])
            for dim, index in enumerate(indices)
        )
    if value.target in _ALL_TRUE_PASSTHROUGH_OPS:
        return _is_all_true(value.args[0], seen)
    return False


def _is_zero_based_iota(value: Any, length: int | torch.SymInt) -> bool:
    if not isinstance(value, torch.fx.Node) or value.op != "call_function":
        return False
    seen: set[torch.fx.Node] = set()
    while value.target in _INDEX_PASSTHROUGH_OPS:
        if value in seen:
            return False
        seen.add(value)
        value = value.args[0]
        if not isinstance(value, torch.fx.Node) or value.op != "call_function":
            return False

    return (
        value.target is prims.iota.default
        and _known_equal(get_arg_value(value, 0, "length"), length)
        and _known_equal(get_arg_value(value, 1, "start"), 0)
        and _known_equal(get_arg_value(value, 2, "step"), 1)
        and get_arg_value(value, 3, "dtype") is torch.int64
    )


def _match_iota_index(
    value: Any,
    length: int | torch.SymInt,
    other_length: int | torch.SymInt,
    *,
    is_key: bool,
) -> bool:
    if not isinstance(value, torch.fx.Node):
        return False
    meta = _tensor_meta(value)
    if meta is None or meta.dtype is not torch.int64 or meta.ndim < 2:
        return False

    shape = meta.shape
    stride = meta.stride()
    if is_key:
        if not _known_equal(shape[-1], length):
            return False
        if not _known_equal(shape[-2], 1) and not (
            _known_equal(shape[-2], other_length) and _known_equal(stride[-2], 0)
        ):
            return False
    else:
        if not _known_equal(shape[-2], length):
            return False
        if not _known_equal(shape[-1], 1) and not (
            _known_equal(shape[-1], other_length) and _known_equal(stride[-1], 0)
        ):
            return False

    return _is_zero_based_iota(value, length)


def _is_causal_comparison(
    value: Any, q_length: int | torch.SymInt, k_length: int | torch.SymInt
) -> bool:
    return (
        isinstance(value, torch.fx.Node)
        and value.op == "call_function"
        and value.target is aten.le.Tensor
        and _match_iota_index(value.args[0], k_length, q_length, is_key=True)
        and _match_iota_index(value.args[1], q_length, k_length, is_key=False)
    )


def _is_causal_condition(
    value: Any, q_length: int | torch.SymInt, k_length: int | torch.SymInt
) -> bool:
    value = _unwrap_expands(value)
    leaves: list[Any] = []
    stack = [value]
    while stack:
        leaf = stack.pop()
        if (
            isinstance(leaf, torch.fx.Node)
            and leaf.op == "call_function"
            and leaf.target in _AND_OPS
        ):
            stack.extend(leaf.args[:2])
        else:
            leaves.append(leaf)

    causal_leaves = sum(
        _is_causal_comparison(leaf, q_length, k_length) for leaf in leaves
    )
    return causal_leaves == 1 and all(
        _is_causal_comparison(leaf, q_length, k_length) or _is_all_true(leaf)
        for leaf in leaves
    )


def _is_causal_bias(
    bias: Any, q_length: int | torch.SymInt, k_length: int | torch.SymInt
) -> bool:
    if not isinstance(bias, torch.fx.Node):
        return False
    bias_meta = _tensor_meta(bias)
    if (
        bias_meta is None
        or bias_meta.ndim < 2
        or not _known_equal(bias_meta.shape[-2], q_length)
        or not _known_equal(bias_meta.shape[-1], k_length)
    ):
        return False

    bias = _unwrap_expands(bias)
    if bias.op != "call_function" or bias.target is not aten.where.self:
        return False
    zero = _uniform_value(bias.args[1])
    neg_inf = _uniform_value(bias.args[2])
    return (
        isinstance(zero, (int, float))
        and zero == 0
        and isinstance(neg_inf, float)
        and math.isinf(neg_inf)
        and neg_inf < 0
        and _is_causal_condition(bias.args[0], q_length, k_length)
    )


def _collect_ancestors(root: torch.fx.Node, result: set[torch.fx.Node]) -> None:
    stack = [root]
    while stack:
        node = stack.pop()
        if node in result:
            continue
        result.add(node)
        stack.extend(node.all_input_nodes)


def replace_causal_bias_with_is_causal(graph: torch.fx.Graph) -> int:
    """Replace exact materialized causal biases on inference efficient SDPA."""
    dead_candidates: set[torch.fx.Node] = set()
    replacements = 0

    for node in graph.nodes:
        if (
            node.op != "call_function"
            or node.target is not aten._scaled_dot_product_efficient_attention.default
        ):
            continue

        dropout_p = get_arg_value(node, 5, "dropout_p")
        is_causal = get_arg_value(node, 6, "is_causal")
        if dropout_p not in (None, 0, 0.0) or is_causal not in (None, False):
            continue

        query = get_arg_value(node, 0, "query")
        key = get_arg_value(node, 1, "key")
        bias = get_arg_value(node, 3, "attn_bias")
        if not all(isinstance(arg, torch.fx.Node) for arg in (query, key, bias)):
            continue
        query_meta = _tensor_meta(query)
        key_meta = _tensor_meta(key)
        if (
            query_meta is None
            or key_meta is None
            or query_meta.ndim < 2
            or key_meta.ndim < 2
        ):
            continue

        q_length = query_meta.shape[-2]
        k_length = key_meta.shape[-2]
        if not _known_equal(q_length, k_length) or not _is_causal_bias(
            bias, q_length, k_length
        ):
            continue

        _collect_ancestors(bias, dead_candidates)
        node.update_arg(3, None)
        if len(node.args) > 6:
            node.update_arg(6, True)
        else:
            node.update_kwarg("is_causal", True)
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
        counters["inductor"]["causal_bias_to_is_causal"] += replacements

    return replacements
