import math
import operator
from typing import Any

import torch
from torch._dynamo.utils import counters, detect_fake_mode
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    _can_use_cuda_graph_conditional_nodes,
)
from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

from .. import config
from ..pattern_matcher import fwd_only, get_arg_value


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

_INDEX_VALUE_PASSTHROUGH_OPS = {
    aten.alias.default,
    aten.clone.default,
    aten.detach.default,
}

_CAUSAL = 1
_ALL_TRUE = 2
_INVALID = 3
_NOT_UNIFORM = object()
_UNRESOLVED = object()


def _tensor_meta(node: torch.fx.Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _resolve_scalar(value: Any) -> Any:
    if not isinstance(value, torch.fx.Node):
        return value

    meta_value = value.meta.get("val", _UNRESOLVED)
    if isinstance(
        meta_value,
        (bool, int, float, torch.SymBool, torch.SymFloat, torch.SymInt),
    ):
        return meta_value

    if value.op == "call_function" and value.target is aten.sym_size.int:
        tensor = get_arg_value(value, 0, "self")
        dim = _resolve_scalar(get_arg_value(value, 1, "dim"))
        if isinstance(tensor, torch.fx.Node) and isinstance(dim, int):
            tensor_meta = _tensor_meta(tensor)
            if tensor_meta is not None and -tensor_meta.ndim <= dim < tensor_meta.ndim:
                return tensor_meta.shape[dim]
    return _UNRESOLVED


def _known_equal(lhs: Any, rhs: Any) -> bool:
    lhs = _resolve_scalar(lhs)
    rhs = _resolve_scalar(rhs)
    if lhs is _UNRESOLVED or rhs is _UNRESOLVED:
        return False
    if isinstance(lhs, (bool, int, float)) and isinstance(
        rhs, (bool, int, float)
    ):
        return lhs == rhs
    try:
        return statically_known_true(sym_eq(lhs, rhs))
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False


def _known_positive(value: Any) -> bool:
    value = _resolve_scalar(value)
    if value is _UNRESOLVED:
        return False
    try:
        return statically_known_true(value > 0)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False


def _uniform_value(value: Any) -> Any:
    if not isinstance(value, torch.fx.Node):
        return value
    if value.op != "call_function" or value.target is not aten.full.default:
        return _NOT_UNIFORM
    return _resolve_scalar(get_arg_value(value, 1, "fill_value"))


def _unit_shape_axis(
    shape: torch.Size, length: int | torch.SymInt
) -> int | None:
    axis = None
    for dim, size in enumerate(shape):
        if _known_equal(size, 1):
            continue
        if axis is not None or not _known_equal(size, length):
            return None
        axis = dim
    return axis


def _trace_iota_axis(
    value: Any,
    memo: dict[torch.fx.Node, tuple[int | torch.SymInt, int] | None],
) -> tuple[int | torch.SymInt, int] | None:
    if not isinstance(value, torch.fx.Node) or value.op != "call_function":
        return None
    if value in memo:
        return memo[value]
    memo[value] = None

    if value.target is prims.iota.default:
        length = _resolve_scalar(get_arg_value(value, 0, "length"))
        if (
            length is not _UNRESOLVED
            and _known_equal(get_arg_value(value, 1, "start"), 0)
            and _known_equal(get_arg_value(value, 2, "step"), 1)
            and get_arg_value(value, 3, "dtype") is torch.int64
        ):
            memo[value] = (length, 0)
        return memo[value]

    source = get_arg_value(value, 0, "self")
    source_info = _trace_iota_axis(source, memo)
    source_meta = _tensor_meta(source) if isinstance(source, torch.fx.Node) else None
    value_meta = _tensor_meta(value)
    if source_info is None or source_meta is None or value_meta is None:
        return None
    length, axis = source_info

    if value.target in _INDEX_VALUE_PASSTHROUGH_OPS:
        if source_meta.ndim == value_meta.ndim:
            memo[value] = source_info
        return memo[value]

    if value.target is aten.unsqueeze.default:
        dim = _resolve_scalar(get_arg_value(value, 1, "dim"))
        if not isinstance(dim, int):
            return None
        if dim < 0:
            dim += value_meta.ndim
        if 0 <= dim < value_meta.ndim:
            memo[value] = (length, axis + int(dim <= axis))
        return memo[value]

    if value.target is aten.expand.default:
        rank_delta = value_meta.ndim - source_meta.ndim
        if rank_delta < 0:
            return None
        output_axis = axis + rank_delta
        if _known_equal(source_meta.shape[axis], length) and _known_equal(
            value_meta.shape[output_axis], length
        ):
            memo[value] = (length, output_axis)
        return memo[value]

    if value.target in (aten.reshape.default, aten.view.default):
        if _unit_shape_axis(source_meta.shape, length) != axis:
            return None
        output_axis = _unit_shape_axis(value_meta.shape, length)
        if output_axis is not None:
            memo[value] = (length, output_axis)
        return memo[value]

    return None


def _is_iota_at_axis(
    value: Any,
    length: int | torch.SymInt,
    axis: int,
    memo: dict[torch.fx.Node, tuple[int | torch.SymInt, int] | None],
) -> bool:
    if not isinstance(value, torch.fx.Node):
        return False
    meta = _tensor_meta(value)
    info = _trace_iota_axis(value, memo)
    if meta is None or info is None or meta.dtype is not torch.int64:
        return False
    if axis < 0:
        axis += meta.ndim
    return (
        0 <= axis < meta.ndim
        and info[1] == axis
        and _known_equal(info[0], length)
        and _known_equal(meta.shape[axis], length)
    )


def _is_all_true(
    value: Any,
    memo: dict[torch.fx.Node, bool],
    iota_memo: dict[torch.fx.Node, tuple[int | torch.SymInt, int] | None],
) -> bool:
    if value is True:
        return True
    if not isinstance(value, torch.fx.Node) or value.op != "call_function":
        return False
    if value in memo:
        return memo[value]
    memo[value] = False

    if value.target is aten.full.default:
        meta = _tensor_meta(value)
        result = (
            meta is not None
            and meta.dtype is torch.bool
            and _resolve_scalar(get_arg_value(value, 1, "fill_value")) is True
        )
    elif value.target in _AND_OPS:
        result = _is_all_true(
            get_arg_value(value, 0, "self"), memo, iota_memo
        ) and _is_all_true(get_arg_value(value, 1, "other"), memo, iota_memo)
    elif value.target is aten.index.Tensor:
        source = get_arg_value(value, 0, "self")
        source_meta = (
            _tensor_meta(source) if isinstance(source, torch.fx.Node) else None
        )
        indices = get_arg_value(value, 1, "indices")
        result = (
            source_meta is not None
            and isinstance(indices, (list, tuple))
            and len(indices) <= source_meta.ndim
            and _is_all_true(source, memo, iota_memo)
            and all(
                index is None
                or (
                    (info := _trace_iota_axis(index, iota_memo)) is not None
                    and _known_equal(info[0], source_meta.shape[dim])
                )
                for dim, index in enumerate(indices)
            )
        )
    elif value.target in _ALL_TRUE_PASSTHROUGH_OPS:
        result = _is_all_true(get_arg_value(value, 0, "self"), memo, iota_memo)
    else:
        result = False

    memo[value] = result
    return result


def _is_causal_comparison(
    value: Any,
    q_length: int | torch.SymInt,
    k_length: int | torch.SymInt,
    iota_memo: dict[torch.fx.Node, tuple[int | torch.SymInt, int] | None],
) -> bool:
    if (
        not isinstance(value, torch.fx.Node)
        or value.op != "call_function"
        or value.target is not aten.le.Tensor
    ):
        return False
    meta = _tensor_meta(value)
    if (
        meta is None
        or meta.ndim < 2
        or not _known_equal(meta.shape[-2], q_length)
        or not _known_equal(meta.shape[-1], k_length)
    ):
        return False
    return _is_iota_at_axis(
        get_arg_value(value, 0, "self"), k_length, -1, iota_memo
    ) and _is_iota_at_axis(
        get_arg_value(value, 1, "other"), q_length, -2, iota_memo
    )


def _classify_condition(
    value: Any,
    q_length: int | torch.SymInt,
    k_length: int | torch.SymInt,
    memo: dict[torch.fx.Node, int],
    all_true_memo: dict[torch.fx.Node, bool],
    iota_memo: dict[torch.fx.Node, tuple[int | torch.SymInt, int] | None],
) -> int:
    if not isinstance(value, torch.fx.Node) or value.op != "call_function":
        return _ALL_TRUE if value is True else _INVALID
    if value in memo:
        return memo[value]
    memo[value] = _INVALID

    if _is_causal_comparison(value, q_length, k_length, iota_memo):
        result = _CAUSAL
    elif value.target in _AND_OPS:
        lhs = _classify_condition(
            get_arg_value(value, 0, "self"),
            q_length,
            k_length,
            memo,
            all_true_memo,
            iota_memo,
        )
        rhs = _classify_condition(
            get_arg_value(value, 1, "other"),
            q_length,
            k_length,
            memo,
            all_true_memo,
            iota_memo,
        )
        if lhs == _INVALID or rhs == _INVALID:
            result = _INVALID
        elif lhs == _CAUSAL or rhs == _CAUSAL:
            result = _CAUSAL
        else:
            result = _ALL_TRUE
    elif (
        value.target in _INDEX_VALUE_PASSTHROUGH_OPS
        or value.target is aten.expand.default
    ):
        result = _classify_condition(
            get_arg_value(value, 0, "self"),
            q_length,
            k_length,
            memo,
            all_true_memo,
            iota_memo,
        )
    elif _is_all_true(value, all_true_memo, iota_memo):
        result = _ALL_TRUE
    else:
        result = _INVALID

    memo[value] = result
    return result


def _is_causal_bias(
    bias: Any, q_length: int | torch.SymInt, k_length: int | torch.SymInt
) -> bool:
    if not isinstance(bias, torch.fx.Node):
        return False
    bias_meta = _tensor_meta(bias)
    if (
        bias_meta is None
        or bias_meta.ndim != 4
        or not _known_equal(bias_meta.shape[-2], q_length)
        or not _known_equal(bias_meta.shape[-1], k_length)
    ):
        return False

    while bias.op == "call_function" and bias.target is aten.expand.default:
        bias = get_arg_value(bias, 0, "self")
        if not isinstance(bias, torch.fx.Node):
            return False
    if bias.op != "call_function" or bias.target is not aten.where.self:
        return False

    zero = _uniform_value(get_arg_value(bias, 1, "self"))
    neg_inf = _uniform_value(get_arg_value(bias, 2, "other"))
    if not (
        isinstance(zero, (int, float))
        and zero == 0
        and isinstance(neg_inf, float)
        and math.isinf(neg_inf)
        and neg_inf < 0
    ):
        return False

    try:
        return (
            _classify_condition(
                get_arg_value(bias, 0, "condition"),
                q_length,
                k_length,
                {},
                {},
                {},
            )
            == _CAUSAL
        )
    except RecursionError:
        return False


def _same_shape(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    return lhs.ndim == rhs.ndim and all(
        _known_equal(left, right) for left, right in zip(lhs.shape, rhs.shape)
    )


def _collect_ancestors(root: torch.fx.Node, result: set[torch.fx.Node]) -> None:
    stack = [root]
    while stack:
        node = stack.pop()
        if node in result:
            continue
        result.add(node)
        stack.extend(node.all_input_nodes)


def _make_attention_branches(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    compute_log_sumexp: bool,
    scale: float | None,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule] | None:
    op = aten._scaled_dot_product_efficient_attention.default

    def causal(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Any:
        return (op(q, k, v, None, compute_log_sumexp, 0.0, True, scale=scale)[0],)

    def additive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Any:
        batch, heads, q_length, _ = q.shape
        k_length = k.shape[-2]
        bias = torch.full(
            (q_length, k_length),
            -float("inf"),
            dtype=q.dtype,
            device=q.device,
        ).triu(diagonal=1)
        bias = bias.expand(batch, heads, q_length, k_length)
        return (
            op(q, k, v, bias, compute_log_sumexp, 0.0, False, scale=scale)[0],
        )

    fake_mode = detect_fake_mode((query, key, value))
    if fake_mode is None:
        return None
    try:
        with fake_mode:
            args = (query, key, value)
            return (
                fwd_only(causal, args, run_functional_passes=False),
                fwd_only(additive, args, run_functional_passes=False),
            )
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None


def _set_meta(node: torch.fx.Node, value: Any) -> torch.fx.Node:
    node.meta["val"] = value
    return node


def _insert_guarded_attention(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    query: torch.fx.Node,
    key: torch.fx.Node,
    value: torch.fx.Node,
    output: torch.fx.Node,
    threshold: float,
    branch_names: tuple[str, str],
) -> torch.fx.Node:
    graph = gm.graph
    query_meta = _tensor_meta(query)
    key_meta = _tensor_meta(key)
    value_meta = _tensor_meta(value)
    if query_meta is None or key_meta is None or value_meta is None:
        raise AssertionError("attention inputs must have tensor metadata")

    causal_name, additive_name = branch_names
    fake_mode = detect_fake_mode((query_meta, key_meta))
    if fake_mode is None:
        raise AssertionError("attention inputs must use fake tensors")

    with graph.inserting_before(node), fake_mode:
        # Infinity norms preserve NaN and Inf while bounding all three operands.
        norm_values = aten._foreach_norm.Scalar(
            (query_meta, key_meta, value_meta), float("inf")
        )
        norms = graph.call_function(
            aten._foreach_norm.Scalar,
            ((query, key, value), float("inf")),
        )
        norms.meta["val"] = norm_values
        query_norm = _set_meta(
            graph.call_function(operator.getitem, (norms, 0)), norm_values[0]
        )
        key_norm = _set_meta(
            graph.call_function(operator.getitem, (norms, 1)), norm_values[1]
        )
        value_norm = _set_meta(
            graph.call_function(operator.getitem, (norms, 2)), norm_values[2]
        )
        qk_max = _set_meta(
            graph.call_function(aten.maximum.default, (query_norm, key_norm)),
            aten.maximum.default(query_norm.meta["val"], key_norm.meta["val"]),
        )
        qkv_max = _set_meta(
            graph.call_function(aten.maximum.default, (qk_max, value_norm)),
            aten.maximum.default(qk_max.meta["val"], value_norm.meta["val"]),
        )
        safe = _set_meta(
            graph.call_function(aten.le.Scalar, (qkv_max, threshold)),
            aten.le.Scalar(qkv_max.meta["val"], threshold),
        )
        causal = graph.get_attr(causal_name)
        additive = graph.get_attr(additive_name)
        conditional = graph.call_function(
            torch.ops.higher_order.cond,
            (safe, causal, additive, (query, key, value)),
        )
        conditional.meta["val"] = (output.meta["val"],)
        conditional.meta["inductor_cudagraphable_cond"] = True
        guarded = graph.call_function(operator.getitem, (conditional, 0))
        guarded.meta.update(output.meta)
    return guarded


def replace_causal_bias_with_is_causal(gm: torch.fx.GraphModule) -> int:
    """Replace exact materialized causal biases on inference efficient SDPA."""
    if not (
        _can_use_cuda_graph_conditional_nodes()
        and config.graph_partition
        and config.triton.cudagraphs
        and config.triton.cudagraph_trees
        and config.cudagraph_policy is None
        and config.implicit_fallbacks
        and not config.cpp_wrapper
        and not config.fx_wrapper
    ):
        return 0

    dead_candidates: set[torch.fx.Node] = set()
    branch_cache: dict[tuple[Any, ...], tuple[str, str]] = {}
    replacements = 0

    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not aten._scaled_dot_product_efficient_attention.default
        ):
            continue

        output_users = [
            user
            for user in node.users
            if user.op == "call_function"
            and user.target is operator.getitem
            and user.args[1] == 0
        ]
        if len(output_users) != len(node.users) or not output_users:
            continue
        output = output_users[0]
        if _tensor_meta(output) is None:
            continue

        dropout_p = _resolve_scalar(get_arg_value(node, 5, "dropout_p"))
        is_causal = _resolve_scalar(get_arg_value(node, 6, "is_causal"))
        compute_log_sumexp = _resolve_scalar(
            get_arg_value(node, 4, "compute_log_sumexp")
        )
        scale = _resolve_scalar(node.kwargs.get("scale"))
        if (
            not (dropout_p is None or _known_equal(dropout_p, 0))
            or not (is_causal is None or _known_equal(is_causal, False))
            or not isinstance(compute_log_sumexp, bool)
            or scale is _UNRESOLVED
        ):
            continue
        attention_scale = None
        if scale is not None:
            if not isinstance(scale, (int, float)) or isinstance(scale, bool):
                continue
            try:
                attention_scale = float(scale)
            except OverflowError:
                continue
            if (
                not math.isfinite(attention_scale)
                or attention_scale <= 0
                or attention_scale > torch.finfo(torch.float32).max
            ):
                continue

        query = get_arg_value(node, 0, "query")
        key = get_arg_value(node, 1, "key")
        value = get_arg_value(node, 2, "value")
        bias = get_arg_value(node, 3, "attn_bias")
        if (
            not isinstance(query, torch.fx.Node)
            or not isinstance(key, torch.fx.Node)
            or not isinstance(value, torch.fx.Node)
            or not isinstance(bias, torch.fx.Node)
        ):
            continue
        query_meta = _tensor_meta(query)
        key_meta = _tensor_meta(key)
        value_meta = _tensor_meta(value)
        bias_meta = _tensor_meta(bias)
        if (
            query_meta is None
            or key_meta is None
            or value_meta is None
            or bias_meta is None
        ):
            continue
        if any(
            not isinstance(dim, int)
            for meta in (query_meta, key_meta, value_meta)
            for dim in (*meta.shape, *meta.stride())
        ):
            continue
        if (
            query_meta.ndim != 4
            or not _same_shape(query_meta, key_meta)
            or not _same_shape(query_meta, value_meta)
            or query_meta.dtype
            not in (torch.float16, torch.bfloat16, torch.float32)
            or key_meta.dtype is not query_meta.dtype
            or value_meta.dtype is not query_meta.dtype
            or bias_meta.dtype is not query_meta.dtype
            or query_meta.device.type != "cuda"
            or key_meta.device != query_meta.device
            or value_meta.device != query_meta.device
            or bias_meta.device != query_meta.device
        ):
            continue

        q_length = query_meta.shape[-2]
        k_length = key_meta.shape[-2]
        head_dim = query_meta.shape[-1]
        if (
            not isinstance(head_dim, int)
            or not all(_known_positive(dim) for dim in query_meta.shape)
            or not _known_equal(q_length, k_length)
            or not all(
                _known_equal(actual, expected)
                for actual, expected in zip(
                    bias_meta.shape,
                    (
                        query_meta.shape[0],
                        query_meta.shape[1],
                        q_length,
                        k_length,
                    ),
                )
            )
            or not _is_causal_bias(bias, q_length, k_length)
        ):
            continue

        effective_scale = head_dim**-0.5 if attention_scale is None else attention_scale
        accumulation_max = min(
            torch.finfo(torch.float32).max, torch.finfo(query_meta.dtype).max
        )
        # Additive -inf can produce NaN where causal masking skips exceptional
        # scores or values. Use is_causal only when every intermediate is finite.
        threshold = math.sqrt(
            accumulation_max / (4.0 * head_dim * max(1.0, effective_scale))
        )
        threshold = min(threshold, torch.finfo(query_meta.dtype).max / 2.0)
        input_aliases = (
            query_meta is key_meta,
            query_meta is value_meta,
            key_meta is value_meta,
        )
        branch_key = (
            *(
                (meta.dtype, meta.device, tuple(meta.shape), tuple(meta.stride()))
                for meta in (query_meta, key_meta, value_meta)
            ),
            input_aliases,
            compute_log_sumexp,
            attention_scale,
        )
        branch_names = branch_cache.get(branch_key)
        if branch_names is None:
            branches = _make_attention_branches(
                query_meta,
                key_meta,
                value_meta,
                compute_log_sumexp,
                attention_scale,
            )
            if branches is None:
                continue

            branch_index = len(branch_cache)
            causal_name = f"_causal_attention_{branch_index}_causal"
            additive_name = f"_causal_attention_{branch_index}_additive"
            while hasattr(gm, causal_name) or hasattr(gm, additive_name):
                branch_index += 1
                causal_name = f"_causal_attention_{branch_index}_causal"
                additive_name = f"_causal_attention_{branch_index}_additive"
            gm.add_module(causal_name, branches[0])
            gm.add_module(additive_name, branches[1])
            branch_names = (causal_name, additive_name)
            branch_cache[branch_key] = branch_names

        _collect_ancestors(bias, dead_candidates)
        guarded = _insert_guarded_attention(
            gm,
            node,
            query,
            key,
            value,
            output,
            threshold,
            branch_names,
        )
        for output_user in output_users:
            output_user.replace_all_uses_with(guarded)
            gm.graph.erase_node(output_user)
        gm.graph.erase_node(node)
        replacements += 1

    if replacements:
        for node in reversed(gm.graph.nodes):
            if (
                node in dead_candidates
                and not node.users
                and node.op not in ("placeholder", "output")
                and not node.is_impure()
            ):
                gm.graph.erase_node(node)
        counters["inductor"]["causal_bias_to_is_causal"] += replacements

    return replacements
