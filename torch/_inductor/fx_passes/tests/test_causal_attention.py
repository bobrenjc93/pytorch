# Owner(s): ["module: inductor"]

import unittest
from unittest.mock import patch

import torch
from torch._dynamo.utils import counters
from torch._higher_order_ops.cudagraph_conditional_nodes import (
    _can_use_cuda_graph_conditional_nodes,
    _has_cuda_graph_conditional_node_support,
)
from torch._inductor.cudagraph_trees import CUDAGraphNode, get_container
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.fx_passes.causal_attention import (
    replace_causal_bias_with_is_causal,
)
from torch._inductor.pattern_matcher import fwd_only, get_arg_value
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


aten = torch.ops.aten
prims = torch.ops.prims


@torch._dynamo.dont_skip_tracing
def _attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    padding: torch.Tensor | None = None,
    *,
    indexed_padding: bool = False,
    wrong_axis: bool = False,
    diagonal_offset: int = 0,
    and_depth: int = 0,
    scale: float | None = None,
) -> torch.Tensor:
    batch, heads, q_length, _ = query.shape
    k_length = key.shape[-2]
    q_index = torch.arange(q_length, device=query.device).view(
        1, 1, q_length, 1
    )
    if wrong_axis:
        k_index = torch.arange(batch, device=query.device).view(batch, 1, 1, 1)
        k_index = k_index.expand(batch, 1, q_length, k_length)
    else:
        k_index = torch.arange(k_length, device=query.device).view(
            1, 1, 1, k_length
        )
    if diagonal_offset:
        q_index = q_index + diagonal_offset
    condition = k_index <= q_index

    if indexed_padding:
        all_true = torch.full(
            (batch, k_length), True, dtype=torch.bool, device=query.device
        )
        batch_index = torch.arange(batch, device=query.device).view(batch, 1)
        batch_index = batch_index.expand(batch, k_length)
        token_index = torch.arange(k_length, device=query.device).view(1, k_length)
        token_index = token_index.expand(batch, k_length)
        padding = all_true[batch_index, token_index]
    if padding is not None:
        condition = condition & padding.view(batch, 1, 1, k_length)
    for _ in range(and_depth):
        condition = condition & condition

    condition = condition.expand(batch, 1, q_length, k_length)
    zero = torch.full((), 0.0, dtype=query.dtype, device=query.device)
    neg_inf = torch.full((), -float("inf"), dtype=query.dtype, device=query.device)
    bias = torch.where(condition, zero, neg_inf).expand(
        batch, heads, q_length, k_length
    )
    return aten._scaled_dot_product_efficient_attention.default(
        query, key, value, bias, False, 0.0, False, scale=scale
    )[0]


def _trace_attention(
    shape: tuple[int, ...] = (2, 4, 8, 16),
    *,
    key_length: int | None = None,
    padding: bool = False,
    indexed_padding: bool = False,
    wrong_axis: bool = False,
    diagonal_offset: int = 0,
    and_depth: int = 0,
    scale: float | None = None,
) -> torch.fx.GraphModule:
    def fn(*args: torch.Tensor) -> torch.Tensor:
        return _attention(
            *args,
            indexed_padding=indexed_padding,
            wrong_axis=wrong_axis,
            diagonal_offset=diagonal_offset,
            and_depth=and_depth,
            scale=scale,
        )

    fake_mode = FakeTensorMode()
    with fake_mode:
        kv_length = shape[2] if key_length is None else key_length
        kv_shape = (*shape[:2], kv_length, shape[3])
        inputs = [torch.empty(shape, device="cuda", dtype=torch.bfloat16)]
        inputs.extend(
            torch.empty(kv_shape, device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )
        if padding:
            inputs.append(
                torch.empty(
                    (shape[0], kv_shape[2]), device="cuda", dtype=torch.bool
                )
            )
        gm = fwd_only(fn, inputs, run_functional_passes=False)
    gm.graph.eliminate_dead_code()
    gm.recompile()
    return gm


def _target_count(gm: torch.fx.GraphModule, target: object) -> int:
    return sum(node.target is target for node in gm.graph.nodes)


class CausalAttentionPassTests(TestCase):
    def setUp(self):
        super().setUp()
        config_context = torch._inductor.config.patch(
            {"triton.cudagraphs": True}
        )
        config_context.__enter__()
        self.addCleanup(config_context.__exit__, None, None, None)
        capture_context = patch(
            "torch._inductor.fx_passes.causal_attention."
            "_can_use_cuda_graph_conditional_nodes",
            return_value=True,
        )
        capture_context.start()
        self.addCleanup(capture_context.stop)

    def test_exact_causal_bias(self):
        gm = _trace_attention()

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 1)
        gm.graph.lint()
        self.assertEqual(_target_count(gm, torch.ops.higher_order.cond), 1)
        self.assertEqual(_target_count(gm, aten.where.self), 0)
        self.assertEqual(_target_count(gm, prims.iota.default), 0)

        conditional = next(
            node
            for node in gm.graph.nodes
            if node.target is torch.ops.higher_order.cond
        )
        self.assertTrue(conditional.meta["inductor_cudagraphable_cond"])
        causal = getattr(gm, conditional.args[1].target)
        attention = next(
            node
            for node in causal.graph.nodes
            if node.target is aten._scaled_dot_product_efficient_attention.default
        )
        self.assertIsNone(get_arg_value(attention, 3, "attn_bias"))
        self.assertTrue(get_arg_value(attention, 6, "is_causal"))

    def test_compile_time_all_true_padding(self):
        gm = _trace_attention(indexed_padding=True)

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 1)
        gm.graph.lint()

    def test_rejects_wrong_iota_axis(self):
        gm = _trace_attention(shape=(8, 2, 8, 16), wrong_axis=True)

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_rejects_diagonal_offset(self):
        gm = _trace_attention(diagonal_offset=1)

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_rejects_rectangular_attention(self):
        gm = _trace_attention(key_length=6)

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_rejects_dropout(self):
        gm = _trace_attention()
        attention = next(
            node
            for node in gm.graph.nodes
            if node.target is aten._scaled_dot_product_efficient_attention.default
        )
        attention.kwargs = {"dropout_p": 0.1}
        gm.graph.lint()

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_rejects_runtime_padding(self):
        gm = _trace_attention(padding=True)

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_rejects_unsupported_conditional_capture(self):
        gm = _trace_attention()

        with patch(
            "torch._inductor.fx_passes.causal_attention."
            "_can_use_cuda_graph_conditional_nodes",
            return_value=False,
        ):
            self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_requires_cudagraphs(self):
        gm = _trace_attention()

        with torch._inductor.config.patch({"triton.cudagraphs": False}):
            self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_requires_implicit_fallbacks(self):
        gm = _trace_attention()

        with torch._inductor.config.patch({"implicit_fallbacks": False}):
            self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_rejects_float32_overflowing_scale(self):
        gm = _trace_attention(scale=1e39)

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_conditional_capture_availability(self):
        with patch.object(torch.version, "cuda", None):
            self.assertFalse(_can_use_cuda_graph_conditional_nodes())
        with patch.object(torch.version, "cuda", "12.3"):
            self.assertFalse(_can_use_cuda_graph_conditional_nodes())
        driver_module = (
            "torch._higher_order_ops.cudagraph_conditional_nodes."
            "_get_cuda_library"
        )
        driver_check = (
            "torch._higher_order_ops.cudagraph_conditional_nodes."
            "_check_cuda"
        )
        _has_cuda_graph_conditional_node_support.cache_clear()
        try:
            with patch.object(torch.version, "cuda", "12.5"), patch(
                driver_module
            ) as get_driver, patch(driver_check) as check_driver, patch.object(
                torch._C,
                "_accelerator_getAllocatorSettings",
                return_value="",
            ):
                def return_old_driver_version(output):
                    output._obj.value = 12030
                    return 0

                driver = get_driver.return_value
                driver.cuDriverGetVersion.side_effect = return_old_driver_version
                self.assertFalse(_can_use_cuda_graph_conditional_nodes())
                driver.cuDriverGetVersion.assert_called_once()
                check_driver.assert_called_once_with(0)
        finally:
            _has_cuda_graph_conditional_node_support.cache_clear()
        with patch.object(
            torch._C,
            "_accelerator_getAllocatorSettings",
            return_value="graph_capture_record_stream_reuse:True",
        ):
            self.assertFalse(_can_use_cuda_graph_conditional_nodes())

    def test_shared_and_dag(self):
        gm = _trace_attention(and_depth=32)

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 1)
        gm.graph.lint()

    def test_keyword_attention_call(self):
        gm = _trace_attention()
        attention = next(
            node
            for node in gm.graph.nodes
            if node.target is aten._scaled_dot_product_efficient_attention.default
        )
        query, key, value, bias, compute_log_sumexp = attention.args
        attention.args = ()
        attention.kwargs = {
            "query": query,
            "key": key,
            "value": value,
            "attn_bias": bias,
            "compute_log_sumexp": compute_log_sumexp,
            "dropout_p": 0.0,
            "is_causal": False,
        }
        gm.graph.lint()

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 1)
        gm.graph.lint()

    def test_keyword_bias_call(self):
        gm = _trace_attention()
        where = next(
            node for node in gm.graph.nodes if node.target is aten.where.self
        )
        condition, zero, neg_inf = where.args
        where.args = ()
        where.kwargs = {
            "condition": condition,
            "self": zero,
            "other": neg_inf,
        }
        gm.graph.lint()

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 1)
        gm.graph.lint()

    def test_symbolic_iota_length(self):
        def fn(query, key, value):
            return _attention(query, key, value)

        shape_env = ShapeEnv()
        fake_mode = FakeTensorMode(shape_env=shape_env)
        base = fake_mode.from_tensor(
            torch.empty((1, 2, 8, 16), dtype=torch.bfloat16),
            static_shapes=False,
        )
        with torch._C._DisableTorchDispatch():
            metas = tuple(
                torch.empty_strided(
                    tuple(base.shape),
                    tuple(base.stride()),
                    dtype=base.dtype,
                    device="meta",
                )
                for _ in range(3)
            )
        inputs = tuple(
            fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, meta, torch.device("cuda")
            )
            for meta in metas
        )
        gm = make_fx(fn, select_decomp_table(), tracing_mode="symbolic")(*inputs)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        self.assertTrue(
            any(
                node.target is prims.iota.default
                and isinstance(node.args[0], torch.fx.Node)
                for node in gm.graph.nodes
            )
        )

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 0)
        gm.graph.lint()

    def test_reuses_attention_branches(self):
        def twice(*args: torch.Tensor) -> torch.Tensor:
            return _attention(*args) + _attention(*args)

        fake_mode = FakeTensorMode()
        with fake_mode:
            inputs = tuple(
                torch.empty(
                    (2, 4, 8, 16), device="cuda", dtype=torch.bfloat16
                )
                for _ in range(3)
            )
            gm = fwd_only(twice, inputs, run_functional_passes=False)
        gm.graph.eliminate_dead_code()
        gm.recompile()

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 2)
        self.assertEqual(len(dict(gm.named_children())), 2)
        gm.graph.lint()

    def test_does_not_reuse_branches_across_input_aliasing(self):
        def mixed(query, key, value):
            return _attention(query, query, query) + _attention(query, key, value)

        fake_mode = FakeTensorMode()
        with fake_mode:
            inputs = tuple(
                torch.empty(
                    (2, 4, 8, 16), device="cuda", dtype=torch.bfloat16
                )
                for _ in range(3)
            )
            gm = fwd_only(mixed, inputs, run_functional_passes=False)
        gm.graph.eliminate_dead_code()
        gm.recompile()

        self.assertEqual(replace_causal_bias_with_is_causal(gm), 2)
        self.assertEqual(len(dict(gm.named_children())), 4)
        gm.graph.lint()


@unittest.skipUnless(
    _can_use_cuda_graph_conditional_nodes(),
    "CUDA 12.4 or greater is required for CUDA graph conditional nodes",
)
class CausalAttentionNumericsTests(TestCase):
    def setUp(self):
        super().setUp()
        config_context = torch._inductor.config.patch(
            {"triton.cudagraphs": True}
        )
        config_context.__enter__()
        self.addCleanup(config_context.__exit__, None, None, None)

    @parametrize("case", ("score_overflow", "nonfinite_value"))
    @parametrize("dtype", (torch.bfloat16, torch.float32))
    def test_special_values(self, device, case, dtype):
        shape = (2, 2, 128, 64)
        query = torch.zeros(shape, device=device, dtype=dtype)
        key = torch.zeros_like(query)
        value = torch.randn_like(query)
        if case == "score_overflow":
            limit = torch.finfo(query.dtype).max
            query[0, :, 0] = limit
            key[0, :, 1:] = limit
        else:
            query.normal_()
            key.normal_()
            value[0, :, -1] = float("inf")

        torch._dynamo.reset()
        counters["inductor"].clear()
        compiled = torch.compile(_attention, fullgraph=True)
        with torch.no_grad():
            expected = _attention(query, key, value)
            actual = compiled(query, key, value)

        self.assertTrue(torch.isnan(expected).any())
        self.assertEqual(actual, expected, equal_nan=True)
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 1)

    def test_cudagraph_switches_to_exceptional_path(self, device):
        shape = (1, 1, 128, 64)
        safe_inputs = tuple(
            torch.randn(shape, device=device, dtype=torch.bfloat16)
            for _ in range(3)
        )
        exceptional_inputs = tuple(tensor.clone() for tensor in safe_inputs)
        exceptional_inputs[0].zero_()
        exceptional_inputs[1].zero_()
        limit = torch.finfo(torch.bfloat16).max
        exceptional_inputs[0][:, :, 0] = limit
        exceptional_inputs[1][:, :, 1:] = limit

        torch._dynamo.reset()
        counters["inductor"].clear()
        compiled = torch.compile(_attention, fullgraph=True)
        with torch.no_grad():
            inputs_to_test = (
                safe_inputs,
                safe_inputs,
                exceptional_inputs,
                safe_inputs,
            )
            for inputs in inputs_to_test:
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(
                    compiled(*inputs), _attention(*inputs), equal_nan=True
                )

        self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
        manager = get_container(torch.cuda.current_device()).tree_manager
        self.assertIsNotNone(manager)
        self.assertIsInstance(manager.current_node, CUDAGraphNode)
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 1)

    def test_allocator_change_after_cudagraph_warmup(self, device):
        inputs = tuple(
            torch.randn(
                (1, 1, 128, 64), device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )

        torch._dynamo.reset()
        counters["inductor"].clear()
        compiled = torch.compile(_attention, fullgraph=True)
        original_settings = torch._C._accelerator_getAllocatorSettings()
        setting_name = "graph_capture_record_stream_reuse"
        restore_settings = ",".join(
            setting
            for setting in original_settings.split(",")
            if not setting.strip().startswith(f"{setting_name}:")
        )
        restore_settings = ",".join(
            setting for setting in (restore_settings, f"{setting_name}:False") if setting
        )
        try:
            with torch.no_grad():
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*inputs), _attention(*inputs))
                torch._C._accelerator_setAllocatorSettings(
                    f"{setting_name}:True"
                )
                torch.compiler.cudagraph_mark_step_begin()
                self.assertEqual(compiled(*inputs), _attention(*inputs))
        finally:
            torch._C._accelerator_setAllocatorSettings(restore_settings)

        with torch.no_grad():
            torch.compiler.cudagraph_mark_step_begin()
            self.assertEqual(compiled(*inputs), _attention(*inputs))
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 1)
        self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

    def test_dynamic_shapes(self, device):
        torch._dynamo.reset()
        counters["inductor"].clear()
        compiled = torch.compile(_attention, fullgraph=True, dynamic=True)
        for length in (96, 128):
            inputs = tuple(
                torch.randn(
                    (1, 1, length, 64), device=device, dtype=torch.bfloat16
                )
                for _ in range(3)
            )
            with torch.no_grad():
                self.assertEqual(compiled(*inputs), _attention(*inputs))
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 0)

    def test_partially_dynamic_sequence_length(self, device):
        query = torch.randn(
            (1, 1, 96, 64), device=device, dtype=torch.bfloat16
        )
        torch._dynamo.mark_dynamic(query, 2)

        torch._dynamo.reset()
        counters["inductor"].clear()
        compiled = torch.compile(_attention, fullgraph=True)
        with torch.no_grad():
            self.assertEqual(
                compiled(query, query, query),
                _attention(query, query, query),
            )
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 0)

    def test_implicit_fallbacks_disabled(self, device):
        inputs = tuple(
            torch.randn(
                (1, 1, 96, 64), device=device, dtype=torch.bfloat16
            )
            for _ in range(3)
        )

        torch._dynamo.reset()
        counters["inductor"].clear()
        with torch._inductor.config.patch({"implicit_fallbacks": False}):
            compiled = torch.compile(_attention, fullgraph=True)
            with torch.no_grad():
                self.assertEqual(compiled(*inputs), _attention(*inputs))
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 0)

    def test_float32_overflowing_scale(self, device):
        shape = (1, 1, 96, 64)
        query = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        key = torch.zeros_like(query)
        value = torch.randn_like(query)

        torch._dynamo.reset()
        counters["inductor"].clear()
        compiled = torch.compile(_attention, fullgraph=True)
        with torch.no_grad():
            expected = _attention(query, key, value, scale=1e39)
            actual = compiled(query, key, value, scale=1e39)
        self.assertTrue(torch.isnan(expected).any())
        self.assertEqual(actual, expected, equal_nan=True)
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 0)

    def test_training_does_not_rewrite(self, device):
        inputs = tuple(
            torch.randn(
                (1, 1, 128, 64),
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for _ in range(3)
        )
        torch._dynamo.reset()
        counters["inductor"].clear()
        compiled = torch.compile(_attention, fullgraph=True)

        self.assertEqual(compiled(*inputs), _attention(*inputs))
        self.assertEqual(counters["inductor"]["causal_bias_to_is_causal"], 0)


instantiate_device_type_tests(
    CausalAttentionNumericsTests, globals(), only_for="cuda"
)


if __name__ == "__main__":
    run_tests()
