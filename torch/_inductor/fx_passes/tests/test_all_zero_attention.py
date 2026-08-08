# Owner(s): ["module: inductor"]

import operator

import torch
from torch import fx
from torch._inductor.fx_passes.all_zero_attention import (
    remove_all_zero_sdpa_biases,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


aten = torch.ops.aten
prims = torch.ops.prims


def make_graph(
    *,
    argument_form="positional",
    threshold=0,
    is_causal=False,
    bias_shape=(2, 4, 8, 8),
    bias_dtype=torch.float32,
    share_predicate=False,
):
    graph = fx.Graph()
    query = graph.placeholder("query")
    key = graph.placeholder("key")
    value = graph.placeholder("value")
    iota = graph.call_function(
        prims.iota.default,
        (8,),
        {
            "start": 0,
            "step": 1,
            "dtype": torch.int64,
            "device": torch.device("cuda"),
            "requires_grad": False,
        },
    )
    indices = graph.call_function(aten.unsqueeze.default, (iota, 0))
    predicate = graph.call_function(aten.ge.Scalar, (indices, threshold))
    predicate = graph.call_function(aten.expand.default, (predicate, [2, 1, 8, 8]))
    zero = graph.call_function(
        aten.full.default,
        ([], 0.0),
        {"dtype": torch.float32, "device": torch.device("cuda")},
    )
    neg_inf = graph.call_function(
        aten.full.default,
        ([], -float("inf")),
        {"dtype": torch.float32, "device": torch.device("cuda")},
    )
    bias = graph.call_function(aten.where.self, (predicate, zero, neg_inf))
    bias = graph.call_function(aten.expand.default, (bias, [2, 4, 8, 8]))

    query.meta["val"] = torch.empty((2, 4, 8, 16), device="meta")
    key.meta["val"] = torch.empty((2, 4, 8, 16), device="meta")
    bias.meta["val"] = torch.empty(bias_shape, dtype=bias_dtype, device="meta")

    if argument_form == "positional":
        args = (query, key, value, bias, False, 0.0, is_causal)
        kwargs = {}
    else:
        args = (query, key, value)
        kwargs = {
            "attn_bias": bias,
            "compute_log_sumexp": False,
            "dropout_p": 0.0,
            "is_causal": is_causal,
        }
    attention = graph.call_function(
        aten._scaled_dot_product_efficient_attention.default, args, kwargs
    )
    result = graph.call_function(operator.getitem, (attention, 0))
    graph.output((result, predicate) if share_predicate else result)
    return fx.GraphModule({}, graph), attention, bias, predicate, iota


class TestAllZeroAttention(TestCase):
    @parametrize("argument_form", ("positional", "keyword"))
    def test_removes_bias(self, argument_form):
        gm, attention, _, _, iota = make_graph(argument_form=argument_form)

        self.assertEqual(remove_all_zero_sdpa_biases(gm.graph), 1)
        if argument_form == "positional":
            self.assertIsNone(attention.args[3])
            self.assertFalse(attention.args[6])
        else:
            self.assertIsNone(attention.kwargs["attn_bias"])
            self.assertFalse(attention.kwargs["is_causal"])
        self.assertNotIn(iota, gm.graph.nodes)
        gm.graph.lint()

    def test_non_all_true_predicate(self):
        gm, attention, bias, _, _ = make_graph(threshold=1)

        self.assertEqual(remove_all_zero_sdpa_biases(gm.graph), 0)
        self.assertIs(attention.args[3], bias)
        gm.graph.lint()

    @parametrize(
        "metadata",
        (
            {"bias_shape": (8,)},
            {"bias_shape": (1, 4, 8, 8)},
            {"bias_dtype": torch.float16},
        ),
    )
    def test_metadata_mismatch(self, metadata):
        gm, attention, bias, _, _ = make_graph(**metadata)

        self.assertEqual(remove_all_zero_sdpa_biases(gm.graph), 0)
        self.assertIs(attention.args[3], bias)
        gm.graph.lint()

    def test_causal_mode(self):
        gm, attention, bias, _, _ = make_graph(is_causal=True)

        self.assertEqual(remove_all_zero_sdpa_biases(gm.graph), 0)
        self.assertIs(attention.args[3], bias)
        gm.graph.lint()

    def test_shared_ancestor_is_preserved(self):
        gm, attention, bias, predicate, iota = make_graph(share_predicate=True)

        self.assertEqual(remove_all_zero_sdpa_biases(gm.graph), 1)
        self.assertIsNone(attention.args[3])
        self.assertNotIn(bias, gm.graph.nodes)
        self.assertIn(predicate, gm.graph.nodes)
        self.assertIn(iota, gm.graph.nodes)
        gm.graph.lint()


instantiate_parametrized_tests(TestAllZeroAttention)


if __name__ == "__main__":
    run_tests()
