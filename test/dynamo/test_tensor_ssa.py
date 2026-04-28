# Owner(s): ["module: dynamo"]

import operator
from unittest import mock

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import EagerAndRecordGraphs


class TensorSSATests(torch._dynamo.test_case.TestCase):
    def test_straight_line_tensor_ops_skip_generic_fake_value(self):
        def fn(a, b):
            result = a.clone()
            result = result + b
            result = result + 8 * b
            result = result.sin()
            return result

        backend = EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        a = torch.ones(10)
        b = torch.randn(10)

        with mock.patch(
            "torch._dynamo.variables.builder.get_fake_value",
            side_effect=AssertionError("generic fake-value path used"),
        ):
            self.assertEqual(opt_fn(a, b), fn(a, b))

        targets = [
            node.target
            for node in backend.graphs[0].graph.nodes
            if node.op in ("call_function", "call_method")
        ]
        self.assertEqual(
            targets, ["clone", operator.add, operator.mul, operator.add, "sin"]
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
