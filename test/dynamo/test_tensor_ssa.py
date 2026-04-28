# Owner(s): ["module: dynamo"]

import contextlib
import operator
from unittest import mock

import torch
import torch._dynamo.test_case
from torch._dynamo import tensor_ssa
from torch._dynamo.testing import EagerAndRecordGraphs


class TensorSSATests(torch._dynamo.test_case.TestCase):
    @contextlib.contextmanager
    def collect_fastpath_hits(self):
        hits = []
        orig_stack_op = tensor_ssa.maybe_fastpath_tensor_stack_op
        orig_method = tensor_ssa.maybe_fastpath_tensor_method

        def stack_op_wrapper(tx, fn, args):
            result = orig_stack_op(tx, fn, args)
            if result is not None:
                hits.append(("stack", fn))
            return result

        def method_wrapper(tx, tensor, name, args, kwargs):
            result = orig_method(tx, tensor, name, args, kwargs)
            if result is not None:
                hits.append(("method", name))
            return result

        with (
            mock.patch.object(
                tensor_ssa,
                "maybe_fastpath_tensor_stack_op",
                side_effect=stack_op_wrapper,
            ),
            mock.patch.object(
                tensor_ssa,
                "maybe_fastpath_tensor_method",
                side_effect=method_wrapper,
            ),
        ):
            yield hits

    def test_straight_line_tensor_ops_use_fastpath(self):
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

        with self.collect_fastpath_hits() as hits:
            self.assertEqual(opt_fn(a, b), fn(a, b))

        self.assertEqual(
            hits,
            [
                ("method", "clone"),
                ("stack", operator.add),
                ("stack", operator.mul),
                ("stack", operator.add),
                ("method", "sin"),
            ],
        )

        targets = [
            node.target
            for node in backend.graphs[0].graph.nodes
            if node.op in ("call_function", "call_method")
        ]
        self.assertEqual(
            targets, ["clone", operator.add, operator.mul, operator.add, "sin"]
        )

    def test_config_flag_disables_fastpath(self):
        def fn(a, b):
            return (a + b).sin()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        a = torch.randn(4)
        b = torch.randn(4)

        with (
            torch._dynamo.config.patch(enable_tensor_ssa_fastpath=False),
            self.collect_fastpath_hits() as hits,
        ):
            self.assertEqual(opt_fn(a, b), fn(a, b))

        self.assertEqual(hits, [])

    def test_tensor_subclass_torch_function_falls_back(self):
        class TestSubclass(torch.Tensor):
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                return super().__torch_function__(func, types, args, kwargs or {})

        def fn(x):
            return x + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4).as_subclass(TestSubclass)

        with self.collect_fastpath_hits() as hits:
            self.assertEqual(opt_fn(x), fn(x))

        self.assertEqual(hits, [])

    def test_sparse_tensor_falls_back(self):
        def fn(x):
            return x + x

        opt_fn = torch.compile(fn, backend="eager")
        indices = torch.tensor([[0, 1, 1], [2, 0, 2]])
        values = torch.tensor([3.0, 4.0, 5.0])
        x = torch.sparse_coo_tensor(indices, values, (2, 3))

        with self.collect_fastpath_hits() as hits:
            self.assertEqual(opt_fn(x).to_dense(), fn(x).to_dense())

        self.assertEqual(hits, [])

    def test_symnode_constant_matmul_and_unary_fastpath(self):
        def fn(a, b):
            return -(a @ b + a.shape[0] + 2)

        backend = EagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True, dynamic=True)
        a = torch.randn(3, 4)
        b = torch.randn(4, 5)

        with self.collect_fastpath_hits() as hits:
            self.assertEqual(opt_fn(a, b), fn(a, b))

        self.assertEqual(
            hits,
            [
                ("stack", operator.matmul),
                ("stack", operator.add),
                ("stack", operator.add),
                ("stack", operator.neg),
            ],
        )

    def test_fastpath_preserves_shape_dependency_current_node(self):
        from functorch.experimental.control_flow import cond
        from torch.fx.experimental.proxy_tensor import make_fx

        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1

        def true_fn(x):
            return x + a

        def false_fn(x):
            return x + b

        def fn(x):
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

        gm = make_fx(fn, tracing_mode="symbolic", _allow_non_fake_inputs=True)(
            torch.randn(2, 3)
        )

        self.assertIn(
            "sym_size_int_1 = torch.ops.aten.sym_size.int(x_1, 1)",
            gm.code,
        )
        self.assertIn(
            "(x_1, _tensor_constant0, sym_size_int_1, sym_size_int, _tensor_constant1)",
            gm.code,
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
