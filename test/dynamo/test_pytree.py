# Owner(s): ["module: dynamo"]
# flake8: noqa: B001,B006,B020,B021,B950,C405,C416,E711,E721,E722,E731,F401,F403,F405,F541,F821,F823
# ruff: noqa: F403,F405,F841
try:
    from .dynamo_test_common import *
except ImportError:
    from dynamo_test_common import *


class MiscTestsPyTree(torch._inductor.test_case.TestCase):
    @parametrize_pytree_module
    def test_tracing_pytree(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]

        counter = CompileCounter()
        torch.compile(fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 3)

    @parametrize_pytree_module
    def test_tracing_nested_pytree(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = [xs, xs, xs, xs]

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_tuples(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = (xs, xs, xs, xs)

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 12)

    @parametrize_pytree_module
    def test_tracing_nested_dicts(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsl = {
            "a": xs,
            "b": xs,
            "c": xs,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    @parametrize_pytree_module
    def test_tracing_nested_mixed_all(self, pytree):
        def fn(xs):
            flat_xs, spec = pytree.tree_flatten(xs)
            res = [x.clone() for x in flat_xs]
            if pytree.__name__ == "optree":
                # The treespec argument comes first in OpTree / JAX PyTree
                return pytree.tree_unflatten(spec, res)
            return pytree.tree_unflatten(res, spec)

        xs = [torch.tensor(i) for i in range(3)]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}
        xsl = {
            "a": xs,
            "b": xsa,
            "c": xsb,
        }

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsl)
        real_out = fn(xsl)
        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 18)

    @parametrize_pytree_module
    def test_tracing_nested_tensor_subclass(self, pytree):
        from torch.testing._internal.two_tensor import TwoTensor
        from torch.utils.checkpoint import checkpoint

        def fn(xs):
            nested_xs = [[xs]]
            flat_xs, spec = pytree.tree_flatten(xs)
            return flat_xs[0].clone()

        # use checkpoint to trigger a "sourceless" tensor subclass
        def checkpoint_fn(xs):
            return checkpoint(fn, xs, use_reentrant=True)

        xs = TwoTensor(torch.ones(2, 2), torch.ones(2, 2))

        counter = CompileCounter()
        torch.compile(checkpoint_fn, backend=counter, fullgraph=True)(xs)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 2)

    @parametrize_pytree_module
    def test_pytree_tree_leaves(self, pytree):
        def fn(x):
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves = pytree.tree_leaves(tree)
            return leaves

        x = torch.randn(3, 2)
        expected = fn(x)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_flatten_unflatten(self, pytree):
        def fn(x, y):
            tree = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            leaves, treespec = pytree.tree_flatten(tree)
            new_leaves = [
                x - 1,
                y,
                x * y,
                3.0,
                y - 2,
                1,
                torch.zeros(2, 2),
                2 * y,
                -y,
                x + y,
                x - y,
                torch.ones(3, 2),
                1,
            ]
            if pytree.__name__ == "optree":
                # `None` is a internal node rather than leaf in default OpTree / JAX PyTree
                new_leaves.pop()
                # The treespec argument comes first in OpTree / JAX PyTree
                new_tree = pytree.tree_unflatten(treespec, new_leaves)
            else:
                new_tree = pytree.tree_unflatten(new_leaves, treespec)
            return leaves, new_tree

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map(self, pytree):
        def fn(x, y):
            tree1 = {
                "a": [x, x - 1],
                "b": x + 2,
                "c": (
                    x,
                    3.0,
                    collections.deque([0.0, -x, 1, 2], maxlen=3),
                ),
                "d": collections.OrderedDict(
                    {
                        "e": torch.return_types.qr((2 * x, None)),
                        "f": MyTuple(x, x + 1, torch.zeros(4, 3)),
                    },
                ),
            }
            tree2 = collections.OrderedDict(
                [
                    ("c", (y, 3.0, collections.deque([1, -y, 10.0]))),
                    ("a", [y, y + 1]),
                    ("b", y + 2),
                    (
                        "d",
                        {
                            "f": MyTuple(torch.ones(4, 3), -y, y + 1),
                            "e": torch.return_types.qr((2 * y, None)),
                        },
                    ),
                ],
            )
            return pytree.tree_map(lambda u, v: (u, v), tree1, tree2)

        x = torch.randn(3, 2)
        y = torch.randn(3, 2)
        expected = fn(x, y)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)
        actual = fn_opt(x, y)

        self.assertEqual(actual, expected)

    @parametrize_pytree_module
    def test_pytree_tree_map_dict_order(self, pytree):
        def fn(tree):
            new_tree = pytree.tree_map(lambda x: x, tree)
            return list(new_tree.keys()), list(new_tree.values())

        x = torch.randn(3, 2)
        fn_opt = torch.compile(fullgraph=True, backend="eager")(fn)

        tree1 = {"b": x + 2, "a": x, "c": x - 1}
        expected1 = fn(tree1)
        actual1 = fn_opt(tree1)
        self.assertEqual(actual1, expected1)

        tree2 = collections.OrderedDict([("b", x + 2), ("a", x), ("c", x - 1)])
        expected2 = fn(tree2)
        actual2 = fn_opt(tree2)
        self.assertEqual(actual2, expected2)

        tree3 = collections.defaultdict(int, {"b": x + 2, "a": x, "c": x - 1})
        expected3 = fn(tree3)
        actual3 = fn_opt(tree3)
        self.assertEqual(actual3, expected3)

    @parametrize_pytree_module
    def test_pytree_tree_map_only(self, pytree):
        if not callable(getattr(pytree, "tree_map_only", None)):
            # OpTree and JAX PyTree do not have `tree_map_only`
            return

        def fn(xs):
            def mapper(x):
                return x.clone()

            y = pytree.tree_map_only(torch.Tensor, mapper, xs)
            return y

        xs = [torch.tensor(i) for i in range(3)] + ["hi"]
        xsa = (xs, xs)
        xsb = {"aa": xsa, "ab": xs}

        counter = CompileCounter()
        comp_out = torch.compile(fn, backend=counter, fullgraph=True)(xsb)
        real_out = fn(xsb)

        self.assertEqual(comp_out, real_out)
        self.assertEqual(counter.frame_count, 1)
        self.assertEqual(counter.op_count, 9)

    def test_pytree_register_constant_with_side_effect(self):
        class Foo:
            pass

        class Bar:
            def __eq__(self, other):
                return super().__eq__(other)

            def __hash__(self):
                return 0

        python_pytree.register_constant(Bar)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, obj):
            obj.attr = {3: Bar()}
            return x + 1

        inp = torch.ones(3)
        self.assertEqual(fn(inp, Foo()), inp + 1)


instantiate_parametrized_tests(MiscTestsPyTree)
if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
