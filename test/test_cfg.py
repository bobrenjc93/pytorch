# Owner(s): ["module: fx"]

import torch
import torch.cfg as cfg
from torch.fx import Graph, GraphModule
from torch.testing._internal.common_utils import run_tests, TestCase


class TestTorchCfg(TestCase):
    def test_manual_cfg_is_valid_and_printable(self):
        x = cfg.Value("x", cfg.TensorSpec.from_tensor(torch.randn(2, 3)))
        negative_input = cfg.Value("negative_input", x.spec)
        result = cfg.Value("result", x.spec)
        pred = cfg.Value("pred", cfg.ScalarSpec(bool))
        negated = cfg.Value("negated", x.spec)

        graph = cfg.Graph(
            name="branchy",
            entry="entry",
            blocks=(
                cfg.Block(
                    name="entry",
                    parameters=(x,),
                    instructions=(
                        cfg.Instruction(
                            name="lt_zero",
                            opcode="call_function",
                            target=torch.ops.aten.lt.Scalar,
                            inputs=(x, 0),
                            outputs=(pred,),
                        ),
                    ),
                    terminator=cfg.Branch(
                        pred,
                        cfg.Successor("negative", (x,)),
                        cfg.Successor("done", (x,)),
                    ),
                ),
                cfg.Block(
                    name="negative",
                    parameters=(negative_input,),
                    instructions=(
                        cfg.Instruction(
                            name="negate",
                            opcode="call_function",
                            target=torch.neg,
                            inputs=(negative_input,),
                            outputs=(negated,),
                        ),
                    ),
                    terminator=cfg.Jump(cfg.Successor("done", (negated,))),
                ),
                cfg.Block(
                    name="done",
                    parameters=(result,),
                    instructions=(),
                    terminator=cfg.Return(result),
                ),
            ),
        )

        rendered = graph.format()
        self.assertIn("graph branchy:", rendered)
        self.assertIn("block entry", rendered)
        self.assertIn("branch %pred -> negative(%x), done(%x)", rendered)
        self.assertIn("jump done(%negated)", rendered)

    def test_tensor_spec_from_tensor_handles_dense_and_nested_tensors(self):
        dense = torch.randn(2, 3, requires_grad=True)
        dense_spec = cfg.TensorSpec.from_tensor(dense)

        self.assertEqual(dense_spec.shape, (2, 3))
        self.assertEqual(dense_spec.dtype, dense.dtype)
        self.assertEqual(dense_spec.device, dense.device)
        self.assertEqual(dense_spec.stride, dense.stride())
        self.assertTrue(dense_spec.requires_grad)

        if not hasattr(torch, "nested") or not hasattr(torch.nested, "nested_tensor"):
            self.skipTest("nested tensors are unavailable")

        nested = torch.nested.nested_tensor([torch.randn(2, 3), torch.randn(4, 3)])
        nested_spec = cfg.TensorSpec.from_tensor(nested)

        self.assertIsNone(nested_spec.stride)

    def test_from_fx_normalizes_metadata_and_preserves_structure(self):
        fx_graph = Graph()
        x = fx_graph.placeholder("x")
        x.meta["example_value"] = torch.randn(2, 3)
        add = fx_graph.call_function(torch.add, args=(x, 1.0))
        add.meta["val"] = torch.randn(2, 3)
        fx_graph.output((x, {"sum": add}))

        graph = cfg.from_fx(fx_graph, name="from_fx")

        entry = graph.block("entry")
        self.assertEqual(entry.parameters[0].name, "x")
        self.assertIsInstance(entry.parameters[0].spec, cfg.TensorSpec)
        self.assertEqual(len(entry.instructions), 1)
        instruction = entry.instructions[0]
        self.assertEqual(instruction.name, "add")
        self.assertEqual(str(instruction.inputs[1]), "1.0")
        self.assertIsInstance(instruction.outputs[0].spec, cfg.TensorSpec)
        self.assertEqual(
            entry.terminator.format(),
            "return (%x, {sum: %add})",
        )

    def test_from_fx_accepts_graph_module_and_preserves_stack_trace(self):
        fx_graph = Graph()
        x = fx_graph.placeholder("x")
        x.meta["example_value"] = torch.randn(2, 3)
        add = fx_graph.call_function(torch.add, args=(x, 1.0))
        add.meta["val"] = torch.randn(2, 3)
        add.meta["stack_trace"] = "frame 1\nexample.py:5 in forward"
        fx_graph.output(add)

        graph = cfg.from_fx(
            GraphModule(torch.nn.Module(), fx_graph),
            name="graph_module",
        )

        entry = graph.block("entry")
        self.assertEqual(graph.name, "graph_module")
        self.assertEqual(entry.terminator.format(), "return %add")
        self.assertEqual(
            entry.instructions[0].location.format(),
            "example.py:5 in forward",
        )

    def test_invalid_cfg_raises_validation_error(self):
        x = cfg.Value("x")
        with self.assertRaisesRegex(
            cfg.ValidationError,
            "expects 1 arguments but received 2",
        ):
            cfg.Graph(
                name="invalid",
                entry="entry",
                blocks=(
                    cfg.Block(
                        name="entry",
                        parameters=(x,),
                        terminator=cfg.Jump(
                            cfg.Successor("done", (x, cfg.literal(1))),
                        ),
                    ),
                    cfg.Block(
                        name="done",
                        parameters=(cfg.Value("y"),),
                        terminator=cfg.Return(None),
                    ),
                ),
            )

    def test_invalid_cfg_rejects_cross_block_reference(self):
        x = cfg.Value("x")
        y = cfg.Value("y")

        with self.assertRaisesRegex(
            cfg.ValidationError,
            "references undefined value 'x'",
        ):
            cfg.Graph(
                name="invalid",
                entry="entry",
                blocks=(
                    cfg.Block(
                        name="entry",
                        parameters=(x,),
                        terminator=cfg.Jump(cfg.Successor("done")),
                    ),
                    cfg.Block(
                        name="done",
                        instructions=(
                            cfg.Instruction(
                                name="negate",
                                opcode="call_function",
                                target=torch.neg,
                                inputs=(x,),
                                outputs=(y,),
                            ),
                        ),
                        terminator=cfg.Return(y),
                    ),
                ),
            )


if __name__ == "__main__":
    run_tests()
