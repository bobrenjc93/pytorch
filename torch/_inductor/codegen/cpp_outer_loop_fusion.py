# mypy: allow-untyped-defs
import functools
import operator

import sympy

from ..scheduler import BaseSchedulerNode, FusedSchedulerNode, Scheduler, SchedulerNode
from .cpp import CppKernel, CppKernelProxy, LoopNest, ParallelDepth


class OuterLoopFusedKernel(CppKernel):
    def __init__(self, kernel_group):
        super().__init__(kernel_group.args, kernel_group.ws.num_threads)
        self.inner: list[LoopNest] = []

    def decide_parallel_depth(self, max_parallel_depth, threads):
        kernels_parallel_depth = []
        nested_kernels: list[CppKernel] = [
            loop_nest.get_kernel() for loop_nest in self.inner
        ]
        # TODO(leslie-fang-intel): only enable parallel within all outer loop levels.
        for kernel in nested_kernels:
            # For any ScalarKernel, VecKernel, or Tile2DKernel,
            # they should all have the same call_ranges
            call_ranges = kernel.call_ranges
            assert call_ranges is not None
            kernels_parallel_depth.append(
                kernel.decide_parallel_depth(
                    ParallelDepth(
                        parallel_depth=(
                            len(call_ranges) - max_parallel_depth.start_depth
                        ),
                        start_depth=max_parallel_depth.start_depth,
                    ),
                    threads,
                ).parallel_depth
            )
        return ParallelDepth(
            parallel_depth=min(
                max_parallel_depth.parallel_depth, max(kernels_parallel_depth)
            ),
            start_depth=max_parallel_depth.start_depth,
        )


class OuterLoopFusedSchedulerNode(FusedSchedulerNode):
    @classmethod
    def fuse(  # type: ignore[override]
        cls, node1: BaseSchedulerNode, node2: BaseSchedulerNode, outer_loop_fusion_depth
    ):
        assert node1.scheduler is node2.scheduler
        assert all(
            type(node)
            in (
                OuterLoopFusedSchedulerNode,
                SchedulerNode,
                FusedSchedulerNode,
            )
            for node in (node1, node2)
        )
        if any(type(node) is OuterLoopFusedSchedulerNode for node in (node1, node2)):
            return cls(
                node1.scheduler,
                # pyrefly: ignore [bad-argument-type]
                (
                    list(node1.get_outer_nodes())
                    if type(node1) is OuterLoopFusedSchedulerNode
                    else [
                        node1,
                    ]
                )
                + (
                    list(node2.get_outer_nodes())
                    if type(node2) is OuterLoopFusedSchedulerNode
                    else [
                        node2,
                    ]
                ),
                outer_loop_fusion_depth,
            )
        else:
            return cls(node1.scheduler, [node1, node2], outer_loop_fusion_depth)  # type: ignore[list-item]

    def __init__(
        self,
        scheduler: "Scheduler",
        outer_fused_nodes: list[FusedSchedulerNode | SchedulerNode],
        outer_loop_fusion_depth,
    ):
        self.outer_fused_nodes: list[FusedSchedulerNode | SchedulerNode] = (
            outer_fused_nodes
        )
        self.outer_loop_fusion_depth = outer_loop_fusion_depth
        flatten_snodes = []
        for _node in self.outer_fused_nodes:
            assert isinstance(_node, (SchedulerNode, FusedSchedulerNode))
            flatten_snodes.extend(list(_node.get_nodes()))
        super().__init__(scheduler, flatten_snodes)  # type: ignore[arg-type]

    def get_outer_nodes(self):
        return self.outer_fused_nodes

    def check_outer_fusion_loop_level_attr(
        self, cpp_kernel_proxy_list, outer_loop_fusion_depth
    ):
        # This function ensures that the same tiling split is applied at each loop level within the outer loop fusion depth.
        # In the fusion stage, we only examine nodes with same vars and reduce.
        # However, for nodes with same vars and reduce, the loops may still have different tile splits.
        # For example (test_expr_vec_non_contiguous in test_cpu_repro.py):
        #   * buf0 tiling along the 2nd loop level, buf1 tiling along the 3rd loop level.
        # If the check failed, we should fall back to standard loop codegen.
        def _inner(
            left_loop_nest: LoopNest,
            right_loop_nest: LoopNest,
            loop_fusion_depth: int,
            current_checking_depth: int,
        ) -> bool:
            assert left_loop_nest.loops
            assert right_loop_nest.loops
            left_loop_level = left_loop_nest.loops[current_checking_depth]
            right_loop_level = right_loop_nest.loops[current_checking_depth]
            # Check if same loop level attr
            outer_loops_attr_compare_list = [
                "var",
                "size",
                "offset",
                "steps",
            ]
            if not (
                all(
                    getattr(left_loop_level, attr_compare)
                    == getattr(right_loop_level, attr_compare)
                    for attr_compare in outer_loops_attr_compare_list
                )
            ):
                return False

            assert loop_fusion_depth >= 1
            if (loop_fusion_depth := loop_fusion_depth - 1) > 0:
                # Check next loop level attr
                current_checking_depth = current_checking_depth + 1
                assert current_checking_depth < len(left_loop_nest.loops)
                assert current_checking_depth < len(right_loop_nest.loops)
                if not _inner(
                    left_loop_nest,
                    right_loop_nest,
                    loop_fusion_depth,
                    current_checking_depth,
                ):
                    return False

            return True

        for idx in range(len(cpp_kernel_proxy_list) - 1):
            left_loop_nest = cpp_kernel_proxy_list[idx].loop_nest
            right_loop_nest = cpp_kernel_proxy_list[idx + 1].loop_nest
            if not _inner(
                left_loop_nest,
                right_loop_nest,
                outer_loop_fusion_depth,
                0,
            ):
                return False

        for cpp_kernel_proxy in cpp_kernel_proxy_list:
            outer_ranges = functools.reduce(
                operator.mul,
                cpp_kernel_proxy.ranges[:outer_loop_fusion_depth],
            )
            # When the range of the first inner loop is much larger than the range of
            # all outer loops, do not fuse outer loop and fallback to standard loop codegen,
            # so that the inner loops with larger range have a chance to be parallelized.
            # We set a conservative threshold here:
            # First inner loop range / all outer loops range > 300.
            if (
                len(cpp_kernel_proxy.ranges) > outer_loop_fusion_depth
                and isinstance(outer_ranges, sympy.Integer)
                and isinstance(
                    cpp_kernel_proxy.ranges[outer_loop_fusion_depth],
                    sympy.Integer,
                )
                and outer_ranges * 300
                < cpp_kernel_proxy.ranges[outer_loop_fusion_depth]
            ):
                return False

        return True

    def merge_outer_fusion_kernels(
        self,
        cpp_kernel_proxy_list,
    ):
        kernel_group = cpp_kernel_proxy_list[0].kernel_group
        outer_loop_fused_kernel = OuterLoopFusedKernel(kernel_group)
        outer_loop_fused_kernel.inner = [
            proxy.loop_nest.from_loop_level(self.outer_loop_fusion_depth)
            for proxy in cpp_kernel_proxy_list
        ]
        outer_fused_proxy = cpp_kernel_proxy_list[0]
        outer_fused_proxy.loop_nest.kernel = outer_loop_fused_kernel
        outer_fused_proxy.loop_nest.loops = outer_fused_proxy.loop_nest.loops[
            : self.outer_loop_fusion_depth
        ]
        return outer_fused_proxy
