"""
Python polyfills for torch.utils.pytree
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, TYPE_CHECKING

import torch.utils._pytree as python_pytree

from ..decorators import substitute_in_graph


if TYPE_CHECKING:
    import builtins

    from torch.utils._cxx_pytree import PyTree


__all__: list[str] = []


if python_pytree._cxx_pytree_exists:
    import optree

    import torch.utils._cxx_pytree as cxx_pytree

    @substitute_in_graph(
        optree._C.is_dict_insertion_ordered,
        can_constant_fold_through=True,
    )
    def always_true(*args: Any, **kwargs: Any) -> Literal[True]:
        # In namespace 'torch', the dictionary is always traversed in insertion order.
        return True

    @substitute_in_graph(cxx_pytree.tree_iter)
    def tree_iter(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> Iterable[Any]:
        stack = [tree]
        while stack:
            curr = stack.pop()
            if curr is None or (is_leaf is not None and is_leaf(curr)):
                yield curr
                continue
            if optree.register_pytree_node.get(type(curr), namespace="torch") is None:  # type: ignore[attr-defined]
                yield curr
                continue

            (
                children,
                metadata,
                entries,
                unflatten_func,
            ) = optree.tree_flatten_one_level(
                curr,
                is_leaf=is_leaf,
                none_is_leaf=True,
                namespace="torch",
            )
            stack.extend(reversed(children))

    __all__ += ["tree_iter"]

    @substitute_in_graph(cxx_pytree.tree_leaves, can_constant_fold_through=True)
    def tree_leaves(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> list[Any]:
        return list(tree_iter(tree, is_leaf=is_leaf))

    __all__ += ["tree_leaves"]

    @dataclass(frozen=True)
    class PyTreeSpec:
        _children: tuple[PyTreeSpec, ...]
        _type: builtins.type | None
        _metadata: Any
        _entries: tuple[Any, ...]
        _unflatten_func: Callable[[Any | None, Iterable[PyTree]], PyTree] | None

        num_nodes: int = field(init=False)
        num_leaves: int = field(init=False)
        num_children: int = field(init=False)
        none_is_leaf: bool = field(init=False)
        namespace: str = field(init=False)

        def __post_init__(self) -> None:
            if self._type is None:
                assert len(self._children) == 0
                assert self._metadata is None
                assert self._entries is None
                assert self._unflatten_func is None
                object.__setattr__(self, "num_nodes", 1)
                object.__setattr__(self, "num_leaves", 1)
                object.__setattr__(self, "num_children", 0)
            else:
                assert callable(self._unflatten_func)
                num_nodes = sum((spec.num_nodes for spec in self._children), start=1)
                num_leaves = sum(spec.num_leaves for spec in self._children)
                num_children = len(self._children)
                object.__setattr__(self, "num_nodes", num_nodes)
                object.__setattr__(self, "num_leaves", num_leaves)
                object.__setattr__(self, "num_children", num_children)

            object.__setattr__(self, "none_is_leaf", True)
            object.__setattr__(self, "namespace", "torch")

        @property
        def type(self) -> builtins.type | None:
            return self._type

        def is_leaf(self) -> bool:
            return self.num_nodes == 1 and self.num_leaves == 1

        def children(self) -> list[PyTreeSpec]:
            return list(self._children)

        def child(self, index: int) -> PyTreeSpec:
            return self._children[index]

        def entries(self) -> list[Any]:
            return list(self._entries)

        def entry(self, index: int) -> Any:
            return self._entries[index]

        def unflatten(self, leaves: Iterable[Any]) -> PyTree:
            if not isinstance(leaves, (list, tuple)):
                leaves = list(leaves)
            if len(leaves) != self.num_leaves:
                raise ValueError(
                    f"treespec.unflatten(leaves): `leaves` has length {len(leaves)} "
                    f"but the spec refers to a pytree that holds {self.num_leaves} "
                    f"items ({self}).",
                )
            if self.is_leaf():
                return leaves[0]

            # Recursively unflatten the children
            start = 0
            end = 0
            subtrees = []
            for subspec in self._children:
                end += subspec.num_leaves
                subtrees.append(subspec.unflatten(leaves[start:end]))
                start = end

            assert callable(self._unflatten_func)
            return self._unflatten_func(self._metadata, subtrees)

    leafspec = PyTreeSpec([], None, None, None, None)

    @substitute_in_graph(cxx_pytree.tree_flatten, can_constant_fold_through=False)  # type: ignore[arg-type]
    def tree_flatten(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> tuple[list[Any], PyTreeSpec]:
        def helper(node: PyTree, leaves: list[Any]) -> PyTreeSpec:
            if node is None or (is_leaf is not None and is_leaf(node)):
                leaves.append(node)
                return leafspec

            node_type = type(node)
            if optree.register_pytree_node.get(node_type, namespace="torch") is None:  # type: ignore[attr-defined]
                leaves.append(node)
                return leafspec

            (
                children,
                metadata,
                entries,
                unflatten_func,
            ) = optree.tree_flatten_one_level(
                node,
                is_leaf=is_leaf,
                none_is_leaf=True,
                namespace="torch",
            )

            subspecs = tuple(helper(child, leaves) for child in children)
            return PyTreeSpec(subspecs, node_type, metadata, entries, unflatten_func)  # type: ignore[arg-type]

        leaves: list[Any] = []
        treespec = helper(tree, leaves)
        return leaves, treespec

    __all__ += ["tree_flatten"]

    @substitute_in_graph(cxx_pytree.tree_structure, can_constant_fold_through=False)  # type: ignore[arg-type]
    def tree_structure(
        tree: PyTree,
        is_leaf: Callable[[PyTree], bool] | None = None,
    ) -> PyTreeSpec:
        return tree_flatten(tree, is_leaf=is_leaf)[1]  # type: ignore[return-value]

    __all__ += ["tree_structure"]

    @substitute_in_graph(cxx_pytree.tree_unflatten, can_constant_fold_through=False)  # type: ignore[arg-type]
    def tree_unflatten(leaves: Iterable[Any], treespec: PyTreeSpec) -> PyTree:
        if not isinstance(treespec, PyTreeSpec):
            raise TypeError(
                f"tree_unflatten(values, spec): Expected `spec` to be instance of "
                f"TreeSpec but got item of type {type(treespec)}."
            )
        return treespec.unflatten(leaves)

    __all__ += ["tree_unflatten"]
