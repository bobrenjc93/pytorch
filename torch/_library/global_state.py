from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, cast
from weakref import WeakKeyDictionary, WeakValueDictionary


def _default_ctx_getter() -> None:
    return None


_T = TypeVar("_T")


class LibraryGlobalState:
    """
    Central storage for Python-side torch.library registries.

    These containers are intentionally still re-exported from their historical
    modules for compatibility. New shared registry state should live here so
    lifecycle and cleanup behavior is easier to audit.
    """

    def __init__(self) -> None:
        self.impls: set[str] = set()
        self.defs: set[str] = set()
        self.keep_alive: list[Any] = []

        self.simple_registry: Any | None = None
        self.global_ctx_getter: Callable[[], Any] = _default_ctx_getter

        self.custom_opdefs: WeakValueDictionary[str, Any] = WeakValueDictionary()
        self.custom_opdef_to_lib: dict[str, Any] = {}
        self.legacy_custom_op_registry: dict[str, Any] = {}

        self.fake_class_registry: Any | None = None
        self.triton_ops_to_kernels: dict[str, list[object]] = {}

        self.opaque_types: WeakKeyDictionary[Any, Any] = WeakKeyDictionary()
        self.opaque_types_by_name: dict[str, Any] = {}

    def get_or_create_simple_registry(self, factory: type[_T]) -> _T:
        if self.simple_registry is None:
            self.simple_registry = factory()
        return cast(_T, self.simple_registry)

    def get_or_create_fake_class_registry(self, factory: type[_T]) -> _T:
        if self.fake_class_registry is None:
            self.fake_class_registry = factory()
        return cast(_T, self.fake_class_registry)


library_state = LibraryGlobalState()


__all__ = ["LibraryGlobalState", "library_state"]
