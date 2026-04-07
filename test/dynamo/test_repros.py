"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""

# Owner(s): ["module: dynamo"]

try:
    from ._repros_common import (
        instantiate_device_type_tests,
        instantiate_parametrized_tests,
        LoggingTestCase,
        torch,
    )
    from ._repros_device import ReproTestsDeviceMixin
    from ._repros_part1 import LRUCacheWarningTestsMixin, ReproTestsMixin1
    from ._repros_part2 import ReproTestsMixin2
    from ._repros_part3 import ReproTestsMixin3
except ImportError:
    from _repros_common import (
        instantiate_device_type_tests,
        instantiate_parametrized_tests,
        LoggingTestCase,
        torch,
    )
    from _repros_device import ReproTestsDeviceMixin
    from _repros_part1 import LRUCacheWarningTestsMixin, ReproTestsMixin1
    from _repros_part2 import ReproTestsMixin2
    from _repros_part3 import ReproTestsMixin3


# Keep the public test classes in this module so existing imports, device-type
# expansion, and dynamic-shapes wrappers continue to work unchanged.
class ReproTests(
    torch._dynamo.test_case.TestCase,
):
    def setUp(self) -> None:
        import contextlib

        try:
            from .utils import install_guard_manager_testing_hook
        except ImportError:
            from utils import install_guard_manager_testing_hook

        self.exit_stack = contextlib.ExitStack()
        self.exit_stack.enter_context(
            install_guard_manager_testing_hook(self.guard_manager_clone_hook_fn)
        )
        super().setUp()

    def tearDown(self) -> None:
        self.exit_stack.close()
        super().tearDown()


class LRUCacheWarningTests(LoggingTestCase):
    pass


class ReproTestsDevice(torch._dynamo.test_case.TestCase):
    pass


def _materialize_mixin_members(target_cls, *mixins, skip_names=()):
    # PyTorch's test instantiators mutate generic_cls.__dict__, so split mixin
    # members must be rebound onto the public classes before expansion runs.
    skip_names = set(skip_names)
    for mixin in mixins:
        for name, value in mixin.__dict__.items():
            if name.startswith("__") or name in skip_names:
                continue
            if name in target_cls.__dict__:
                raise AssertionError(
                    f"Duplicate attribute name {target_cls.__name__}.{name}"
                )
            setattr(target_cls, name, value)


_materialize_mixin_members(
    ReproTests,
    ReproTestsMixin1,
    ReproTestsMixin2,
    ReproTestsMixin3,
    skip_names={"setUp", "tearDown"},
)
_materialize_mixin_members(LRUCacheWarningTests, LRUCacheWarningTestsMixin)
_materialize_mixin_members(ReproTestsDevice, ReproTestsDeviceMixin)


instantiate_parametrized_tests(ReproTests)

devices = ["cuda", "hpu"]
instantiate_device_type_tests(ReproTestsDevice, globals(), only_for=devices)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
