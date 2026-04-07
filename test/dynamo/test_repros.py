"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_rewrite_assert_with_msg and test_rewrite_assert_without_msg)
"""

# Owner(s): ["module: dynamo"]

try:
    from ._repros_common import (
        instantiate_device_type_tests,
        instantiate_parametrized_tests,
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
        torch,
    )
    from _repros_device import ReproTestsDeviceMixin
    from _repros_part1 import LRUCacheWarningTestsMixin, ReproTestsMixin1
    from _repros_part2 import ReproTestsMixin2
    from _repros_part3 import ReproTestsMixin3


# Keep the public test classes in this module so existing imports, device-type
# expansion, and dynamic-shapes wrappers continue to work unchanged.
class ReproTests(
    LRUCacheWarningTestsMixin,
    ReproTestsMixin1,
    ReproTestsMixin2,
    ReproTestsMixin3,
    torch._dynamo.test_case.TestCase,
):
    pass


class ReproTestsDevice(ReproTestsDeviceMixin, torch._dynamo.test_case.TestCase):
    pass


instantiate_parametrized_tests(ReproTests)

devices = ["cuda", "hpu"]
instantiate_device_type_tests(ReproTestsDevice, globals(), only_for=devices)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
