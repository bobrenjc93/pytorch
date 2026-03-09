#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import unittest

from tools.code_analyzer.gen_oplist import throw_if_any_op_includes_overloads

from torchgen.selective_build.selector import SelectiveBuilder


class GenOplistTest(unittest.TestCase):
    def test_throw_if_any_op_includes_overloads(self) -> None:
        selective_builder = SelectiveBuilder.from_yaml_str(
            """
operators:
  aten::op1:
    is_root_operator: No
    is_used_for_training: No
    include_all_overloads: Yes
  aten::op2:
    is_root_operator: No
    is_used_for_training: No
    include_all_overloads: No
  aten::op3:
    is_root_operator: No
    is_used_for_training: No
    include_all_overloads: Yes
"""
        )
        with self.assertRaises(Exception):
            throw_if_any_op_includes_overloads(selective_builder)

        selective_builder = SelectiveBuilder.from_yaml_str(
            """
operators:
  aten::op1:
    is_root_operator: No
    is_used_for_training: No
    include_all_overloads: No
  aten::op2:
    is_root_operator: No
    is_used_for_training: No
    include_all_overloads: No
  aten::op3:
    is_root_operator: No
    is_used_for_training: No
    include_all_overloads: No
"""
        )

        # Here we do not expect it to throw an exception since none of the ops
        # include all overloads.
        throw_if_any_op_includes_overloads(selective_builder)
