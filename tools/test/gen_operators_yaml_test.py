#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import argparse
import json
import unittest
from collections import defaultdict
from typing import cast, TypedDict
from unittest.mock import Mock, patch

# pyrefly: ignore [import-error, missing-import]
from gen_operators_yaml import (
    fill_output,
    get_parser_options,
    make_filter_from_options,
    verify_all_specified_present,
)


DepGraph = dict[str, set[str]]


class _MockOptions(argparse.Namespace):
    root_ops: str
    training_root_ops: list[str]
    output_path: str
    dep_graph_yaml_path: str
    model_name: str
    model_versions: str | None
    model_assets: str | None
    model_backends: str | None
    models_yaml_path: list[str] | None
    include_all_operators: bool
    rule_name: str
    not_include_all_overloads_static_root_ops: bool
    not_include_all_overloads_closure_ops: bool


class _ModelMetadata(TypedDict):
    name: str
    version: int
    asset: str
    backend: str


class _ModelConfig(TypedDict, total=False):
    model: _ModelMetadata
    root_operators: list[str]
    traced_operators: list[str]


class _OperatorConfig(TypedDict):
    include_all_overloads: bool


def _mock_options() -> _MockOptions:
    options = _MockOptions()
    options.root_ops = "aten::add,aten::cat"
    options.training_root_ops = []
    options.output_path = "/tmp"
    options.dep_graph_yaml_path = "dummy_pytorch_op_deps.yaml"
    options.model_name = "test_model"
    options.model_versions = None
    options.model_assets = None
    options.model_backends = None
    options.models_yaml_path = None
    options.include_all_operators = False
    options.rule_name = "test_rule"
    options.not_include_all_overloads_static_root_ops = True
    options.not_include_all_overloads_closure_ops = True

    return options


def _mock_load_op_dep_graph() -> DepGraph:
    result: defaultdict[str, set[str]] = defaultdict(set)
    result["aten::add"] = {"aten::add", "aten::as_strided_"}
    result["aten::cat"] = {"aten::cat", "aten::as_strided_"}
    return dict(result)


def _make_model_config(
    name: str,
    version: int,
    asset: str,
    *,
    include_traced_operators: bool,
) -> _ModelConfig:
    config: _ModelConfig = {
        "model": {
            "name": name,
            "version": version,
            "asset": asset,
            "backend": "CPU",
        },
        "root_operators": [],
    }
    if include_traced_operators:
        config["traced_operators"] = []
    return config


class GenOperatorsYAMLTest(unittest.TestCase):
    def test_filter_creation(self) -> None:
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=None,
            model_backends=None,
        )
        config = [
            _make_model_config("abc", 100, "asset-1", include_traced_operators=True),
            _make_model_config("abc", 102, "asset-1", include_traced_operators=False),
            _make_model_config("abcd", 100, "asset-1", include_traced_operators=True),
            _make_model_config("abc", 101, "asset-2", include_traced_operators=False),
        ]

        filtered_configs = list(filter(filter_func, config))
        if len(filtered_configs) != 2:
            raise AssertionError(
                f"Expected 2 elements in filtered_configs, but got {len(filtered_configs)}"
            )

    def test_verification_success(self) -> None:
        filter_func = make_filter_from_options(
            model_name="abc",
            model_versions=["100", "101"],
            model_assets=["asset-1", "asset-2"],
            model_backends=None,
        )
        config = [
            _make_model_config("abc", 100, "asset-1", include_traced_operators=True),
            _make_model_config("abc", 101, "asset-2", include_traced_operators=False),
        ]
        filtered_configs = list(filter(filter_func, config))
        try:
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2"],
                model_versions=["100", "101"],
                selected_models_yaml=filtered_configs,
                rule_name="test",
                model_name="abc",
                new_style_rule=True,
            )
        except Exception:
            self.fail(
                "expected verify_all_specified_present to succeed instead it raised an exception"
            )

    def test_verification_fail(self) -> None:
        config = [
            _make_model_config("abc", 100, "asset-1", include_traced_operators=True),
            _make_model_config("abc", 101, "asset-2", include_traced_operators=False),
        ]

        good_assets = ["asset-1", "asset-2"]
        good_versions = ["100", "101"]
        good_name = "abc"

        # Test bad asset
        filter_func_bad_asset = make_filter_from_options(
            model_name=good_name,
            model_versions=good_versions,
            model_assets=["asset-1", "asset-2", "asset-3"],
            model_backends=None,
        )
        filtered_configs_asset = list(filter(filter_func_bad_asset, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=["asset-1", "asset-2", "asset-3"],
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_asset,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # Test bad version
        filter_func_bad_version = make_filter_from_options(
            model_name=good_name,
            model_versions=["100", "101", "102"],
            model_assets=good_assets,
            model_backends=None,
        )
        filtered_configs_version = list(filter(filter_func_bad_version, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=["100", "101", "102"],
                selected_models_yaml=filtered_configs_version,
                rule_name="test",
                model_name=good_name,
                new_style_rule=True,
            )

        # Test bad name
        filter_func_bad_name = make_filter_from_options(
            model_name="abcd",
            model_versions=good_versions,
            model_assets=good_assets,
            model_backends=None,
        )
        filtered_configs_name = list(filter(filter_func_bad_name, config))
        with self.assertRaises(RuntimeError):
            verify_all_specified_present(
                model_assets=good_assets,
                model_versions=good_versions,
                selected_models_yaml=filtered_configs_name,
                rule_name="test",
                model_name="abcd",
                new_style_rule=True,
            )

    @patch("gen_operators_yaml.parse_options", return_value=_mock_options())
    @patch(
        "gen_operators_yaml.load_op_dep_graph", return_value=_mock_load_op_dep_graph()
    )
    def test_fill_output_with_arguments_not_include_all_overloads(
        self, mock_parse_options: Mock, mock_load_op_dep_graph: Mock
    ) -> None:
        parser = argparse.ArgumentParser(description="Generate used operators YAML")
        options = get_parser_options(parser)

        model_dict: dict[str, object] = {
            "model_name": options.model_name,
            "asset_info": {},
            "is_new_style_rule": False,
        }
        output: dict[str, object] = {"debug_info": [json.dumps(model_dict)]}

        fill_output(output, options)

        operators = cast(dict[str, _OperatorConfig], output["operators"])
        for op_val in operators.values():
            self.assertFalse(op_val["include_all_overloads"])
