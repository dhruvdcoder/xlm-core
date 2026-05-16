"""Unit tests for :mod:`xlm.commands.scaffold_model` (pure helpers only)."""

import ast
import json
from pathlib import Path

import pytest

from xlm.commands.scaffold_model import (
    create_template_context,
    generate_datamodule_file,
    generate_init_file,
    generate_loss_file,
    generate_metrics_file,
    generate_model_file,
    generate_predictor_file,
    generate_types_file,
    update_xlm_models_file,
    validate_model_name,
)


class TestValidateModelName:
    @pytest.mark.parametrize("name", ["my_model", "a", "x9_y", "abc123"])
    def test_accepts_valid(self, name):
        assert validate_model_name(name) == name

    @pytest.mark.parametrize(
        "name",
        ["MyModel", "9foo", "my-model", "", "Model_x", "_leading_underscore"],
    )
    def test_rejects_invalid(self, name):
        with pytest.raises(ValueError, match="must start"):
            validate_model_name(name)


class TestCreateTemplateContext:
    def test_basic(self):
        ctx = create_template_context("my_model")
        assert ctx["model_name"] == "my_model"
        assert ctx["model_name_upper"] == "MY_MODEL"
        assert ctx["model_class_name"] == "MyModel"
        assert ctx["model_class_prefix"] == "MyModel"

    def test_single_word(self):
        ctx = create_template_context("foo")
        assert ctx["model_class_name"] == "Foo"

    def test_multi_word(self):
        ctx = create_template_context("a_b_c")
        assert ctx["model_class_name"] == "ABC"


class TestUpdateXlmModelsFile:
    def test_creates_file_when_missing(self, tmp_path: Path):
        path = tmp_path / "xlm_models.json"
        update_xlm_models_file("foo", path)
        assert path.exists()
        assert json.loads(path.read_text()) == {"foo": "foo"}

    def test_appends_to_existing(self, tmp_path: Path):
        path = tmp_path / "xlm_models.json"
        path.write_text(json.dumps({"existing": "existing"}))
        update_xlm_models_file("foo", path)
        assert json.loads(path.read_text()) == {
            "existing": "existing",
            "foo": "foo",
        }

    def test_idempotent_on_second_call(self, tmp_path: Path):
        path = tmp_path / "xlm_models.json"
        update_xlm_models_file("foo", path)
        update_xlm_models_file("foo", path)
        assert json.loads(path.read_text()) == {"foo": "foo"}

    def test_recovers_from_invalid_json(self, tmp_path: Path):
        path = tmp_path / "xlm_models.json"
        path.write_text("not valid json {")
        update_xlm_models_file("foo", path)
        assert json.loads(path.read_text()) == {"foo": "foo"}


class TestGenerateFiles:
    """Smoke-tests that each generator writes a syntactically valid Python file."""

    @pytest.fixture()
    def context(self):
        return create_template_context("my_model")

    @pytest.fixture()
    def model_dir(self, tmp_path: Path):
        d = tmp_path / "outer" / "my_model"
        d.mkdir(parents=True)
        return d

    @staticmethod
    def _assert_python_parses(path: Path):
        assert path.exists(), f"file not created: {path}"
        # ``ast.parse`` raises SyntaxError on bad f-string substitutions.
        ast.parse(path.read_text())

    def test_types_file(self, model_dir, context):
        generate_types_file(model_dir, context)
        self._assert_python_parses(model_dir / "types_my_model.py")

    def test_model_file(self, model_dir, context):
        generate_model_file(model_dir, context)
        self._assert_python_parses(model_dir / "model_my_model.py")

    def test_loss_file(self, model_dir, context):
        generate_loss_file(model_dir, context)
        self._assert_python_parses(model_dir / "loss_my_model.py")

    def test_predictor_file(self, model_dir, context):
        generate_predictor_file(model_dir, context)
        self._assert_python_parses(model_dir / "predictor_my_model.py")

    def test_datamodule_file(self, model_dir, context):
        generate_datamodule_file(model_dir, context)
        self._assert_python_parses(model_dir / "datamodule_my_model.py")

    def test_metrics_file(self, model_dir, context):
        generate_metrics_file(model_dir, context)
        self._assert_python_parses(model_dir / "metrics_my_model.py")

    def test_init_file_external(self, model_dir, context):
        generate_init_file(model_dir, context, is_core=False)
        self._assert_python_parses(model_dir / "__init__.py")

    def test_init_file_core(self, model_dir, context):
        generate_init_file(model_dir, context, is_core=True)
        self._assert_python_parses(model_dir / "__init__.py")
