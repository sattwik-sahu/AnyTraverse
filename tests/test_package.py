import importlib
import tomllib
from pathlib import Path

import anytraverse
from anytraverse import typing as anyt


def test_public_api_exports_resolve() -> None:
    for name in anytraverse.__all__:
        assert hasattr(anytraverse, name), name


def test_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    assert anytraverse.__version__ == pyproject["project"]["version"]
    assert pyproject["project"]["requires-python"] == ">=3.12"


def test_models_subpackage_exports_resolve() -> None:
    models = importlib.import_module("anytraverse.models")
    for name in models.__all__:
        assert hasattr(models, name), name


def test_typing_aliases_exist() -> None:
    for name in anyt.__all__:
        assert hasattr(anyt, name), name
