"""Smoke tests for modules that pull in heavy optional deps.

These skip cleanly when ``shap`` / ``streamlit`` are not installed, so the core
suite stays runnable in a minimal environment.
"""

from __future__ import annotations

import pytest


def test_explain_module_imports():
    pytest.importorskip("shap")
    import src.explain as explain  # noqa: F401

    assert hasattr(explain, "compute_global_shap")


def test_app_module_imports():
    pytest.importorskip("streamlit")
    pytest.importorskip("shap")
    import app  # noqa: F401

    assert hasattr(app, "main")
