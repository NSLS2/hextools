from __future__ import annotations

import importlib.metadata

import hextools as m


def test_version():
    assert importlib.metadata.version("hextools") == m.__version__
