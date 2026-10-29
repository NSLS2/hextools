import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import IPython
from IPython.core.interactiveshell import InteractiveShell
from qtpy.QtWidgets import QApplication
from tiled.structures.core import Spec

from hextools.gui import window
from hextools.gui.spec import GuiSpec


@pytest.mark.parametrize("mock", [False, True])
@pytest.mark.parametrize("history", [None, "explicit.json"])
def test_command_line_overrides_base_environment(monkeypatch, tmp_path, mock, history):
    profile = tmp_path / "test.py"
    profile.write_text("")
    spec = GuiSpec(
        name="Synthetic",
        techniques=(),
        profile_dir=tmp_path,
        default_profile="test",
        plan_history_env="TEST_GUI_HISTORY",
        environment={"TEST_GUI_MODE": "live", "TEST_GUI_HISTORY": "base.json"},
        mock_environment={"TEST_GUI_MODE": "mock"},
    )
    monkeypatch.setattr(window, "load_spec", lambda _: spec)
    monkeypatch.setattr(IPython, "start_ipython", lambda *args, **kwargs: None)
    monkeypatch.setenv("TEST_GUI_MODE", "inherited")
    monkeypatch.setenv("TEST_GUI_HISTORY", "inherited.json")
    argv = ["--mock"] if mock else []
    if history:
        argv.extend(["--history-file", history])
    window.run("synthetic:SPEC", argv)
    assert os.environ["TEST_GUI_MODE"] == ("mock" if mock else "live")
    assert os.environ["TEST_GUI_HISTORY"] == (history or "base.json")


@pytest.mark.parametrize("client_name", ["c", "tiled_reading_client"])
@pytest.mark.parametrize("run_specs", [(), (Spec("synthetic-run"),)])
def test_scan_browser_requires_run_specs(monkeypatch, RE, client_name, run_specs):
    app = QApplication.instance() or QApplication([])
    shell = InteractiveShell.instance()
    monkeypatch.setitem(shell.user_ns, client_name, object())
    selector = window.QtTabbedTechniqueSelector(
        RE, GuiSpec(name="Synthetic", techniques=(), run_specs=run_specs)
    )
    try:
        if run_specs:
            assert selector.scan_browser is not None
            assert selector.scan_browser._spec_combo.count() == 1
            assert selector.scan_browser._spec_combo.currentText() == "synthetic-run"
        else:
            assert selector.scan_browser is None
    finally:
        selector.close()
        app.processEvents()
