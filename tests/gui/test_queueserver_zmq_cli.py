from contextlib import nullcontext

import pytest

pytest.importorskip("qtpy")
pytest.importorskip("bluesky_widgets")

import hextools.gui.window as gui_window


def test_gui_connects_directly_to_queueserver_zmq(monkeypatch):
    clients = []
    windows = []

    class Client:
        def __init__(self, **kwargs):
            clients.append(kwargs)

    monkeypatch.setattr(gui_window, "RunEngineClient", Client)
    monkeypatch.setattr(gui_window, "gui_qt", lambda _title: nullcontext())
    monkeypatch.setattr(gui_window, "set_settings_scope", lambda *_: None)
    monkeypatch.setattr(gui_window, "apply_bnl_theme", lambda **_: None)
    monkeypatch.setattr(
        gui_window, "saved_theme", lambda: None
    )
    monkeypatch.setattr(
        gui_window,
        "QtDataAcquisitionWindow",
        lambda **kwargs: windows.append(kwargs),
    )

    gui_window.run(
        "hextools.gui_config:HEX_GUI",
        [
            "--queueserver-zmq-control-addr",
            "tcp://127.0.0.1:60615",
            "--queueserver-zmq-info-addr",
            "tcp://127.0.0.1:60625",
        ],
    )

    assert clients == [
        {
            "zmq_control_addr": "tcp://127.0.0.1:60615",
            "zmq_info_addr": "tcp://127.0.0.1:60625",
        }
    ]
    assert len(windows) == 1
    assert windows[0]["spec"].title == "HEX Data Acquisition"
    assert isinstance(windows[0]["re_client"], Client)


def test_zmq_info_address_requires_control_address():
    with pytest.raises(SystemExit, match="2"):
        gui_window.run(
            "hextools.gui_config:HEX_GUI",
            ["--queueserver-zmq-info-addr", "tcp://127.0.0.1:60625"],
        )
