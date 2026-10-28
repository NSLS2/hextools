import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def session_client():
    """A GUI's connection to a session with an empty namespace and its own RunEngine."""
    pytest.importorskip("qtpy")
    pytest.importorskip("bluesky_widgets")
    from bluesky import RunEngine
    from qtpy.QtWidgets import QApplication

    from hextools.gui.session_client import SessionClient
    from hextools.gui.session_server import SessionServer, exec_in

    QApplication.instance() or QApplication([])
    RE = RunEngine({}, call_returns_result=True)
    namespace = {"RE": RE}
    server = SessionServer(RE, namespace, execute=exec_in(namespace))
    client = SessionClient(server.path)
    client.connect_to_session()
    yield client
    client.close()
    server.close()
