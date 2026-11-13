"""Qt signals for a RunEngine client's state, plans, catalog, history, and logs."""

from __future__ import annotations

from qtpy.QtCore import QObject, Signal

from hextools.gui.run_engine_clients import RunEngineClient
from hextools.gui.signal_status import _Forwarder


class QtLogRecords(QObject):
    """Emit :attr:`record` on the Qt thread for each record the client logs.

    The subscription ends when :meth:`stop` is called or this object (or its
    parent) is deleted.

    Parameters
    ----------
    re_client : RunEngineClient
        The client whose log records to watch.
    parent : QObject
        Owner; usually the widget that shows the records.
    """

    #: ``(created, levelno, levelname, message)`` of each record.
    record = Signal(float, int, str, str)

    def __init__(self, re_client: RunEngineClient, parent: QObject):
        super().__init__(parent)
        emit = self.record.emit
        forwarder = _Forwarder(lambda record: emit(*record))
        self._forwarder = forwarder
        self.destroyed.connect(forwarder.cancel)
        token = re_client.subscribe_logs(lambda *record: forwarder(record))
        forwarder.on_cancel = lambda: re_client.unsubscribe_logs(token)

    def stop(self):
        """End the subscription."""
        self._forwarder.cancel()


class QtRunEngineState(QObject):
    """Emit :attr:`changed` on the Qt thread whenever the RunEngine's state changes.

    The subscription ends when :meth:`stop` is called or this object (or its
    parent) is deleted.

    Parameters
    ----------
    re_client : RunEngineClient
        The RunEngine to watch.
    parent : QObject
        Owner; usually the widget that reacts to the state.
    """

    #: The new state, e.g. ``"paused"``.
    changed = Signal(str)

    def __init__(self, re_client: RunEngineClient, parent: QObject):
        super().__init__(parent)
        # Emitted from the RunEngine's thread; Qt queues it to the receivers'.
        self._forwarder = _Forwarder(self.changed.emit)
        self.destroyed.connect(self._forwarder.cancel)
        token = re_client.subscribe_state(self._forwarder)
        self._forwarder.on_cancel = lambda: re_client.unsubscribe_state(token)

    def stop(self):
        """End the subscription."""
        self._forwarder.cancel()


class QtPlanHistoryChanges(QObject):
    """Emit :attr:`changed` on the Qt thread whenever the plan history changes.

    The subscription ends when :meth:`stop` is called or this object (or its
    parent) is deleted.

    Parameters
    ----------
    re_client : RunEngineClient
        The client whose plan history to watch.
    parent : QObject
        Owner; usually the widget that shows the history.
    """

    #: The whole history, a tuple of
    #: :class:`~hextools.gui.run_engine_clients.HistoryItem`, oldest first.
    changed = Signal(object)

    def __init__(self, re_client: RunEngineClient, parent: QObject):
        super().__init__(parent)
        self._forwarder = _Forwarder(self.changed.emit)
        self.destroyed.connect(self._forwarder.cancel)
        token = re_client.subscribe_history(self._forwarder)
        self._forwarder.on_cancel = lambda: re_client.unsubscribe_history(token)

    def stop(self):
        """End the subscription."""
        self._forwarder.cancel()


class QtCurrentPlan(QObject):
    """Emit :attr:`changed` on the Qt thread whenever the current plan changes.

    The subscription ends when :meth:`stop` is called or this object (or its
    parent) is deleted.

    Parameters
    ----------
    re_client : RunEngineClient
        The RunEngine to watch.
    parent : QObject
        Owner; usually the widget that shows the plan.
    """

    #: The :class:`~hextools.gui.run_engine_clients.PlanInfo`, or None once the
    #: plan finishes.
    changed = Signal(object)

    def __init__(self, re_client: RunEngineClient, parent: QObject):
        super().__init__(parent)
        self._forwarder = _Forwarder(self.changed.emit)
        self.destroyed.connect(self._forwarder.cancel)
        token = re_client.subscribe_plan(self._forwarder)
        self._forwarder.on_cancel = lambda: re_client.unsubscribe_plan(token)

    def stop(self):
        """End the subscription."""
        self._forwarder.cancel()


class QtCatalog(QObject):
    """Emit :attr:`changed` on the Qt thread whenever the plans or devices change.

    The subscription ends when :meth:`stop` is called or this object (or its
    parent) is deleted.

    Parameters
    ----------
    re_client : RunEngineClient
        The client whose plan catalog and device registry to watch.
    parent : QObject
        Owner; usually the widget that offers the plans or devices.
    """

    #: The new :class:`~hextools.catalog.model.PlanCatalog` and
    #: :class:`~hextools.catalog.model.DeviceRegistry`.
    changed = Signal(object, object)

    def __init__(self, re_client: RunEngineClient, parent: QObject):
        super().__init__(parent)
        emit = self.changed.emit
        forwarder = _Forwarder(lambda pair: emit(*pair))
        self._forwarder = forwarder
        self.destroyed.connect(forwarder.cancel)
        token = re_client.subscribe_catalog(
            lambda catalog, registry: forwarder((catalog, registry))
        )
        forwarder.on_cancel = lambda: re_client.unsubscribe_catalog(token)

    def stop(self):
        """End the subscription."""
        self._forwarder.cancel()
