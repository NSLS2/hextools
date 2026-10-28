"""Error helpers shared by the GUI and the RunEngine logging (no Qt)."""

from bluesky.utils import FailedStatus


def unwrap_failed_status(error: BaseException) -> BaseException:
    """Return the error behind a ``FailedStatus``, or ``error`` itself otherwise.

    The RE raises ``FailedStatus(status) from <original error>`` when a status fails.
    """
    while isinstance(error, FailedStatus) and error.__cause__ is not None:
        error = error.__cause__
    return error
