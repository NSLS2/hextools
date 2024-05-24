from __future__ import annotations

import datetime
import socket
from socket import gaierror, herror

from caproto.threading.client import Context


def now():
    """A helper function to return ISO 8601 formatted datetime string."""
    return datetime.datetime.now().isoformat()


def replace_curlies(string, how_many=2):
    """A helper function to replace multiple curly braces with one."""
    return string.replace("{" * how_many, "{").replace("}" * how_many, "}")


def get_ioc_hostname(pvname):
    """A helper function to get the IOC hostname based on the provided PV."""

    ctx = Context()
    (pv,) = ctx.get_pvs(pvname)  # pylint: disable=unbalanced-tuple-unpacking
    pv.wait_for_connection()
    s = pv.circuit_manager.socket

    epics_addr = s.getpeername()[0]
    sci_addr = epics_addr.split(".")
    sci_addr[2] = str(int(sci_addr[2]) - 3)
    sci_addr = ".".join(sci_addr)

    try:
        hostname = socket.gethostbyaddr(sci_addr)[0]
    except (gaierror, herror):
        hostname = socket.gethostbyaddr(epics_addr)[0]

    return hostname
