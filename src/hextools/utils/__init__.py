"""General utility functions for hextools."""

import asyncio
import os
import shutil
import sys
from collections.abc import AsyncIterable, MutableMapping
from datetime import datetime
from typing import Any, TypeVar

from bluesky import plan_stubs as bps
from bluesky.run_engine import RunEngine
from IPython.core.getipython import get_ipython
from nslsii.sync_experiment import sync_experiment
from nslsii.utils import open_redis_client, print_proposal_info
from ophyd_async.core import (
    Device,
    DeviceVector,
)
from redis_json_dict.redis_json_dict import RedisJSONDict
from rich import print as rprint

from .nl_msg_hook import nl_msg_hook as nl_msg_hook
from .path_providers import (
    StartDocAwareFilenameProvider as StartDocAwareFilenameProvider,
)

NSVarT = TypeVar("NSVarT")


def get_obj_from_ipython_ns(var_name: str, var_type: type[NSVarT]) -> NSVarT | None:
    """Get an obj from the IPython ns if it exists and is of the correct type.

    var_name : str
        The name of the variable to retrieve from the IPython namespace.
    var_type : type[NSVarT]
        The expected type of the variable.

    Returns
    -------
    NSVarT | None
        The object from the IPython namespace if it exists and is of the correct
        type, otherwise None.
    """
    ipython = get_ipython()
    if ipython is not None:
        obj = ipython.user_ns.get(var_name)
        if isinstance(obj, var_type):
            return obj
    return None


def ensure_available(var_type: type[NSVarT], **kwargs: NSVarT | None) -> NSVarT:
    """Ensure that an object with a given name is available in the current scope.

    If the object is not available locally, attempt to retrieve it from the IPython
    namespace.

    Parameters
    ----------
    var_type : type[NSVarT]
        The expected type of the object.
    **kwargs : NSVarT | None
        A single keyword argument where the key is the name of the variable and the
        value is the object or None.

    Returns
    -------
    NSVarT
        The object if it is available locally or in the IPython namespace.

    Raises
    ------
    ValueError
        If the object is not available locally or in the IPython namespace.
    TypeError
        If the provided value is not of the expected type or None.
    """
    if len(kwargs) != 1:
        raise ValueError("Can only check availability of a single object at a time.")

    name = next(iter(kwargs))
    value = kwargs[name]
    if not isinstance(value, var_type) and value is not None:
        raise TypeError(
            f"Value for {name} must be of type {var_type} or None, is {type(value)}"
        )
    elif isinstance(value, var_type):
        return value
    else:
        value = get_obj_from_ipython_ns(name, var_type)
        if value is not None:
            return value
        raise ValueError(
            f"Object {name} of type {var_type} is not available locally, "
            "or in the IPython namespace!"
        )


async def merge_async_iterables(*aiterables: AsyncIterable):
    """Merge multiple async iterables into a single async iterable.

    Parameters
    ----------
    *aiterables : AsyncIterable
        The async iterables to merge.

    Returns
    -------
    AsyncIterable
        An async iterable that yields items from all input async iterables as they
        become available.
    """
    queue: asyncio.Queue = asyncio.Queue()
    sentinel = object()

    async def forward(ait):
        async for item in ait:
            await queue.put(item)

    async def run():
        await asyncio.gather(*(asyncio.ensure_future(forward(a)) for a in aiterables))
        await queue.put(sentinel)

    task = asyncio.ensure_future(run())
    while (item := await queue.get()) is not sentinel:
        yield item
    await task


def is_running_in_ci() -> bool:
    """Check if the code is running in a continuous integration environment.

    Returns
    -------
    bool
        True if running in a continuous integration environment, False otherwise.
    """
    return os.environ.get("HEXTOOLS_RUNNING_IN_CI", "false").lower() in [
        "true",
        "1",
        "yes",
    ]


def show_docs(name: str, doc: dict[str, Any]):
    """Print out the bluesky documents in a readable format.

    Parameters
    ----------
    name : str
        The name of the document.
    doc : dict[str, Any]
        The document to print.
    """
    rprint(f"------- {name} ---------")
    rprint(doc)


def initialize_run_engine() -> RunEngine:
    """Initialize the bluesky RunEngine with appropriate metadata.

    Returns
    -------
    RunEngine
        An initialized bluesky RunEngine with appropriate metadata.
    """
    if is_running_in_ci():
        return RunEngine(
            {
                "data_session": "pass-123456",
                "cycle": (
                    f"{datetime.today().year}-{int(datetime.today().month / 4) + 1}"
                ),
                "proposal": {
                    "title": "Mock mode proposal",
                    "type": "Mock Commissioning",
                    "pi_name": os.getenv("USER", "Unknown"),
                },
            }
        )
    return RunEngine(RedisJSONDict(open_redis_client(redis_ssl=True), ""))


def start_beamtime(proposal_id: int, verbose: bool = True) -> None:
    """Start a beamtime for the given proposal ID.

    Parameters
    ----------
    proposal_id : int
        The ID of the proposal to start the beamtime for.
    verbose : bool, optional
        Whether to print detailed information, by default True.
    """
    md = sync_experiment(proposal_id, "HEX", redis_ssl=True)

    rprint(
        f"Started beamtime for proposal ID [bold][blue]{proposal_id}[/blue][/bold].\n"
    )
    rprint(
        "Current time: [italic]"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/italic]\n"
    )
    print_proposal_info(md)


NSLS2_API_URL = "https://api.nsls2.bnl.gov"
_MOCK_PROPOSALS = {
    "123456": {"title": "Mock mode proposal", "type": "Mock Commissioning"},
    "314159": {"title": "Another mock proposal", "type": "Mock General User"},
}
_MOCK_COMMISSIONING = ["123456"]


def _sorted_desc(proposals: dict[str, str]) -> list[tuple[str, str]]:
    return sorted(
        proposals.items(),
        key=lambda p: int(p[0]) if p[0].isdigit() else 0,
        reverse=True,
    )


def list_proposals(beamline: str = "HEX") -> dict[str, list[tuple[str, str]]]:
    """Return ``(number, title)`` of the proposals for ``beamline``, newest first.

    Returns
    -------
    dict[str, list[tuple[str, str]]]
        ``"user"``: the beamline's proposals in the current cycle, and
        ``"commissioning"``: its commissioning proposals.
    """
    if is_running_in_ci():
        titles = {p: info["title"] for p, info in _MOCK_PROPOSALS.items()}
        return {
            "user": [(p, t) for p, t in titles.items() if p not in _MOCK_COMMISSIONING],
            "commissioning": [(p, titles.get(p, "")) for p in _MOCK_COMMISSIONING],
        }

    import httpx

    with httpx.Client(base_url=NSLS2_API_URL, timeout=10) as client:
        cycle = (
            client.get("/v1/facility/nsls2/cycles/current")
            .raise_for_status()
            .json()["cycle"]
        )
        in_cycle = set(
            client.get(f"/v1/facility/nsls2/cycle/{cycle}/proposals")
            .raise_for_status()
            .json()["proposals"]
        )
        # The beamline-filtered listing needs an API key; this public one doesn't.
        beamline_proposals = (
            client.get("/v1/proposals/recent/1000", params={"beamline": beamline})
            .raise_for_status()
            .json()["proposals"]
        )
        commissioning = set(
            client.get("/v1/proposals/commissioning", params={"beamline": beamline})
            .raise_for_status()
            .json()
            .get("commissioning_proposals", [])
        )
    titles = {p["proposal_id"]: p.get("title") or "" for p in beamline_proposals}
    user = {p: t for p, t in titles.items() if p in in_cycle and p not in commissioning}
    return {
        "user": _sorted_desc(user),
        "commissioning": _sorted_desc({p: titles.get(p, "") for p in commissioning}),
    }


def get_proposal_info(proposal_number: int | str) -> dict[str, str]:
    """Return the ``title``, ``type``, and ``pi_name`` of a proposal."""
    if is_running_in_ci():
        info = _MOCK_PROPOSALS.get(
            str(proposal_number), {"title": "Mock proposal", "type": "Mock"}
        )
        return {**info, "pi_name": os.getenv("USER", "Unknown")}

    import httpx

    with httpx.Client(base_url=NSLS2_API_URL, timeout=10) as client:
        proposal = (
            client.get(f"/v1/proposal/{proposal_number}")
            .raise_for_status()
            .json()["proposal"]
        )
    if "error_message" in proposal:
        raise ValueError(proposal["error_message"])
    pi = next((u for u in proposal.get("users") or [] if u.get("is_pi")), {})
    return {
        "title": proposal.get("title") or "",
        "type": proposal.get("type") or "",
        "pi_name": f"{pi.get('first_name') or ''} {pi.get('last_name') or ''}".strip(),
    }


def _run_sync_experiment(
    proposal_number: int,
    username: str,
    password: str,
    beamline: str,
    timeout: float = 60.0,
) -> str:
    """Run ``sync-experiment``, answering its prompts.

    Return its output, or raise on failure.
    """
    import pexpect

    exe = shutil.which(
        "sync-experiment", path=os.path.dirname(sys.executable)
    ) or shutil.which("sync-experiment")
    if exe is None:
        raise RuntimeError("The sync-experiment command was not found.")

    child = pexpect.spawn(
        exe,
        ["--beamline", beamline, "--proposal", str(proposal_number), "--enable-ssl"],
        encoding="utf-8",
        timeout=timeout,
    )
    output = ""
    try:
        for prompt, answer in (("Username : ", username), ("Password : ", password)):
            if child.expect([prompt, pexpect.EOF]) == 1:
                output += child.before or ""
                break
            output += (child.before or "") + prompt
            child.sendline(answer)
        else:
            child.expect(pexpect.EOF)
            output += child.before or ""
    except pexpect.TIMEOUT:
        raise RuntimeError("sync-experiment timed out.") from None
    finally:
        child.close(force=True)

    if child.exitstatus != 0:
        lines = [line for line in output.splitlines() if line.strip()]
        raise RuntimeError(
            lines[-1]
            if lines
            else f"sync-experiment failed (exit status {child.exitstatus})."
        )
    return output


def switch_proposal(
    proposal_number: int,
    username: str,
    password: str,
    md: MutableMapping[str, Any] | None = None,
    beamline: str = "HEX",
) -> MutableMapping[str, Any]:
    """Authenticate ``username`` and make ``proposal_number`` the active proposal.

    Runs ``nslsii``'s ``sync-experiment`` command, answering its prompts with the
    given credentials. In mock mode any credentials are accepted and only ``md``
    (e.g. ``RE.md``) is updated.

    Returns
    -------
    MutableMapping[str, Any]
        The updated metadata: ``md`` if given (Redis-backed outside mock mode), else
        an empty dict.
    """
    if is_running_in_ci():
        if md is None:
            raise ValueError("In mock mode, the metadata to update must be given.")
        md["data_session"] = f"pass-{proposal_number}"
        md["username"] = username
        md["start_datetime"] = datetime.now().isoformat()
        info = get_proposal_info(proposal_number)
        md["proposal"] = {
            "proposal_id": str(proposal_number),
            "title": info["title"],
            "type": info["type"],
            "pi_name": info["pi_name"],
        }
        return md

    _run_sync_experiment(proposal_number, username, password, beamline)
    return md if md is not None else {}


PIPE = "│"
ELBOW = "└──"
TEE = "├──"
PIPE_PREFIX = "│   "
SPACE_PREFIX = "    "


def _get_children(device: Device) -> list[tuple[str, Device]]:
    """Return the (name, child) pairs of a device, sorted for display.

    Supplementary method for building the tree view of a device.
    """
    children = [
        (name, child) for name, child in device.children() if isinstance(child, Device)
    ]
    if isinstance(device, DeviceVector):
        # DeviceVector children are stringified integer indices.
        return sorted(children, key=lambda item: int(item[0]))
    return sorted(children, key=lambda item: item[0])


def _make_tree_body(tree: list[str], device: Device, prefix=""):
    """Create the tree body.

    Supplementary method for building the tree view of a device.
    """
    entries = _get_children(device)
    last_index = len(entries) - 1
    for index, (name, child) in enumerate(entries):
        if index == 0:
            tree.append(prefix + PIPE)
        connector = ELBOW if index == last_index else TEE
        tree.append(f"{prefix}{connector} {name}")
        child_prefix = prefix + (SPACE_PREFIX if index == last_index else PIPE_PREFIX)
        _make_tree_body(tree, child, prefix=child_prefix)


def print_device_tree(device: Device, indent: int = 0) -> None:
    """Print the device tree for a given device.

    Parameters
    ----------
    device : Device
        The device whose tree is to be printed.
    indent : int
        The indentation level for the current device.
    """
    x = []
    _make_tree_body(x, device)
    print("\n".join(x))


# Set by LastDarkFlatUIDTracker.
DARK_FLAT_MD_KEYS = ("last_dark_flat_uid", "last_dark_flat_scan_id")
DARK_REFERENCE_MD_KEYS = ("last_dark_reference_uid", "last_dark_reference_scan_id")


def clear_last_dark_flat(RE: RunEngine | None = None):
    """Forget the last dark/flat, so later scans link none until a new one is taken.

    Parameters
    ----------
    RE : RunEngine | None, optional
        The RunEngine to update. If None, it is taken from the IPython namespace.
    """
    RE = ensure_available(RunEngine, RE=RE)
    for key in DARK_FLAT_MD_KEYS:
        RE.md.pop(key, None)
    yield from bps.null()


def reset_scan_id(new_scan_id: int = 0, RE: RunEngine | None = None):
    """Reset the RunEngine's scan ID, and forget the last dark/flat.

    Parameters
    ----------
    new_scan_id : int, optional
        The value to set ``RE.md["scan_id"]`` to, by default 0.
    RE : RunEngine | None, optional
        The RunEngine to update. If None, it is taken from the IPython namespace.
    """
    RE = ensure_available(RunEngine, RE=RE)
    RE.md["scan_id"] = new_scan_id
    # Earlier dark/flats belong to the scans being numbered over.
    yield from clear_last_dark_flat(RE=RE)
