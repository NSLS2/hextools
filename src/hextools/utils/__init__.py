"""General utility functions for hextools."""

import asyncio
import os
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
from .proposals import MOCK_PROPOSAL, MockProposalClient, NSLS2ProposalClient

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
                "data_session": f"pass-{MOCK_PROPOSAL.number}",
                "cycle": (
                    f"{datetime.today().year}-{int(datetime.today().month / 4) + 1}"
                ),
                "proposal": {
                    "title": MOCK_PROPOSAL.title,
                    "type": MOCK_PROPOSAL.type,
                    "pi_name": MOCK_PROPOSAL.pi_name,
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
        md.update(
            MockProposalClient().change_proposal(proposal_number, username, password)
        )
        return md

    md = md if md is not None else {}
    md.update(
        NSLS2ProposalClient(beamline).change_proposal(
            proposal_number, username, password
        )
    )
    return md


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
