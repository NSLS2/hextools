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
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.prompts import Prompts
from nslsii.sync_experiment import sync_experiment
from nslsii.utils import open_redis_client
from ophyd_async.core import (
    Device,
    DeviceProcessor,
    DeviceVector,
    NotConnectedError,
    Signal,
    wait_for_connection,
    walk_devices,
)
from ophyd_async.epics.motor import Motor as AsyncEpicsMotor
from ophyd_async.sim import SimMotor
from pygments.token import Token
from redis_json_dict.redis_json_dict import RedisJSONDict
from rich import print as rprint
from rich.console import Console

from .nl_msg_hook import nl_msg_hook as nl_msg_hook
from .path_providers import (
    StartDocAwareFilenameProvider as StartDocAwareFilenameProvider,
)

Steppable = AsyncEpicsMotor | SimMotor

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


def print_version_info():
    """Print version information for bluesky, ophyd_async, tiled, and hextools."""
    from bluesky import __version__ as bluesky_version
    from ophyd_async import __version__ as ophyd_async_version
    from tiled import __version__ as tiled_version

    from hextools import __version__ as hextools_version

    rprint("\n[bold]Version Information[/bold]")
    rprint(f"  [bold]bluesky[/bold]: [blue]{bluesky_version}[/blue]")
    rprint(f"  [bold]ophyd_async[/bold]: [blue]{ophyd_async_version}[/blue]")
    rprint(f"  [bold]tiled[/bold]: [blue]{tiled_version}[/blue]")
    rprint(f"  [bold]hextools[/bold]: [blue]{hextools_version}[/blue]\n")


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


class ProposalIDPrompt(Prompts):
    """Custom IPython prompt that shows the current proposal ID."""

    def __init__(self, RE: RunEngine, shell: TerminalInteractiveShell):
        super().__init__(shell)
        self._RE = RE

    def in_prompt_tokens(self, cli=None):
        return [
            (
                Token.Prompt,
                f"{self._RE.md.get('data_session', 'N/A')} [",
            ),
            (Token.PromptNum, str(self.shell.execution_count)),
            (Token.Prompt, "]: "),
        ]


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


def print_proposal_info(md: MutableMapping[str, Any]):
    """Print the proposal information from the RunEngine metadata.

    md : MutableMapping[str, Any]
        The metadata dictionary from the RunEngine.
    """
    proposal_md = md.get("proposal", {})
    if proposal_md:
        rprint(
            "Active proposal:\n"
            f"  Proposal title: [italic]{proposal_md['title']}[/italic]\n"
            f"  Proposal type: [italic]{proposal_md['type']}[/italic]\n"
            f"  Proposal PI: [italic]{proposal_md['pi_name']}[/italic]\n"
        )


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


# Set by LastDarkFlatUIDTracker; they belong to the proposal the dark/flat was taken in.
DARK_FLAT_MD_KEYS = ("last_dark_flat_uid", "last_dark_flat_scan_id")


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
    (e.g. ``RE.md``) is updated. Moving to a different proposal removes the
    previous proposal's dark/flat (``DARK_FLAT_MD_KEYS``) from ``md``.

    Returns
    -------
    MutableMapping[str, Any]
        The updated metadata: ``md`` if given (Redis-backed outside mock mode), else
        an empty dict.
    """
    new_session = f"pass-{proposal_number}"
    changing = md is not None and md.get("data_session") != new_session

    if is_running_in_ci():
        if md is None:
            raise ValueError("In mock mode, the metadata to update must be given.")
        md["data_session"] = new_session
        md["username"] = username
        md["start_datetime"] = datetime.now().isoformat()
        info = get_proposal_info(proposal_number)
        md["proposal"] = {
            "proposal_id": str(proposal_number),
            "title": info["title"],
            "type": info["type"],
            "pi_name": info["pi_name"],
        }
    else:
        _run_sync_experiment(proposal_number, username, password, beamline)

    # Only once the switch succeeded: a failed login keeps the current proposal.
    if md is not None and changing:
        for key in DARK_FLAT_MD_KEYS:
            md.pop(key, None)
    return md if md is not None else {}


def _failed_paths(error: Exception, prefix: str = "") -> list[str]:
    """Dotted paths of the leaves of a (nested) NotConnectedError."""
    sub = error.sub_errors if isinstance(error, NotConnectedError) else {}
    if not sub:
        return [prefix.rstrip(".")]
    return [
        p for name, err in sub.items() for p in _failed_paths(err, f"{prefix}{name}.")
    ]


def _count_connected_signals(device: Device, error: Exception) -> tuple[int, int]:
    """Count (connected, total) signals of ``device``, given its connection error."""
    signals = [
        path for path, d in walk_devices(device).items() if isinstance(d, Signal)
    ]
    if not signals:
        return 0, 1
    failed = [p for p in _failed_paths(error) if p]
    if not failed:
        return 0, len(signals)
    down = [s for s in signals if any(s == f or s.startswith(f + ".") for f in failed)]
    return len(signals) - len(down), len(signals)


def auto_init_devices(
    timeout: float = 1.0, ok_message="OK", fail_message="FAILED", verbose: bool = False
) -> DeviceProcessor:
    """Create a DeviceProcessor that connects devices, printing status for each.

    Parameters
    ----------
    timeout : float
        The timeout in seconds for each device connection attempt.
    verbose : bool, optional
        Whether to print detailed information, by default False.

    Returns
    -------
    DeviceProcessor
        A DeviceProcessor that connects devices and prints their connection status.
    """
    mock = is_running_in_ci()
    # highlight=False stops rich from coloring dot-runs as ellipses.
    console = Console(highlight=False)
    console.print("\n[bold]Connecting to beamline devices...[/bold]\n")

    async def _process_devices(devices: dict[str, Device]):
        for name, device in devices.items():
            if not device.name:
                device.set_name(name)
        coros = {
            name: device.connect(mock, timeout) for name, device in devices.items()
        }
        failed: set[str] = set()
        reasons: dict[str, str] = {}
        e = NotConnectedError({})
        try:
            await wait_for_connection(**coros)
        except NotConnectedError as err:
            e = err
            failed = set(e.sub_errors.keys())
            reasons = {name: str(err) for name, err in e.sub_errors.items()}

        width = max(len(ok_message), len(fail_message)) + 2
        for name, device in devices.items():
            dots = "." * (40 - len(name))
            # Escaped brackets stay uncolored; only the label text is styled.
            if name not in failed:
                console.print(
                    f"  {name} {dots} \\[[bold green]"
                    f"{ok_message.center(width)}[/bold green]]"
                )
                continue
            line = (
                f"  {name} {dots} \\[[bold red]{fail_message.center(width)}[/bold red]]"
            )
            connected, total = _count_connected_signals(device, e.sub_errors[name])
            if 0 < connected < total:
                line += f" [dim]({connected}/{total} signals connected)[/dim]"
            console.print(line)
        if verbose:
            console.print(
                "\n".join(f"{name}: {reason}" for name, reason in reasons.items())
                if reasons
                else ""
            )

    return DeviceProcessor(_process_devices)


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


def reset_scan_id(new_scan_id: int = 0, RE: RunEngine | None = None):
    """Reset the RunEngine's scan ID.

    Parameters
    ----------
    new_scan_id : int, optional
        The value to set ``RE.md["scan_id"]`` to, by default 0.
    RE : RunEngine | None, optional
        The RunEngine to update. If None, it is taken from the IPython namespace.
    """
    RE = ensure_available(RunEngine, RE=RE)
    RE.md["scan_id"] = new_scan_id
    yield from bps.null()
