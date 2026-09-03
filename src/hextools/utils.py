"""General utility functions for hextools."""

import asyncio
import os
from collections.abc import MutableMapping
from datetime import datetime
from typing import Any

from bluesky.run_engine import RunEngine
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.prompts import Prompts
from nslsii.sync_experiment import sync_experiment
from nslsii.utils import open_redis_client
from ophyd_async.core import (
    Device,
    DeviceProcessor,
    NotConnectedError,
    wait_for_connection,
)
from pygments.token import Token
from redis_json_dict.redis_json_dict import RedisJSONDict
from rich import print as rprint


async def merge_async_iterables(*aiterables):
    """Merge multiple async iterables into a single async iterable."""
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
    """Check if the code is running in a continuous integration environment."""
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
    """Print out the bluesky documents in a readable format."""
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
    """Initialize the bluesky RunEngine with appropriate metadata."""
    if is_running_in_ci():
        return RunEngine(
            {
                "data_session": "pass-123456",
                "cycle": (
                    f"{datetime.today().year}-{int(datetime.today().month / 4) + 1}"
                ),
            }
        )
    return RunEngine(
        RedisJSONDict(open_redis_client(redis_ssl=True), "")  # type: ignore (TODO: Loosen type of RE.md to Mapping from dict)
    )


def print_proposal_info(md: MutableMapping[str, Any]):
    """Print the proposal information from the RunEngine metadata.

    md : MutableMapping[str, Any]
        The metadata dictionary from the RunEngine.
    """
    proposal_md = md.get("proposal", {})
    if proposal_md:
        rprint(f"Proposal title: [italic]{proposal_md['title']}[/italic]\n")
        rprint(
            f"Proposal type: [italic]{proposal_md['type']}[/italic], "
            f"Proposal PI: [italic]{proposal_md['pi_name']}[/italic]\n"
        )


def start_beamtime(proposal_id: int, verbose: bool = True) -> None:
    """Start a beamtime for the given proposal ID."""
    md = sync_experiment(proposal_id, "HEX", redis_ssl=True)

    rprint(
        f"Started beamtime for proposal ID [bold][blue]{proposal_id}[/blue][/bold].\n"
    )
    rprint(
        "Current time: [italic]"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/italic]\n"
    )
    print_proposal_info(md)


def auto_init_devices(timeout: float = 1.0):
    """Create a DeviceProcessor that connects devices, printing status for each."""
    mock = is_running_in_ci()

    async def process_devices(devices: dict[str, Device]):
        for name, device in devices.items():
            device.set_name(name, child_name_separator="_")
        coros = {
            name: device.connect(mock, timeout) for name, device in devices.items()
        }
        failed: set[str] = set()
        try:
            await wait_for_connection(**coros)
        except NotConnectedError as e:
            failed = set(e.sub_errors.keys())
        for name in devices:
            dots = "." * (40 - len(name))
            if name in failed:
                rprint(f"  {name} {dots} [bold red]Disconnected[/bold red]")
            else:
                rprint(f"  {name} {dots} [bold green]Ok[/bold green]")

    return DeviceProcessor(process_devices)
