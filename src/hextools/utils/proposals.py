"""Look up a facility's proposals: the NSLS-II proposal API, or a mock of it."""

from __future__ import annotations

import os
import random
import shutil
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

NSLS2_API_URL = "https://api.nsls2.bnl.gov"


@dataclass(frozen=True)
class Proposal:
    """A proposal. Listings may leave ``type`` and ``pi_name`` empty."""

    number: str
    title: str = ""
    type: str = ""
    pi_name: str = ""


@dataclass(frozen=True)
class ProposalListing:
    """A beamline's proposals, each newest first."""

    #: Its proposals in the current cycle.
    user: tuple[Proposal, ...] = ()
    #: Its commissioning proposals.
    commissioning: tuple[Proposal, ...] = ()


#: The proposal mock mode starts on.
MOCK_PROPOSAL = Proposal(
    number="123456",
    title="Mock mode proposal",
    type="Mock Commissioning",
    pi_name=os.getenv("USER", "Unknown"),
)


class ProposalClient(ABC):
    """Lists a beamline's proposals, looks up their details, and changes to one."""

    @abstractmethod
    def list_proposals(self) -> ProposalListing:
        """Return the beamline's user and commissioning proposals."""

    @abstractmethod
    def get_proposal(self, number: int | str) -> Proposal:
        """Return the proposal called ``number``, with all its details.

        Raises
        ------
        ValueError
            If there is no such proposal.
        """

    @abstractmethod
    def change_proposal(
        self,
        number: int | str,
        username: str,
        password: str,
        md: MutableMapping[str, Any],
    ) -> None:
        """Sign ``username`` in and make proposal ``number`` the active one.

        Parameters
        ----------
        number : int or str
            The proposal to change to.
        username, password : str
            Credentials of someone on the proposal.
        md : MutableMapping[str, Any]
            The RunEngine metadata (``RE.md``) that records the active proposal.

        Raises
        ------
        ValueError
            If there is no such proposal.
        RuntimeError
            If the change was refused or failed.
        """


def _newest_first(proposals: Iterable[Proposal]) -> tuple[Proposal, ...]:
    return tuple(
        sorted(
            proposals,
            key=lambda p: int(p.number) if p.number.isdigit() else 0,
            reverse=True,
        )
    )


class NSLS2ProposalClient(ProposalClient):
    """Proposals from the NSLS-II proposal API.

    Parameters
    ----------
    beamline : str, optional
        Beamline whose proposals are listed. Default: ``"HEX"``.
    api_url : str, optional
        Base URL of the API.
    timeout : float, optional
        Seconds to wait for each request.
    """

    def __init__(
        self, beamline: str = "HEX", api_url: str = NSLS2_API_URL, timeout: float = 10
    ):
        self.beamline = beamline
        self._api_url = api_url
        self._timeout = timeout

    def _client(self):
        import httpx

        return httpx.Client(base_url=self._api_url, timeout=self._timeout)

    def list_proposals(self) -> ProposalListing:
        with self._client() as client:
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
                client.get(
                    "/v1/proposals/recent/1000", params={"beamline": self.beamline}
                )
                .raise_for_status()
                .json()["proposals"]
            )
            commissioning = set(
                client.get(
                    "/v1/proposals/commissioning", params={"beamline": self.beamline}
                )
                .raise_for_status()
                .json()
                .get("commissioning_proposals", [])
            )
        titles = {p["proposal_id"]: p.get("title") or "" for p in beamline_proposals}
        return ProposalListing(
            user=_newest_first(
                Proposal(number, title)
                for number, title in titles.items()
                if number in in_cycle and number not in commissioning
            ),
            commissioning=_newest_first(
                Proposal(number, titles.get(number, "")) for number in commissioning
            ),
        )

    def get_proposal(self, number: int | str) -> Proposal:
        with self._client() as client:
            proposal = (
                client.get(f"/v1/proposal/{number}")
                .raise_for_status()
                .json()["proposal"]
            )
        if "error_message" in proposal:
            raise ValueError(proposal["error_message"])
        pi = next((u for u in proposal.get("users") or [] if u.get("is_pi")), {})
        return Proposal(
            number=str(number),
            title=proposal.get("title") or "",
            type=proposal.get("type") or "",
            pi_name=f"{pi.get('first_name') or ''} {pi.get('last_name') or ''}".strip(),
        )

    def change_proposal(
        self,
        number: int | str,
        username: str,
        password: str,
        md: MutableMapping[str, Any],
    ) -> None:
        """Run ``nslsii``'s ``sync-experiment``, answering its prompts.

        It writes the proposal to Redis, which ``md`` reads from at the beamline,
        so ``md`` itself isn't written to.
        """
        _run_sync_experiment(number, username, password, self.beamline)


def _run_sync_experiment(
    proposal_number: int | str,
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


_TECHNIQUES = (
    "In situ high-energy X-ray diffraction",
    "Energy-dispersive X-ray diffraction",
    "Fast X-ray tomography",
    "Operando radiography",
    "Strain mapping",
    "Phase identification",
    "Residual stress measurement",
    "3D microstructure imaging",
)
_SAMPLES = (
    "additively manufactured nickel superalloys",
    "lithium-ion battery cathodes",
    "solid-state electrolytes",
    "high-entropy alloys",
    "concrete under freeze-thaw cycling",
    "geological samples at high pressure",
    "welded steel joints",
    "silicon carbide composites",
)
_CONDITIONS = (
    "",
    " under thermomechanical loading",
    " during fast charging",
    " at elevated temperatures",
    " in a reactive gas environment",
)
_COMMISSIONING_TASKS = (
    "Detector commissioning",
    "Beamline alignment",
    "Monochromator calibration",
    "Flyscan timing tests",
    "Sample stage characterization",
    "Shutter and interlock tests",
    "Software integration tests",
    "Optics table survey",
)
_PI_NAMES = (
    "Ada Lovelace",
    "Marie Curie",
    "Rosalind Franklin",
    "Dorothy Hodgkin",
    "William Bragg",
    "Max von Laue",
    "Kathleen Lonsdale",
    "Arthur Compton",
)


class MockProposalClient(ProposalClient):
    """Made-up proposals for mock mode, without the network.

    The proposal mock mode starts on (:data:`MOCK_PROPOSAL`) is always a
    commissioning proposal, beside 10 to 20 generated user and commissioning
    proposals each. The same ``seed`` always generates the same proposals.

    Parameters
    ----------
    seed : int, optional
        Seed for generating the proposals.
    """

    def __init__(self, seed: int = 0):
        rng = random.Random(seed)
        self._proposals: dict[str, Proposal] = {MOCK_PROPOSAL.number: MOCK_PROPOSAL}
        user: list[Proposal] = []
        commissioning: list[Proposal] = [MOCK_PROPOSAL]
        for proposals in (user, commissioning):
            for _ in range(rng.randint(10, 20)):
                number = str(rng.randint(300_000, 399_999))
                while number in self._proposals:
                    number = str(rng.randint(300_000, 399_999))
                if proposals is user:
                    title = (
                        f"{rng.choice(_TECHNIQUES)} of {rng.choice(_SAMPLES)}"
                        f"{rng.choice(_CONDITIONS)}"
                    )
                    proposal_type = "Mock General User"
                else:
                    title = f"HEX {rng.choice(_COMMISSIONING_TASKS)}"
                    proposal_type = MOCK_PROPOSAL.type
                proposal = Proposal(number, title, proposal_type, rng.choice(_PI_NAMES))
                self._proposals[number] = proposal
                proposals.append(proposal)
        self._listing = ProposalListing(
            user=_newest_first(user), commissioning=_newest_first(commissioning)
        )

    def list_proposals(self) -> ProposalListing:
        return self._listing

    def get_proposal(self, number: int | str) -> Proposal:
        try:
            return self._proposals[str(number)]
        except KeyError:
            raise ValueError(f"No proposal {number}") from None

    def change_proposal(
        self,
        number: int | str,
        username: str,
        password: str,
        md: MutableMapping[str, Any],
    ) -> None:
        """Record proposal ``number`` in ``md``; any credentials are accepted."""
        proposal = self.get_proposal(number)
        md["data_session"] = f"pass-{proposal.number}"
        md["username"] = username
        md["start_datetime"] = datetime.now().isoformat()
        md["proposal"] = {
            "proposal_id": proposal.number,
            "title": proposal.title,
            "type": proposal.type,
            "pi_name": proposal.pi_name,
        }
