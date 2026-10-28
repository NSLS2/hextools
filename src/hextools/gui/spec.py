"""What a beamline supplies to the data-acquisition GUI.

The GUI itself is generic; everything specific to one beamline (its techniques,
the plans and cameras in each, its devices, its facility's proposal system,
where it is) is described by a :class:`GuiSpec`. A beamline package defines one
spec and launches the GUI with it, e.g. ``hextools.gui_config.HEX_GUI``.

This module imports no Qt, so a spec can be defined and tested without a display.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DetectorView:
    """A live image viewer for one area detector.

    ``options`` are passed to ``QtTabbedDetectorsWidget.add_detector``.
    """

    name: str
    prefix: str
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Technique:
    """One tab on the left: detector viewers beside a column of plan forms."""

    title: str
    plans: Sequence[tuple[str, Callable[..., Any]]] = ()
    detectors: Sequence[DetectorView] = ()
    combined: Sequence[tuple[str, Sequence[str]]] = ()


@dataclass(frozen=True)
class Site:
    """Where the beamline is, for the date, time and weather box."""

    title: str
    latitude: float
    longitude: float
    timezone: str


@dataclass(frozen=True)
class ProposalBackend:
    """The facility's proposal system, for the Proposal Info box.

    ``md_is_shared`` says whether a switch made on a copy of the RunEngine
    metadata already reaches the Queue Server worker (e.g. Redis-backed).
    """

    list_proposals: Callable[[], Mapping[str, Sequence[tuple[str, str]]]]
    get_proposal_info: Callable[[str], Mapping[str, Any]]
    switch_proposal: Callable[..., Mapping[str, Any]]
    md_is_shared: Callable[[], bool] = lambda: True


@dataclass(frozen=True)
class GuiSpec:
    """Everything beamline-specific about the data-acquisition GUI.

    Parameters
    ----------
    name : str
        Short beamline name, used in the window title, e.g. ``"HEX"``.
    techniques : sequence of Technique
        The technique tabs, in order.
    expected_devices : sequence of (name, type)
        Devices listed on the Available Devices tab.
    logger_name : str
        Logger whose records the Plan Status and Log views show.
    run_specs : sequence
        Tiled specs offered by the Scan Browser.
    status_widgets : sequence of callables
        Extra boxes for the top row in local mode, each built as
        ``factory(run_engine, namespace)``.
    extra_tabs : sequence of (title, callable)
        Extra tabs after the built-in ones, each built as ``factory()``.
    proposals : ProposalBackend, optional
        The facility's proposal system. Without one, Proposal Info only shows
        the metadata and offers no Change Proposal.
    site : Site, optional
        Location for the date, time and weather box; omitted if not given.
    profile_dir : Path, optional
        Folder holding the IPython profiles the launcher can start.
    default_profile : str
        Profile started when none is named.
    environment : mapping
        Environment variables set before the profile starts.
    mock_environment : mapping
        Environment variables set for ``--mock``.
    plan_history_env : str
        Environment variable naming the plan-history file.
    about_packages : sequence of str
        Packages whose versions the About box lists.
    settings_scope : (str, str)
        Organization and application under which user preferences are kept.
    """

    name: str
    techniques: Sequence[Technique]
    expected_devices: Sequence[tuple[str, type]] = ()
    logger_name: str = "bluesky"
    run_specs: Sequence[Any] = ()
    status_widgets: Sequence[Callable[..., Any]] = ()
    extra_tabs: Sequence[tuple[str, Callable[[], Any]]] = ()
    proposals: ProposalBackend | None = None
    site: Site | None = None
    profile_dir: Path | None = None
    default_profile: str = "collection"
    environment: Mapping[str, str] = field(default_factory=dict)
    mock_environment: Mapping[str, str] = field(default_factory=dict)
    plan_history_env: str = "BLUESKY_GUI_PLAN_HISTORY_FILE"
    about_packages: Sequence[str] = ("bluesky", "ophyd-async", "tiled")
    settings_scope: tuple[str, str] = ("bluesky", "bluesky-gui")

    @property
    def title(self) -> str:
        """Window title."""
        return f"{self.name} Data Acquisition"


def load_spec(ref: str) -> GuiSpec:
    """Import a spec named ``"package.module:NAME"``."""
    from importlib import import_module

    module_name, _, attr = ref.partition(":")
    spec = getattr(import_module(module_name), attr)
    if not isinstance(spec, GuiSpec):
        raise TypeError(f"{ref} is a {type(spec).__name__}, not a GuiSpec")
    return spec
