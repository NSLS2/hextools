"""Reusable, beamline-configurable Bluesky Qt GUI."""

from hextools.gui.spec import (
    DetectorView,
    GuiSpec,
    ProposalBackend,
    Site,
    Technique,
    load_spec,
)


def run(spec_ref: str, argv: list[str] | None = None):
    """Launch the generic GUI for a ``"module:SPEC"`` configuration."""
    from hextools.gui.window import run as _run

    return _run(spec_ref, argv)


__all__ = [
    "DetectorView",
    "GuiSpec",
    "ProposalBackend",
    "Site",
    "Technique",
    "load_spec",
    "run",
]
