"""
HEX beamline PandA device definition (ophyd-async).

Mirrors hex-profile-collection ``startup/09-panda.py``: the PandA is an
``HDFPanda`` (fastcs, PVA) — position-compare / pulse-train trigger source
and its own HDF capture (the ``Angle`` dataset comes from ``calc[2]``).

Like ``hextools.detectors.kinetix.make_kinetix``, the default path provider is settable so plans can
retarget the PandA HDF output per scan (``panda.path_provider.set(...)``).
"""

from ophyd_async.fastcs.panda import HDFPanda

from hextools.detectors.kinetix import SettablePathProvider


def make_panda(
    panda_id: int = 1, path_provider: SettablePathProvider | None = None
) -> HDFPanda:
    """
    Instantiate the HEX PandA (``XF:27ID1-ES{PANDA:<id>}:``).

    Parameters
    ----------
    panda_id:
        Which PandA to connect to. Default 1 (the tomography PandA).
    path_provider:
        Decides where the PandA's HDF file lands.  Defaults to a fresh
        ``SettablePathProvider`` (filename ``panda``, matching the legacy
        scripts' panda.hdf); exposed as ``panda.path_provider``.
    """
    if path_provider is None:
        path_provider = SettablePathProvider(filename="panda")

    panda = HDFPanda(
        f"XF:27ID1-ES{{PANDA:{panda_id}}}:",
        path_provider,
        name=f"panda{panda_id}",
    )
    # Not an ophyd-async child device — just a handle so plans can retarget
    # the output directory (see SettablePathProvider).
    panda.path_provider = path_provider
    return panda
