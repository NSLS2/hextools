"""Kinetix detector support for HEX beamline."""

from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.adcore import ADWriterFactory


def kinetix_factory(num: int, path_provider, name: str):
    """Helper factory function to create a KinetixDetector with HDF writer.

    Parameters
    ----------
    num : int
        The detector number.
    path_provider : PathProvider
        The path provider for the HDF writer.
    name : str
        The name of the detector.

    Returns
    -------
    KinetixDetector
        The created Kinetix detector with HDF writer.
    """

    return KinetixDetector(
        f"XF:27ID1-BI{{Kinetix-Det:{num}}}",
        ADWriterFactory.hdf(path_provider),
        proc_suffix="Proc1:",
        name=name,
    )
