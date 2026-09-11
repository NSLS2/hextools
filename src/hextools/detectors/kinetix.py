"""Kinetix detector support for HEX beamline."""

# from ophyd_async.epics.adkinetix import KinetixDetector
# from ophyd_async.epics.adcore import ADWriterFactory

# def kinetix_factory(num: int, path_provider, name: str):
#     """Factory function to create a KinetixDetector with HDF writer."""
#     return KinetixDetector(
#         f"XF:27ID1-BI{{Kinetix-Det:{num}}}",
#         ADWriterFactory.hdf(path_provider),
#         proc_suffix="Proc1:",
#         name=name,
#     )

from ophyd_async.core import PathProvider, TriggerInfo
from ophyd_async.epics.adcore import ADWriterFactory
from ophyd_async.epics.adkinetix import KinetixDetector

# class HEXKinetixDetector(KinetixDetector):
#     """HEX-specific Kinetix detector with HDF writer."""

#     async def prepare(self, value: TriggerInfo | None = None) -> None:
#         proc = getattr(self, "proc", None)
#         hdf = getattr(self, "hdf", None)
#         if proc is not None and hdf is not None:
#             await hdf.nd_array_port.set(await proc.port_name.get_value())
#         await super().prepare(value)

def kinetix_factory(num: int, path_provider: PathProvider, name: str):
    """Factory function to create a HEXKinetixDetector with HDF writer."""
    return KinetixDetector(
        f"XF:27ID1-BI{{Kinetix-Det:{num}}}",
        ADWriterFactory.hdf(path_provider),
        proc_suffix="Proc1:",
        name=name,
    )
