"""Plan stubs for configuring HEX area detectors."""

from collections.abc import (
    Sequence,
)

import bluesky.plan_stubs as bps
from ophyd_async.core import (
    EnableDisable,
    OnOff,
)
from ophyd_async.epics.adcore import (
    ADImageMode,
    AreaDetector,
    NDPluginBaseIO,
    NDProcessFilterType,
    NDProcessIO,
)
from ophyd_async.epics.adkinetix import (
    KinetixDetector,
    KinetixTriggerMode,
)
from ophyd_async.epics.advimba import (
    VimbaDetector,
    VimbaExposeOutMode,
    VimbaTriggerSource,
)

from hextools.devices.detectors.perkin_elmer import PerkinElmerDetector
from hextools.devices.detectors.phantom import PhantomDetector


def enable_preview_mode(
    cam: KinetixDetector | VimbaDetector | PerkinElmerDetector | PhantomDetector,
):
    """Enable preview mode for the Camera.

    This sets the image mode to continuous and starts acquisition. A Phantom
    instead stops any recording and turns on its live preview.

    Parameters
    ----------
    cam : KinetixDetector | VimbaDetector | PerkinElmerDetector | PhantomDetector
        The camera to enable preview mode for.
    """
    if isinstance(cam, PhantomDetector):
        acquiring = yield from bps.rd(cam.driver.acquire)
        if acquiring:
            yield from bps.abs_set(cam.driver.acquire, False, wait=True)
        yield from bps.mv(cam.driver.preview, 1)
        return
    # In the case of a PE, we want to keep the proc plugin on, and instead switch away
    # from having the CB plugin be the source of the image viewing
    if isinstance(cam, PerkinElmerDetector):
        driver_array_port = yield from bps.rd(cam.driver.port_name)
        trans_plugin = cam.get_plugin("trans", NDPluginBaseIO)
        yield from bps.mv(trans_plugin.nd_array_port, driver_array_port)
        return

    acquiring = yield from bps.rd(cam.driver.acquire)
    if acquiring:
        yield from bps.abs_set(cam.driver.acquire, False, wait=True)

    if hasattr(cam, "proc"):
        proc_plugin = cam.get_plugin("proc", NDProcessIO)
        yield from bps.mv(proc_plugin.enable_callbacks, EnableDisable.DISABLE)

    yield from bps.mv(cam.driver.image_mode, ADImageMode.CONTINUOUS)
    if isinstance(cam, KinetixDetector):
        yield from bps.mv(cam.driver.trigger_mode, KinetixTriggerMode.INTERNAL)
    elif isinstance(cam, VimbaDetector):
        yield from bps.mv(
            cam.driver.trigger_mode,
            OnOff.OFF,
            cam.driver.trigger_source,
            VimbaTriggerSource.FREERUN,
            cam.driver.exposure_mode,
            VimbaExposeOutMode.TIMED,
        )
    yield from bps.sleep(0.1)

    # Since this is a busy record, wait=True will block till
    # the acquisition completes
    yield from bps.abs_set(cam.driver.acquire, True, wait=False)


def set_proc_filter_type(
    dets: Sequence[AreaDetector], filter_type: NDProcessFilterType | str
):
    """Set the Proc plugin's filter type on every detector that has one.

    ophyd-async's prepare leaves the filter type to the IOC, so without this an
    averaged acquisition uses whatever the Proc plugin was last left on.

    Parameters
    ----------
    dets : Sequence[AreaDetector]
        The detectors; those without a ``proc`` plugin are skipped.
    filter_type : NDProcessFilterType | str
        The filter type, or its EPICS name (e.g. ``"Average"``).
    """
    filter_type = NDProcessFilterType(filter_type)
    sigs = []
    for det in dets:
        if hasattr(det, "proc"):
            proc_plugin = det.get_plugin("proc", NDProcessIO)
            sigs.extend([proc_plugin.filter_type, filter_type])
    if sigs:
        yield from bps.mv(*sigs)


def set_plugin_queue_sizes(
    dets: AreaDetector | Sequence[AreaDetector], queue_size: int
):
    """Set the queue size of all plugins known to ophyd to a given value.

    Parameters
    ----------
    dets : AreaDetector | Sequence[AreaDetector]
        The detector or list of detectors whose plugin queue sizes are to be set.
    queue_size : int
        The desired queue size for all plugins.
    """
    sigs = []
    if isinstance(dets, AreaDetector):
        dets = [dets]

    for det in dets:
        for _, child in det.children():
            if isinstance(child, NDPluginBaseIO):
                sigs.extend([child.queue_size, queue_size])

    yield from bps.mv(*sigs)


def enable_background_subtraction(det: AreaDetector):
    """Enable background subtraction on the given detector's Proc plugin.

    Parameters
    ----------
    det : AreaDetector
        The detector whose Proc plugin will have background subtraction enabled.
    """
    proc_plugin = det.get_plugin("proc", NDProcessIO)
    yield from bps.abs_set(proc_plugin.save_background, True, wait=True)
    yield from bps.mv(proc_plugin.enable_background, True)
