"""Plans for storing and applying device settings."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from IPython.core.getipython import get_ipython
from ophyd_async.core import Device, YamlSettingsProvider, walk_rw_signals
from ophyd_async.fastcs.panda import HDFPanda, apply_panda_settings
from ophyd_async.plan_stubs import (
    apply_settings,
    apply_settings_if_different,
    retrieve_settings,
    store_settings,
)

_SETTINGS_ROOT = Path(__file__).parent
_SNAPSHOT_DIR = _SETTINGS_ROOT / "snapshots"


def _saved_settings_enum() -> type[StrEnum]:
    """Build an enum of the settings files saved per device, by file stem."""
    stems = sorted(
        {
            path.stem
            for path in _SETTINGS_ROOT.glob("*/*.yaml")
            if path.parent != _SNAPSHOT_DIR and not path.stem.startswith("_")
        }
    )
    return StrEnum("SavedDeviceSettings", {stem: stem for stem in stems})


# Built at import, so settings stored afterwards appear after a reload/restart.
if TYPE_CHECKING:
    SavedDeviceSettings = str
else:
    SavedDeviceSettings = _saved_settings_enum()


def _is_connected(device: Device) -> bool:
    """Return True only if ``device`` has completed a successful connection."""
    if getattr(device, "_mock", None) is not None:
        return True
    task = getattr(device, "_connect_task", None)
    return bool(
        task is not None
        and task.done()
        and not task.cancelled()
        and task.exception() is None
    )


def store_device_settings_snapshot(name: str = "snapshot"):
    """Store a snapshot of the settings for all devices."""
    ipython = get_ipython()
    if ipython is None:
        raise RuntimeError("Could not establish IPython environment")

    snapshot_dir = (
        Path(__file__).parent
        / "snapshots"
        / (name + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    )

    provider = YamlSettingsProvider(snapshot_dir)
    for obj in ipython.user_ns.values():
        if isinstance(obj, Device) and _is_connected(obj) and walk_rw_signals(obj):
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            yield from store_settings(provider, obj.name, obj)


def store_device_settings(
    device: Device,
    name: str | None = None,
    only_config: bool = False,
    overwrite: bool = False,
):
    """Store the settings for a device into a yml file.

    Parameters
    ----------
    device : Device
        The device whose settings are to be stored.
    name : str | None, optional
        The name of the settings file. If None, the device's name is used.
    only_config : bool, optional
        If True, only the configuration settings of the device are stored.
    overwrite : bool, optional
        If True, overwrite the existing settings file if it exists.
    """
    settings_dir = Path(__file__).parent / device.name
    settings_dir.mkdir(parents=True, exist_ok=True)
    provider = YamlSettingsProvider(settings_dir)

    filename = name or device.name
    if (settings_dir / f"{filename}.yaml").exists() and not overwrite:
        filename = filename + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    yield from store_settings(provider, filename, device, only_config=only_config)


def apply_device_settings(
    device: Device, name: SavedDeviceSettings | None = None, only_config: bool = False
):
    """Apply the settings for a device from a yml file.

    Parameters
    ----------
    device : Device
        The device whose settings are to be applied.
    name : SavedDeviceSettings | None, optional
        The name of the saved settings file. If None, the device's name is used.
    only_config : bool, optional
        If True, only the configuration settings of the device are applied.
    """
    settings_dir = Path(__file__).parent / device.name
    provider = YamlSettingsProvider(settings_dir)

    filename = name or device.name
    if not (settings_dir / f"{filename}.yaml").exists():
        raise FileNotFoundError(
            f"Settings file '{filename}.yaml' not found in '{settings_dir}'"
        )

    settings = yield from retrieve_settings(
        provider, filename, device, only_config=only_config
    )

    apply_plan = apply_settings
    if isinstance(device, HDFPanda):
        apply_plan = apply_panda_settings
    yield from apply_settings_if_different(settings, apply_plan=apply_plan)
