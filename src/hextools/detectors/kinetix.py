"""Kinetix detector support for HEX beamline."""

import asyncio

from ophyd_async.core import DeviceMock, callback_on_mock_put, default_mock_class, set_mock_put_proceeds, set_mock_value
from ophyd_async.epics.adkinetix import KinetixDetector, KinetixTriggerMode
from ophyd_async.epics.adcore import ADWriterFactory, NDFileHDF5IO, ADBaseDataType

class KinetixDetectorMock(DeviceMock[KinetixDetector]):
    """Mock behaviour that simulates Kinetix internal series acquisition."""

    async def connect(self, device: KinetixDetector) -> None:
        """Mock signals to simulate Kinetix detector acquisition."""
        # Set default array sizes on driver
        set_mock_value(device.driver.array_size_x, 3200)
        set_mock_value(device.driver.array_size_y, 3200)
        set_mock_value(device.driver.data_type, ADBaseDataType.UINT16)

        # Set default array sizes on HDF plugin if present
        try:
            hdf = device.get_plugin_by_name("hdf", NDFileHDF5IO)
            set_mock_value(hdf.array_size0, 3200)
            set_mock_value(hdf.array_size1, 3200)
            set_mock_value(hdf.file_path_exists, True)
        except (AttributeError, TypeError):
            hdf = None

        # Set default signal values
        set_mock_value(device.driver.acquire_time, 1.0)
        set_mock_value(device.driver.acquire_period, 1.0001)
        set_mock_value(device.driver.num_images, 1)
        set_mock_value(device.driver.serial_number, "Mock Kinetix")
        set_mock_value(device.driver.trigger_mode, KinetixTriggerMode.INTERNAL)

        # Auto-adjust acquire_period when acquire_time exceeds it
        async def _on_acquire_time_write(value: float) -> None:
            current_period = await device.driver.acquire_period.get_value()
            if current_period < value:
                set_mock_value(device.driver.acquire_period, value + 0.0001)

        callback_on_mock_put(device.driver.acquire_time, _on_acquire_time_write)

        # Simulate acquisition on acquire=True
        async def _do_acquisition():
            trigger_mode = await device.driver.trigger_mode.get_value()
            if trigger_mode != KinetixTriggerMode.INTERNAL:
                set_mock_put_proceeds(device.driver.acquire, True)
                return

            num_images = await device.driver.num_images.get_value()
            acquire_time = await device.driver.acquire_time.get_value()
            acquire_period = await device.driver.acquire_period.get_value()

            for i in range(num_images):
                await asyncio.sleep(acquire_time)
                if hdf is not None:
                    num_captured = await hdf.num_captured.get_value()
                    set_mock_value(hdf.num_captured, num_captured + 1)
                if i < num_images - 1:
                    await asyncio.sleep(acquire_period - acquire_time)

            set_mock_value(device.driver.acquire, False)
            set_mock_put_proceeds(device.driver.acquire, True)

        def _on_acquire_write(value: bool) -> None:
            if value:
                set_mock_put_proceeds(device.driver.acquire, False)
                asyncio.ensure_future(_do_acquisition())

        callback_on_mock_put(device.driver.acquire, _on_acquire_write)


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
        ADWriterFactory.hdf(path_provider, hinted=False),
        proc_suffix="Proc1:",
        name=name,
    )
