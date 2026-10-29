from collections.abc import Callable

import pytest
from ophyd_async.core import (
    StaticPathProvider,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import (
    ADBaseDataType,
    ADWriterFactory,
    NDPluginFileIO,
    NDProcessIO,
)
from ophyd_async.epics.adkinetix import KinetixDetector, KinetixTriggerMode


@pytest.fixture
def kinetix_det_factory(
    static_path_provider: StaticPathProvider,
) -> Callable[[int], KinetixDetector]:
    def _factory(
        num: int, with_tiff: bool = False, with_proc: bool = False
    ) -> KinetixDetector:
        with init_devices(mock=True):
            ktx = KinetixDetector(
                f"KTX{num}",
                ADWriterFactory.tiff(static_path_provider)
                if with_tiff
                else ADWriterFactory.hdf(static_path_provider),
                proc_suffix="Proc1:" if with_proc else None,
                name=f"kinetix{num}",
            )
            file_plugin = ktx.get_plugin("tiff" if with_tiff else "hdf", NDPluginFileIO)

        # what the descriptor needs to describe the image
        set_mock_value(ktx.driver.array_size_x, 3200)
        set_mock_value(ktx.driver.array_size_y, 3200)
        set_mock_value(ktx.driver.data_type, ADBaseDataType.UINT16)

        async def _mock_increment_num_captured(_):
            # One Acquire writes num_images frames, as the real camera does; in an
            # external trigger mode it waits for edges instead.
            if await ktx.driver.trigger_mode.get_value() != KinetixTriggerMode.INTERNAL:
                return
            current_num_cap = await file_plugin.num_captured.get_value()
            frames = max(1, await ktx.driver.num_images.get_value())
            if with_proc:
                # the writer sits behind Proc1, which emits one frame per NumFilter
                proc = ktx.get_plugin("proc", NDProcessIO)
                frames //= max(1, await proc.num_filter.get_value())
            set_mock_value(file_plugin.num_captured, current_num_cap + frames)

        callback_on_mock_put(
            file_plugin.capture, lambda _: set_mock_value(file_plugin.num_captured, 0)
        )
        callback_on_mock_put(ktx.driver.acquire, _mock_increment_num_captured)
        callback_on_mock_put(
            file_plugin.file_path,
            lambda _: set_mock_value(file_plugin.file_path_exists, True),
        )
        return ktx

    return _factory
