from collections.abc import Callable

import pytest
from ophyd_async.core import (
    StaticPathProvider,
    callback_on_mock_put,
    init_devices,
    set_mock_value,
)
from ophyd_async.epics.adcore import ADBaseDataType, ADWriterFactory, NDPluginFileIO
from ophyd_async.epics.adkinetix import KinetixDetector


@pytest.fixture
def kinetix_det_factory(
    static_path_provider: StaticPathProvider,
) -> Callable[[int], KinetixDetector]:
    def _factory(num: int, with_tiff: bool = False) -> KinetixDetector:
        with init_devices(mock=True):
            ktx = KinetixDetector(
                f"KTX{num}",
                ADWriterFactory.tiff(static_path_provider)
                if with_tiff
                else ADWriterFactory.hdf(static_path_provider),
                name=f"kinetix{num}",
            )
            file_plugin = ktx.get_plugin("tiff" if with_tiff else "hdf", NDPluginFileIO)

        # what the descriptor needs to describe the image
        set_mock_value(ktx.driver.array_size_x, 3200)
        set_mock_value(ktx.driver.array_size_y, 3200)
        set_mock_value(ktx.driver.data_type, ADBaseDataType.UINT16)

        async def _mock_increment_num_captured(_):
            current_num_cap = await file_plugin.num_captured.get_value()
            set_mock_value(file_plugin.num_captured, current_num_cap + 1)

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
