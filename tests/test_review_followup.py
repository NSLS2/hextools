"""GeRM data-lifecycle regression requested in the operator-fix review."""

import pytest
from ophyd_async.core import EnableDisable, set_mock_value


@pytest.mark.parametrize("file_format", ["hdf", "tiff"])
async def test_direct_germ_trigger_enables_stats(germ, file_format):
    if file_format == "tiff":
        germ.save_as_tiff()
    set_mock_value(germ.stats1.enable_callbacks, EnableDisable.DISABLE)
    set_mock_value(germ.stats1.compute_statistics, False)
    await germ.stage()
    try:
        await germ.trigger()
        assert await germ.stats1.enable_callbacks.get_value() == EnableDisable.ENABLE
        assert await germ.stats1.compute_statistics.get_value() is True
    finally:
        await germ.unstage()
