import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
pytest.importorskip("pyqtgraph")

from bluesky import RunEngine  # noqa: E402
from qtpy.QtCore import QElapsedTimer  # noqa: E402
from qtpy.QtWidgets import QApplication, QTreeWidgetItem  # noqa: E402
from tiled.structures.core import Spec  # noqa: E402

from hextools.gui.run_viewer import (  # noqa: E402
    DatasetInfo,
    QtRunViewer,
    auto_levels,
    frame_key,
    image_statistics,
    list_datasets,
    percentile_density,
    run_streams,
    slice_axes,
)
from hextools.gui.scan_browser import QtScanBrowser, search_runs  # noqa: E402


class FakeArray:
    """Tiled array client: indexing reads a slice, ``read()`` reads it all."""

    def __init__(self, data):
        self.data = data
        self.keys_read = []

    def __getitem__(self, key):
        self.keys_read.append(key)
        return self.data[key]

    def read(self):
        return self.data


class FakeSpec:
    def __init__(self, name):
        self.name = name


class FakeStream:
    """An event stream: one table of per-event columns plus external arrays."""

    specs = (FakeSpec("BlueskyEventStream"),)

    def __init__(self, columns, arrays):
        self.metadata = {"data_keys": {key: {"dtype": "number"} for key in columns}}
        self._columns = {
            "seq_num": np.arange(1, 4),
            "ts_angle": np.zeros(3),
            **columns,
        }
        self.arrays = {key: FakeArray(value) for key, value in arrays.items()}

    def get_contents(self):
        contents = {
            "internal": {
                "attributes": {
                    "structure_family": "table",
                    "structure": {"columns": list(self._columns)},
                }
            }
        }
        for key, array in self.arrays.items():
            contents[key] = {
                "attributes": {
                    "structure_family": "array",
                    "structure": {
                        "shape": list(array.data.shape),
                        "data_type": {
                            "kind": array.data.dtype.kind,
                            "itemsize": array.data.dtype.itemsize,
                        },
                    },
                }
            }
        return contents

    def __getitem__(self, key):
        if key in self.arrays:
            return self.arrays[key]
        return FakeArray(self._columns[key])


class FakeRun(dict):
    def __init__(self, streams, start, nested=False):
        super().__init__({"streams": FakeNamespace(streams)} if nested else streams)
        self.metadata = {"start": start, "stop": {"exit_status": "success"}}
        self.specs = [FakeSpec("BlueskyRun")]


class FakeNamespace(dict):
    specs = ()


class FakeResults:
    def __init__(self, runs):
        self._runs = runs

    def search(self, _query):
        return self

    def items(self):
        return list(self._runs.items())


class FakeClient:
    def __init__(self, runs):
        self._runs = runs

    def __getitem__(self, uid):
        return self._runs[uid]

    def search(self, _query):
        return FakeResults(self._runs)


# Projections are (events, frames per event, rows, columns) as Kinetix writes them.
PROJECTIONS = np.arange(3 * 1 * 4 * 5, dtype="uint16").reshape(3, 1, 4, 5)
ANGLES = np.array([0.0, 90.0, 180.0])


def _tomo_run(nested=False, detectors=("kinetix-det1",)):
    stream = FakeStream(
        columns={"time": np.array([10.0, 11.0, 12.0]), "angle": ANGLES},
        arrays={"kinetix-det1_image": PROJECTIONS},
    )
    start = {"uid": "abc", "scan_id": 7, "time": 100.0, "detectors": list(detectors)}
    return FakeRun({"primary": stream}, start, nested=nested)


def _wait_for(condition, timeout_ms=5000):
    app = QApplication.instance() or QApplication([])
    timer = QElapsedTimer()
    timer.start()
    while not condition():
        app.processEvents()
        assert timer.elapsed() < timeout_ms, "timed out waiting for the viewer"


def _select(viewer, key):
    root = viewer._dataset_tree.invisibleRootItem()
    for i in range(root.childCount()):
        stream_item = root.child(i)
        for j in range(stream_item.childCount()):
            item: QTreeWidgetItem = stream_item.child(j)
            if item.text(0) == key:
                viewer._dataset_tree.setCurrentItem(item)
                return
    raise AssertionError(f"{key} not listed")


def test_frame_key_reads_one_image_and_skips_unit_axes():
    shape = (3, 1, 4, 5)
    assert slice_axes(shape) == [0, 2, 3]
    assert frame_key(shape, 0, 2) == (2, 0, slice(None), slice(None))
    # Slicing along a row gives a sinogram: events by columns.
    assert frame_key(shape, 2, 1) == (slice(None), 0, 1, slice(None))
    assert PROJECTIONS[frame_key(shape, 2, 1)].shape == (3, 5)
    assert slice_axes((4, 5)) == []
    assert frame_key((1, 4, 5)) == (0, slice(None), slice(None))


@pytest.mark.parametrize(
    ("shape", "axis"), [((5,), None), ((1, 5), None), ((3, 4, 5), 1 + 5)]
)
def test_frame_key_refuses_what_is_not_an_image(shape, axis):
    with pytest.raises(ValueError):
        frame_key(shape, axis)


def test_image_statistics_and_levels():
    data = np.array([[0.0, 1.0], [2.0, np.nan]])
    stats = image_statistics(data)
    assert stats["Minimum"] == 0.0
    assert stats["Maximum"] == 2.0
    assert stats["Mean"] == pytest.approx(1.0)
    assert stats["Median"] == 1.0
    assert image_statistics(np.full(3, np.nan)) == {}
    assert auto_levels(np.full((2, 2), 5.0)) == (5.0, 6.0)


def test_percentile_density_is_normalised_and_drops_empty_bins():
    percentiles, density = percentile_density(np.arange(1000.0))
    assert density.sum() == pytest.approx(1.0)
    assert percentiles[0] == pytest.approx(0.5)
    assert np.allclose(density, density[0], rtol=0.05)
    assert percentile_density(np.zeros(10))[0].size == 0


def test_list_datasets_reads_columns_and_arrays_in_one_listing():
    stream = _tomo_run()["primary"]
    datasets = list_datasets("primary", stream)
    assert datasets == [
        DatasetInfo("primary", "time", None, "number"),
        DatasetInfo("primary", "angle", None, "number"),
        DatasetInfo("primary", "kinetix-det1_image", (3, 1, 4, 5), "u2"),
    ]
    assert [d.is_image for d in datasets] == [False, False, True]


@pytest.mark.parametrize("nested", [False, True])
def test_run_streams_handles_both_tiled_layouts(nested):
    assert list(run_streams(_tomo_run(nested=nested))) == ["primary"]


def test_search_runs_filters_by_detector_name():
    client = FakeClient(
        {
            "a": _tomo_run(detectors=("kinetix-det1",)),
            "b": _tomo_run(detectors=("phantom",)),
            "c": _tomo_run(detectors=()),
        }
    )
    spec = Spec("TomoFlyscan")
    assert [r.uid for r in search_runs(client, spec, 0, 1, None, "Kinetix")] == ["a"]
    assert len(search_runs(client, spec, 0, 1, None, "")) == 3


def test_viewer_slices_a_stack_read_from_tiled():
    QApplication.instance() or QApplication([])
    run = _tomo_run()
    viewer = QtRunViewer(FakeClient({"abc": run}), "abc")
    _wait_for(lambda: viewer._datasets)
    assert "Scan 7" in viewer.windowTitle()

    _select(viewer, "kinetix-det1_image")
    _wait_for(lambda: viewer._image is not None)
    np.testing.assert_array_equal(viewer._image, PROJECTIONS[0, 0])
    assert viewer._stats_table.rowCount() > 2

    viewer._index.setValue(2)
    _wait_for(lambda: viewer._image is not None and viewer._image[0, 0] == 40)
    array = run["primary"].arrays["kinetix-det1_image"]
    # Only single frames were requested, never the whole stack.
    assert array.keys_read == [
        (0, 0, slice(None), slice(None)),
        (2, 0, slice(None), slice(None)),
    ]

    viewer._profile_direction.setCurrentIndex(1)
    xs, ys = viewer._profile_curve.getData()
    assert len(ys) == PROJECTIONS.shape[2]
    viewer.close()


def test_viewer_plots_a_column_against_another():
    QApplication.instance() or QApplication([])
    viewer = QtRunViewer(FakeClient({"abc": _tomo_run()}), "abc")
    _wait_for(lambda: viewer._datasets)

    _select(viewer, "angle")
    _wait_for(lambda: viewer._line_curve.getData()[1] is not None)
    np.testing.assert_array_equal(viewer._line_curve.getData()[1], ANGLES)

    viewer._x_key.setCurrentText("time")
    _wait_for(lambda: viewer._line_curve.getData()[0][-1] == 2.0)
    viewer.close()


def test_scan_browser_opens_a_run_in_the_viewer():
    QApplication.instance() or QApplication([])
    client = FakeClient({"abc": _tomo_run()})
    browser = QtScanBrowser(client, [Spec("TomoFlyscan")], RunEngine({}))
    viewer = browser.open_viewer("abc")
    assert viewer.parent() is browser
    assert viewer.isWindow()
    _wait_for(lambda: viewer._datasets)
    viewer.close()
