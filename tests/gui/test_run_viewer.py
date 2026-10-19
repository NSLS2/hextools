import os
import threading

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
        if isinstance(self.data, Exception):
            raise self.data
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
        # Lookups wait while the gate is closed, as a slow Tiled would.
        self.gate = threading.Event()
        self.gate.set()
        self.lookups = []

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
        self.gate.wait(5)
        self.lookups.append(key)
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


class GatedArray(FakeArray):
    """An array whose reads wait while the gate is closed, as a slow Tiled would."""

    def __init__(self, data):
        super().__init__(data)
        self.gate = threading.Event()
        self.gate.set()

    def __getitem__(self, key):
        self.gate.wait(5)
        return super().__getitem__(key)


def _open_gated_image(monkeypatch):
    QApplication.instance() or QApplication([])
    run = _tomo_run()
    array = GatedArray(PROJECTIONS)
    run["primary"].arrays["kinetix-det1_image"] = array
    viewer = QtRunViewer(FakeClient({"abc": run}), "abc")
    _wait_for(lambda: viewer._datasets)
    _select(viewer, "kinetix-det1_image")
    _wait_for(lambda: viewer._image is not None and viewer._frame_worker is None)
    shown = []
    set_image = viewer._image_item.setImage

    def spy(image, **kwargs):
        shown.append(np.array(image))
        set_image(image, **kwargs)

    monkeypatch.setattr(viewer._image_item, "setImage", spy)
    return viewer, array, shown


# PR 96 review: a frame read finishing after the controls moved on was shown,
# so the image and statistics disagreed with the slider.
def test_viewer_drops_a_frame_the_controls_moved_past(monkeypatch):
    viewer, array, shown = _open_gated_image(monkeypatch)
    array.gate.clear()
    viewer._index.setValue(1)
    _wait_for(lambda: viewer._frame_worker is not None)
    viewer._index.setValue(2)
    array.gate.set()
    _wait_for(lambda: viewer._frame_worker is None and not viewer._frame_pending)
    _wait_for(lambda: len(shown) >= 1)
    assert [frame[0, 0] for frame in shown] == [PROJECTIONS[2, 0, 0, 0]]
    np.testing.assert_array_equal(viewer._image, PROJECTIONS[2, 0])
    viewer.close()


# PR 96 review: after a slice-axis change, a stale projection used up the
# first-frame reset, leaving the sinogram with the projection's contrast.
def test_viewer_resets_contrast_for_the_new_axis_not_a_stale_frame(monkeypatch):
    viewer, array, _shown = _open_gated_image(monkeypatch)
    array.gate.clear()
    viewer._index.setValue(1)
    _wait_for(lambda: viewer._frame_worker is not None)
    viewer._slice_axis.setCurrentIndex(1)  # slice along rows: a sinogram
    array.gate.set()
    _wait_for(lambda: viewer._frame_worker is None and not viewer._frame_pending)
    sinogram = PROJECTIONS[:, 0, 0, :]
    np.testing.assert_array_equal(viewer._image, sinogram)
    assert tuple(viewer._image_item.getLevels()) == pytest.approx(auto_levels(sinogram))
    viewer.close()


# PR 96 review: a text column (e.g. a shutter's "Open"/"Closed") raised in the
# plotting callback and left the previous plot under the new label.
@pytest.mark.parametrize("axis", ["y", "x"])
def test_viewer_refuses_to_plot_text_values(axis):
    QApplication.instance() or QApplication([])
    run = _tomo_run()
    stream = run["primary"]
    stream._columns["shutter"] = np.array(["Open", "Closed", "Open"])
    stream.metadata["data_keys"]["shutter"] = {"dtype": "string"}
    viewer = QtRunViewer(FakeClient({"abc": run}), "abc")
    _wait_for(lambda: viewer._datasets)
    _select(viewer, "angle")
    _wait_for(lambda: viewer._line_curve.getData()[1] is not None)

    if axis == "y":
        _select(viewer, "shutter")
    else:
        viewer._x_key.setCurrentText("shutter")
    _wait_for(lambda: "not numbers" in viewer._status.text())
    if axis == "y":
        assert viewer._line_curve.getData()[1] is None
    else:
        # Plotted against the index instead.
        np.testing.assert_array_equal(viewer._line_curve.getData()[0], [0, 1, 2])
        np.testing.assert_array_equal(viewer._line_curve.getData()[1], ANGLES)
    viewer.close()


# (events, frames per event, rows, columns): averaging takes several frames per event.
AVERAGED = np.arange(3 * 2 * 4 * 5, dtype="uint16").reshape(3, 2, 4, 5)


def test_frame_key_reaches_every_axis_of_an_averaged_stack():
    shape = AVERAGED.shape
    assert slice_axes(shape) == [0, 1, 2, 3]
    assert frame_key(shape, 0, 2, {1: 1}) == (2, 1, slice(None), slice(None))
    # A sinogram runs across events, with the frame within each event held.
    assert frame_key(shape, 2, 3, {1: 1}) == (slice(None), 1, 3, slice(None))


def _averaged_run():
    run = _tomo_run()
    run["primary"].arrays["kinetix-det1_image"] = FakeArray(AVERAGED)
    run["primary"].arrays["roi"] = FakeArray(np.ones((4, 5)))
    return run


# PR 96 review: with several frames per event, only the first event's frames
# could be reached; the event axis was always read at 0.
def test_viewer_navigates_events_and_frames_of_an_averaged_stack():
    QApplication.instance() or QApplication([])
    viewer = QtRunViewer(FakeClient({"abc": _averaged_run()}), "abc")
    _wait_for(lambda: viewer._datasets)
    _select(viewer, "kinetix-det1_image")
    _wait_for(lambda: viewer._image is not None)
    assert list(viewer._other_axes) == [1]
    viewer._other_axes[1].setValue(1)
    viewer._index.setValue(2)
    _wait_for(
        lambda: (
            viewer._image is not None and viewer._image[0, 0] == AVERAGED[2, 1, 0, 0]
        )
    )
    viewer.close()


# PR 96 review: the previous stack's slice-axis choices stayed live while the
# next dataset loaded, and picking axis 2 for a 2D array raised IndexError.
def test_viewer_disables_slicing_while_the_next_dataset_loads(monkeypatch):
    QApplication.instance() or QApplication([])
    errors = []
    monkeypatch.setattr("sys.excepthook", lambda *exc: errors.append(exc[1]))
    run = _averaged_run()
    viewer = QtRunViewer(FakeClient({"abc": run}), "abc")
    _wait_for(lambda: viewer._datasets)
    _select(viewer, "kinetix-det1_image")
    _wait_for(lambda: viewer._image is not None and viewer._frame_worker is None)

    run["primary"].gate.clear()
    _select(viewer, "roi")
    assert not viewer._slice_axis.isEnabled()
    viewer._slice_axis.setCurrentIndex(2)
    run["primary"].gate.set()
    _wait_for(lambda: viewer._image is not None and viewer._image.shape == (4, 5))
    assert errors == []
    viewer.close()


# PR 96 review: a failed read left the previous dataset's curve and statistics
# on screen under the new dataset's name.
def test_viewer_clears_the_old_plot_when_a_read_fails():
    QApplication.instance() or QApplication([])
    run = _tomo_run()
    run["primary"]._columns["broken"] = RuntimeError("502 Bad Gateway")
    viewer = QtRunViewer(FakeClient({"abc": run}), "abc")
    _wait_for(lambda: viewer._datasets)
    _select(viewer, "angle")
    _wait_for(lambda: viewer._line_curve.getData()[1] is not None)

    _select(viewer, "broken")
    _wait_for(lambda: "502" in viewer._status.text())
    assert viewer._line_curve.getData()[1] is None
    assert viewer._stats_table.rowCount() == 0
    viewer.close()


# PR 96 review: each finished column read replotted the current selection, and
# started a second read of a column already being read.
def test_viewer_reads_each_column_once():
    QApplication.instance() or QApplication([])
    run = _tomo_run()
    stream = run["primary"]
    viewer = QtRunViewer(FakeClient({"abc": run}), "abc")
    _wait_for(lambda: viewer._datasets)

    stream.gate.clear()
    _select(viewer, "angle")
    _select(viewer, "time")
    stream.gate.set()
    _wait_for(lambda: viewer._line_curve.getData()[1] is not None)
    _wait_for(lambda: len(stream.lookups) >= 2)
    for _ in range(20):
        QApplication.processEvents()
    assert sorted(stream.lookups) == ["angle", "time"]
    np.testing.assert_array_equal(viewer._line_curve.getData()[1], [10.0, 11.0, 12.0])
    viewer.close()
