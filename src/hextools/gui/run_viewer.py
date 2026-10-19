"""Review one Bluesky run's data, read from Tiled, in its own window.

The feature set follows DatView (https://github.com/algotom/datview): zoom, line
profiles, contrast, histogram, percentile density, image statistics and slicing
through a stack, plus 1D plots of scalar data and the run's metadata.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from hextools.gui._threading import create_worker

#: Bookkeeping columns of an event stream that are not data.
_SKIPPED_KEYS = {"seq_num"}
_INDEX = "Index"


@dataclass(frozen=True)
class DatasetInfo:
    """One dataset of a run's event stream, as listed in the viewer."""

    stream: str
    key: str
    #: Full shape for arrays; ``None`` for table columns (one value per event).
    shape: tuple[int, ...] | None
    dtype: str

    @property
    def is_image(self) -> bool:
        """Whether the dataset has at least two axes longer than one."""
        return self.shape is not None and len(_long_axes(self.shape)) >= 2


def fill_metadata_tree(parent: QTreeWidgetItem, value: Any) -> None:
    """Add ``value`` (nested dicts and lists) under ``parent``, one row per key."""
    items = value.items() if isinstance(value, dict) else enumerate(value)
    for key, child in items:
        item = QTreeWidgetItem(parent, [str(key)])
        if isinstance(child, (dict, list)) and child:
            fill_metadata_tree(item, child)
        else:
            item.setText(
                1, child if isinstance(child, str) else json.dumps(child, default=str)
            )


def run_streams(run: Any) -> dict[str, Any]:
    """Return the run's event streams by name, for either Tiled run layout."""
    container = run
    # Older runs keep their streams under a "streams" namespace.
    if "streams" in run and "BlueskyEventStream" not in {
        s.name for s in run["streams"].specs
    }:
        container = run["streams"]
    return {name: container[name] for name in container}


def list_datasets(stream_name: str, stream: Any) -> list[DatasetInfo]:
    """List a stream's datasets in one request; ``stream[key]`` re-lists each time."""
    data_keys = stream.metadata.get("data_keys", {})
    datasets = []
    for key, item in stream.get_contents().items():
        attributes = item["attributes"]
        structure = attributes["structure"]
        if attributes["structure_family"] == "table":
            for column in structure["columns"]:
                if column in _SKIPPED_KEYS or column.startswith("ts_"):
                    continue
                dtype = data_keys.get(column, {}).get("dtype", "")
                datasets.append(DatasetInfo(stream_name, column, None, dtype))
        else:
            data_type = structure.get("data_type") or {}
            dtype = f"{data_type.get('kind', '')}{data_type.get('itemsize', '')}"
            datasets.append(
                DatasetInfo(stream_name, key, tuple(structure["shape"]), dtype)
            )
    return datasets


def load_run(client: Any, uid: str) -> tuple[Any, dict[str, Any], list[DatasetInfo]]:
    """Fetch run ``uid`` from ``client`` with its streams and their datasets."""
    run = client[uid]
    streams = run_streams(run)
    datasets = [
        info for name, stream in streams.items() for info in list_datasets(name, stream)
    ]
    return run, streams, datasets


def _long_axes(shape: tuple[int, ...]) -> list[int]:
    return [axis for axis, length in enumerate(shape) if length > 1]


def slice_axes(shape: tuple[int, ...]) -> list[int]:
    """Axes a stack can be sliced along: all axes longer than one, if three or more."""
    long_axes = _long_axes(shape)
    return long_axes if len(long_axes) >= 3 else []


def shown_axes(shape: tuple[int, ...], slice_axis: int | None = None) -> list[int]:
    """Return the two axes shown as the image when slicing along ``slice_axis``.

    Slicing a stack axis shows the last two (the detector image). Slicing an image
    axis shows the first stack axis against the other image axis: a sinogram.
    """
    long_axes = _long_axes(shape)
    if len(long_axes) < 2:
        raise ValueError(f"Shape {shape} has fewer than two axes longer than one")
    stack, image = long_axes[:-2], long_axes[-2:]
    if slice_axis is None or slice_axis in stack:
        return image
    if slice_axis in image and stack:
        return [stack[0], next(axis for axis in image if axis != slice_axis)]
    raise ValueError(f"Cannot slice shape {shape} along axis {slice_axis}")


def held_axes(shape: tuple[int, ...], slice_axis: int | None = None) -> list[int]:
    """Axes longer than one, neither sliced nor shown; each is held at an index."""
    shown = shown_axes(shape, slice_axis)
    return [
        axis for axis in _long_axes(shape) if axis != slice_axis and axis not in shown
    ]


def frame_key(
    shape: tuple[int, ...],
    slice_axis: int | None = None,
    index: int = 0,
    held: dict[int, int] | None = None,
) -> tuple[int | slice, ...]:
    """Index that reads one 2D image out of an array of ``shape``.

    The image is ``index`` along ``slice_axis`` (one of :func:`slice_axes`), or the
    whole array when only two axes are longer than one. Each of :func:`held_axes`
    is read at its index in ``held`` (default 0); axes of length one at 0.
    """
    shown = shown_axes(shape, slice_axis)
    held = held or {}
    key: list[int | slice] = []
    for axis in range(len(shape)):
        if axis == slice_axis:
            key.append(index)
        elif axis in shown:
            key.append(slice(None))
        else:
            key.append(held.get(axis, 0))
    return tuple(key)


def _finite(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=float).ravel()
    return values[np.isfinite(values)]


def image_statistics(data: np.ndarray) -> dict[str, float]:
    """Minimum, maximum, mean, median, standard deviation and outer percentiles."""
    values = _finite(data)
    if values.size == 0:
        return {}
    p1, p5, p95, p99 = np.percentile(values, [1, 5, 95, 99])
    return {
        "Minimum": float(values.min()),
        "Maximum": float(values.max()),
        "Mean": float(values.mean()),
        "Median": float(np.median(values)),
        "Std. deviation": float(values.std()),
        "1st percentile": float(p1),
        "5th percentile": float(p5),
        "95th percentile": float(p95),
        "99th percentile": float(p99),
    }


def percentile_density(
    data: np.ndarray, bins: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Density of values in each percentile bin, normalised to sum to one.

    Returns the bin-centre percentiles and densities, dropping bins of zero width.
    """
    values = _finite(data)
    if values.size == 0:
        return np.empty(0), np.empty(0)
    percentiles = np.linspace(0, 100, bins + 1)
    edges = np.percentile(values, percentiles)
    counts, _ = np.histogram(values, bins=edges)
    widths = np.diff(edges)
    keep = widths > 0
    density = counts[keep] / widths[keep]
    if density.sum() > 0:
        density = density / density.sum()
    centres = percentiles[:-1] + 50 / bins
    return centres[keep], density


def auto_levels(
    data: np.ndarray, low: float = 1, high: float = 99
) -> tuple[float, float]:
    """Display levels that clip the lowest and highest percent of values."""
    values = _finite(data)
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = (float(v) for v in np.percentile(values, [low, high]))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def _numeric(values: np.ndarray) -> np.ndarray | None:
    """``values`` as floats, or ``None`` if they are not numbers (e.g. "Open")."""
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None


# Worker functions return what they were asked for alongside the result, so a
# result that arrives after the user moved on can be recognised and dropped.


def _fetch_array(stream: Any, info: DatasetInfo) -> tuple[DatasetInfo, Any]:
    return info, stream[info.key]


def _read_frame(
    array: Any, key: tuple[int | slice, ...]
) -> tuple[Any, tuple[int | slice, ...], np.ndarray]:
    return array, key, np.asarray(array[key])


def _read_columns(
    stream: Any, stream_name: str, keys: list[str]
) -> tuple[str, list[str], dict[tuple[str, str], np.ndarray], Exception | None]:
    # A failure is returned, not raised, so the caller can release these keys.
    try:
        columns = {
            (stream_name, key): np.asarray(stream[key].read()).ravel() for key in keys
        }
    except Exception as error:
        return stream_name, keys, {}, error
    return stream_name, keys, columns, None


class QtRunViewer(QWidget):
    """A window reviewing one run read from Tiled.

    Datasets with two or more long axes are shown as images, sliced along a chosen
    axis when there are three; the rest are plotted against the event index or
    another dataset of the same stream.
    """

    def __init__(self, client: Any, uid: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"Run {uid}")
        self.resize(1300, 850)
        self._uid = uid
        self._streams: dict[str, Any] = {}
        self._datasets: list[DatasetInfo] = []
        self._dataset: DatasetInfo | None = None
        self._array: Any = None
        self._image: np.ndarray | None = None
        self._columns: dict[tuple[str, str], np.ndarray] = {}
        self._columns_loading: set[tuple[str, str]] = set()
        self._other_axes: dict[int, QSpinBox] = {}
        self._frame_worker: Any = None
        self._frame_pending = False

        self._status = QLabel("Loading run…")

        self._dataset_tree = QTreeWidget()
        self._dataset_tree.setHeaderLabels(["Dataset", "Shape", "Type"])
        self._dataset_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self._dataset_tree.currentItemChanged.connect(self._on_dataset_selected)
        self._metadata_tree = QTreeWidget()
        self._metadata_tree.setHeaderLabels(["Key", "Value"])
        self._metadata_tree.header().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        side_tabs = QTabWidget()
        side_tabs.addTab(self._dataset_tree, "Datasets")
        side_tabs.addTab(self._metadata_tree, "Metadata")

        self._views = QStackedWidget()
        self._views.addWidget(self._build_image_page())
        self._views.addWidget(self._build_plot_page())

        self._analysis = QTabWidget()
        self._profile_plot = pg.PlotWidget()
        self._profile_curve = self._profile_plot.plot(pen="y")
        self._histogram_plot = pg.PlotWidget()
        self._histogram_plot.setLabel("bottom", "Value")
        self._histogram_plot.setLabel("left", "Count")
        self._histogram_curve = self._histogram_plot.plot(
            stepMode="center", fillLevel=0, brush=(80, 140, 220, 120)
        )
        self._density_plot = pg.PlotWidget()
        self._density_plot.setLabel("bottom", "Percentile")
        self._density_plot.setLabel("left", "Density")
        self._density_curve = self._density_plot.plot(pen="c")
        self._stats_table = QTableWidget(0, 2)
        self._stats_table.setHorizontalHeaderLabels(["Statistic", "Value"])
        self._stats_table.verticalHeader().setVisible(False)
        self._stats_table.horizontalHeader().setStretchLastSection(True)
        self._analysis.addTab(self._profile_plot, "Line profile")
        self._analysis.addTab(self._histogram_plot, "Histogram")
        self._analysis.addTab(self._density_plot, "Percentile density")
        self._analysis.addTab(self._stats_table, "Statistics")

        centre = QSplitter(Qt.Orientation.Vertical)
        centre.addWidget(self._views)
        centre.addWidget(self._analysis)
        centre.setStretchFactor(0, 3)
        centre.setStretchFactor(1, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(side_tabs)
        splitter.addWidget(centre)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([320, 980])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self._status)

        self._start(load_run, client, uid, returned=self._on_run_loaded)

    def _build_image_page(self) -> QWidget:
        self._slice_axis = QComboBox()
        self._slice_axis.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self._slice_axis.currentIndexChanged.connect(self._on_slice_axis_changed)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._index = QSpinBox()
        self._slider.valueChanged.connect(self._index.setValue)
        self._index.valueChanged.connect(self._slider.setValue)
        self._index.valueChanged.connect(self._request_frame)
        self._profile_direction = QComboBox()
        self._profile_direction.addItems(["Horizontal", "Vertical"])
        self._profile_direction.currentIndexChanged.connect(self._on_profile_direction)
        self._zoom_box = QCheckBox("Zoom box")
        self._zoom_box.toggled.connect(self._on_zoom_box)
        reset_zoom = QPushButton("Reset zoom")
        reset_zoom.clicked.connect(lambda: self._image_view.autoRange())
        auto_contrast = QPushButton("Auto contrast")
        auto_contrast.clicked.connect(self._auto_contrast)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Slice axis:"))
        controls.addWidget(self._slice_axis)
        controls.addWidget(self._slider, stretch=1)
        controls.addWidget(self._index)
        self._held_bar = QHBoxLayout()
        controls.addLayout(self._held_bar)
        controls.addWidget(QLabel("Profile:"))
        controls.addWidget(self._profile_direction)
        controls.addWidget(self._zoom_box)
        controls.addWidget(reset_zoom)
        controls.addWidget(auto_contrast)

        plot = pg.PlotWidget()
        self._image_view = plot.getViewBox()
        self._image_view.setAspectLocked(True)
        self._image_view.invertY(True)
        self._image_item = pg.ImageItem(axisOrder="row-major")
        plot.addItem(self._image_item)
        self._profile_line = pg.InfiniteLine(angle=0, movable=True, pen="y")
        self._profile_line.sigPositionChanged.connect(self._update_profile)
        plot.addItem(self._profile_line)
        self._contrast = pg.HistogramLUTWidget(image=self._image_item)

        image_row = QHBoxLayout()
        image_row.addWidget(plot, stretch=1)
        image_row.addWidget(self._contrast)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addLayout(image_row, stretch=1)
        return page

    def _build_plot_page(self) -> QWidget:
        self._x_key = QComboBox()
        self._x_key.currentIndexChanged.connect(self._on_x_key_changed)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("X axis:"))
        controls.addWidget(self._x_key)
        controls.addStretch()
        self._line_plot = pg.PlotWidget()
        self._line_plot.showGrid(x=True, y=True, alpha=0.3)
        self._line_curve = self._line_plot.plot(pen="y", symbol="o", symbolSize=4)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(controls)
        layout.addWidget(self._line_plot, stretch=1)
        return page

    def _start(
        self,
        func: Callable,
        *args: Any,
        returned: Callable[[Any], None],
        finished: Callable[[], None] | None = None,
    ) -> Any:
        """Run ``func`` off the GUI thread; ``returned`` gets its result.

        Pass bound methods: a lambda would run in the worker's thread.
        """
        connect: dict[str, Callable | Sequence[Callable]] = {
            "returned": returned,
            "errored": self._on_error,
        }
        if finished is not None:
            connect["finished"] = finished
        return create_worker(func, *args, _connect=connect)

    def _on_error(self, error: Exception) -> None:
        self._status.setText(f"Error: {type(error).__name__}: {error}")

    def _on_run_loaded(
        self, result: tuple[Any, dict[str, Any], list[DatasetInfo]]
    ) -> None:
        run, self._streams, self._datasets = result
        metadata = dict(run.metadata)
        start = metadata.get("start", {})
        self.setWindowTitle(
            f"Scan {start.get('scan_id', '?')} — {start.get('plan_name', '')}"
            f" — {self._uid}"
        )
        fill_metadata_tree(self._metadata_tree.invisibleRootItem(), metadata)
        stream_items: dict[str, QTreeWidgetItem] = {}
        for info in self._datasets:
            parent = stream_items.get(info.stream)
            if parent is None:
                parent = QTreeWidgetItem(self._dataset_tree, [info.stream])
                parent.setExpanded(True)
                stream_items[info.stream] = parent
            shape = "per event" if info.shape is None else str(info.shape)
            item = QTreeWidgetItem(parent, [info.key, shape, info.dtype])
            item.setData(0, Qt.ItemDataRole.UserRole, info)
        self._status.setText(
            f"{len(self._datasets)} dataset(s) in {len(self._streams)} stream(s). "
            "Click one to view it."
        )

    def _on_dataset_selected(
        self, item: QTreeWidgetItem | None, _previous: QTreeWidgetItem | None = None
    ) -> None:
        info = None if item is None else item.data(0, Qt.ItemDataRole.UserRole)
        if not isinstance(info, DatasetInfo):
            return
        self._dataset = info
        self._array = None
        self._reset_for_selection()
        self._status.setText(f"Loading {info.stream}/{info.key}…")
        if info.is_image:
            stream = self._streams[info.stream]
            self._start(_fetch_array, stream, info, returned=self._on_array_ready)
        else:
            self._show_plot_page(info)

    def _reset_for_selection(self) -> None:
        """Clear the previous dataset's display and disable its controls."""
        self._slice_axis.blockSignals(True)
        self._slice_axis.clear()
        self._slice_axis.blockSignals(False)
        self._slice_axis.setEnabled(False)
        for widget in (self._slider, self._index):
            widget.setEnabled(False)
        self._set_held_axes([])
        self._image = None
        self._image_item.clear()
        self._profile_curve.clear()
        self._line_curve.clear()
        self._clear_analysis()

    def _set_held_axes(self, axes: list[int]) -> None:
        while self._held_bar.count():
            item = self._held_bar.takeAt(0)
            widget = None if item is None else item.widget()
            if widget is not None:
                widget.deleteLater()
        self._other_axes = {}
        shape = None if self._dataset is None else self._dataset.shape
        for axis in axes:
            spin = QSpinBox()
            spin.setRange(0, (shape[axis] if shape else 1) - 1)
            spin.valueChanged.connect(self._request_frame)
            self._held_bar.addWidget(QLabel(f"Axis {axis}:"))
            self._held_bar.addWidget(spin)
            self._other_axes[axis] = spin

    # Images

    def _on_array_ready(self, result: tuple[DatasetInfo, Any]) -> None:
        info, array = result
        if info != self._dataset or info.shape is None:
            return
        self._array = array
        self._image = None
        self._views.setCurrentIndex(0)
        axes = slice_axes(info.shape)
        self._slice_axis.blockSignals(True)
        self._slice_axis.clear()
        for axis in axes:
            self._slice_axis.addItem(f"{axis} (length {info.shape[axis]})", axis)
        self._slice_axis.blockSignals(False)
        self._slice_axis.setEnabled(bool(axes))
        self._on_slice_axis_changed()

    def _on_slice_axis_changed(self, _index: int = 0) -> None:
        axis = self._slice_axis.currentData()
        shape = None if self._dataset is None else self._dataset.shape
        length = 1 if axis is None or shape is None else shape[axis]
        self._image = None
        self._set_held_axes([] if shape is None else held_axes(shape, axis))
        for widget in (self._slider, self._index):
            widget.blockSignals(True)
            widget.setRange(0, length - 1)
            widget.setValue(0)
            widget.blockSignals(False)
            widget.setEnabled(length > 1)
        self._request_frame()

    def _current_frame_key(self) -> tuple[int | slice, ...] | None:
        if self._array is None or self._dataset is None or self._dataset.shape is None:
            return None
        return frame_key(
            self._dataset.shape,
            self._slice_axis.currentData(),
            self._index.value(),
            {axis: spin.value() for axis, spin in self._other_axes.items()},
        )

    def _request_frame(self, _value: int = 0) -> None:
        """Read the selected image; while a read runs, only the latest request waits."""
        key = self._current_frame_key()
        if key is None:
            return
        if self._frame_worker is not None:
            self._frame_pending = True
            return
        self._frame_worker = self._start(
            _read_frame,
            self._array,
            key,
            returned=self._show_image,
            finished=self._on_frame_finished,
        )

    def _on_frame_finished(self) -> None:
        self._frame_worker = None
        if self._frame_pending:
            self._frame_pending = False
            self._request_frame()

    def _show_image(
        self, result: tuple[Any, tuple[int | slice, ...], np.ndarray]
    ) -> None:
        array, key, image = result
        info = self._dataset
        # Matching the array alone let a frame the slider had moved past through.
        if array is not self._array or key != self._current_frame_key() or info is None:
            return
        first = self._image is None
        self._image = image
        self._image_item.setImage(image, autoLevels=False)
        if first:
            self._image_item.setLevels(auto_levels(image))
            self._image_view.autoRange()
            vertical = self._profile_direction.currentIndex() == 1
            self._profile_line.setValue(image.shape[1 if vertical else 0] // 2)
        self._update_profile()
        self._update_analysis(image)
        self._status.setText(
            f"{info.stream}/{info.key}: image {image.shape}, {image.dtype}"
        )

    def _auto_contrast(self) -> None:
        if self._image is not None:
            self._image_item.setLevels(auto_levels(self._image))

    def _on_zoom_box(self, checked: bool) -> None:
        mode = pg.ViewBox.RectMode if checked else pg.ViewBox.PanMode
        self._image_view.setMouseMode(mode)

    def _on_profile_direction(self, index: int) -> None:
        vertical = index == 1
        self._profile_line.setAngle(90 if vertical else 0)
        if self._image is not None:
            self._profile_line.setValue(self._image.shape[1 if vertical else 0] // 2)
        self._update_profile()

    def _update_profile(self) -> None:
        if self._image is None:
            return
        vertical = self._profile_direction.currentIndex() == 1
        limit = self._image.shape[1 if vertical else 0] - 1
        position = int(np.clip(round(self._profile_line.value()), 0, limit))
        profile = self._image[:, position] if vertical else self._image[position, :]
        self._profile_curve.setData(np.asarray(profile, dtype=float))
        self._profile_plot.setTitle(f"{'Column' if vertical else 'Row'} {position}")

    def _update_analysis(self, data: np.ndarray) -> None:
        values = _finite(data)
        if values.size:
            counts, edges = np.histogram(values, bins=256)
            self._histogram_curve.setData(edges, counts)
        else:
            self._histogram_curve.setData([], [])
        self._density_curve.setData(*percentile_density(data))
        stats = {"Shape": str(data.shape), "Type": str(data.dtype)}
        stats.update(
            {name: f"{value:.6g}" for name, value in image_statistics(data).items()}
        )
        self._stats_table.setRowCount(len(stats))
        for row, (name, value) in enumerate(stats.items()):
            self._stats_table.setItem(row, 0, QTableWidgetItem(name))
            self._stats_table.setItem(row, 1, QTableWidgetItem(value))

    def _clear_analysis(self) -> None:
        self._histogram_curve.setData([], [])
        self._density_curve.setData([], [])
        self._stats_table.setRowCount(0)

    # 1D data

    def _show_plot_page(self, info: DatasetInfo) -> None:
        self._views.setCurrentIndex(1)
        self._analysis.setCurrentIndex(3)
        self._x_key.blockSignals(True)
        self._x_key.clear()
        self._x_key.addItem(_INDEX)
        for other in self._datasets:
            if other.stream == info.stream and other != info and not other.is_image:
                self._x_key.addItem(other.key)
        self._x_key.blockSignals(False)
        self._line_plot.setLabel("left", info.key)
        self._plot_1d()

    def _on_x_key_changed(self, _index: int = 0) -> None:
        self._plot_1d()

    def _plot_1d(self) -> None:
        info = self._dataset
        if info is None or info.is_image:
            return
        keys = [info.key]
        x_key = self._x_key.currentText()
        if x_key and x_key != _INDEX:
            keys.append(x_key)
        missing = [key for key in keys if (info.stream, key) not in self._columns]
        to_read = [
            key for key in missing if (info.stream, key) not in self._columns_loading
        ]
        if to_read:
            self._columns_loading.update((info.stream, key) for key in to_read)
            stream = self._streams[info.stream]
            self._start(
                _read_columns,
                stream,
                info.stream,
                to_read,
                returned=self._on_columns_read,
            )
        if missing:
            return
        y = _numeric(self._columns[(info.stream, info.key)])
        if y is None:
            self._line_curve.clear()
            self._clear_analysis()
            self._status.setText(
                f"{info.stream}/{info.key} holds values that are not numbers; "
                "nothing to plot."
            )
            return
        x = None
        if x_key != _INDEX:
            x = _numeric(self._columns[(info.stream, x_key)])
            problem = None
            if x is None:
                problem = "holds values that are not numbers"
            elif x.shape != y.shape:
                problem = f"has {x.size} values and {info.key} has {y.size}"
                x = None
            if problem:
                self._status.setText(f"{x_key} {problem}; plotting against index.")
        self._line_plot.setLabel("bottom", _INDEX if x is None else x_key)
        if x is not None and x_key == "time" and x.size:
            x = x - x[0]
            self._line_plot.setLabel("bottom", "Time since first event", units="s")
        if x is None:
            self._line_curve.setData(y)
        else:
            self._line_curve.setData(x, y)
        self._line_plot.autoRange()
        self._update_analysis(y)
        if x is not None or x_key == _INDEX:
            self._status.setText(f"{info.stream}/{info.key}: {y.size} value(s)")

    def _on_columns_read(
        self,
        result: tuple[
            str, list[str], dict[tuple[str, str], np.ndarray], Exception | None
        ],
    ) -> None:
        stream_name, keys, columns, error = result
        self._columns_loading.difference_update((stream_name, key) for key in keys)
        self._columns.update(columns)
        info = self._dataset
        if info is None or info.is_image or info.stream != stream_name:
            return
        x_key = self._x_key.currentText()
        needed = [info.key] + ([x_key] if x_key and x_key != _INDEX else [])
        if error is not None:
            # Not replotted: that would start the failed read again.
            if set(needed) & set(keys):
                self._status.setText(
                    f"Error reading {stream_name}/{', '.join(keys)}: "
                    f"{type(error).__name__}: {error}"
                )
            return
        if all((info.stream, key) in self._columns for key in needed):
            self._plot_1d()
