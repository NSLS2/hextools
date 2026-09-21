from qtpy.QtWidgets import QTabWidget
from ntnda_qt_viewer import NTNDAViewerWidget

class QtTabbedDetectorsWidget(QTabWidget):
    """Tabbed NTNDArray live viewers, one tab per Kinetix detector."""

    def __init__(self, detectors: dict[str, str], *args, show_rois: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        for label, prefix in detectors.items():
            viewer = NTNDAViewerWidget(prefix=prefix)
            viewer._roi_controls_widget.setVisible(show_rois)  # noqa: SLF001
            self.addTab(viewer, label)
