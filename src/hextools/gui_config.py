"""What the data-acquisition GUI shows at HEX (27-ID).

Everything HEX-specific about the GUI lives here; ``hextools.gui`` is generic.
Launch with ``python -m hextools.gui`` (or the ``hex-gui`` pixi task).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.fastcs.panda import HDFPanda

from hextools.detectors.germ import GeRMDetectorMock
from hextools.detectors.phantom import PhantomDetector
from hextools.edxd import (
    configure_test_pulses,
    edxd_2theta_tilt,
    edxd_calib_scan,
    edxd_count,
    edxd_custom_pos_list_grid,
    edxd_grid_scan,
    edxd_scan,
)
from hextools.gui.spec import DetectorView, GuiSpec, ProposalBackend, Site, Technique
from hextools.machine import NSLS2StorageRing
from hextools.motors import (
    FOV_2_4_mm_Camera,
    FOV_20_40_mm_Camera,
    OpticsTable,
    SampleSmall,
    SampleTower,
    move_motor,
)
from hextools.photon_delivery_system import (
    DCLM,
    FrontEndSlit,
    PPSShutter,
    Shutter,
    Slits,
    change_energy,
    move_slits,
    set_fe_aperture,
    set_filters,
)
from hextools.photon_delivery_system.dclm import change_beam_mode
from hextools.photon_delivery_system.shutter import (
    ensure_shutter_closed,
    ensure_shutter_open,
)
from hextools.settings import (
    apply_device_settings,
    store_device_settings,
    store_device_settings_snapshot,
)
from hextools.specs import ALL_SPECS
from hextools.tomography.alignment import tomo_alignment_scan
from hextools.tomography.flyscans import (
    tomo_1d_step_scan,
    tomo_2d_step_scan,
    tomo_dark_flat,
    tomo_flyscan,
    tomo_flyscan_loop,
)
from hextools.tomography.radiography import take_radiograph
from hextools.utils import (
    get_proposal_info,
    is_running_in_ci,
    list_proposals,
    reset_scan_id,
    switch_proposal,
)


def _shutters(re, namespace):
    from hextools.gui.shutter_status import QtShutterStatus

    shutters = {}
    seen = set()
    for name, obj in sorted(namespace.items()):
        if (
            not name.startswith("_")
            and isinstance(obj, Shutter)
            and id(obj) not in seen
        ):
            shutters[name] = obj
            seen.add(id(obj))

    def command(name: str, open_shutter: bool) -> str:
        plan = ensure_shutter_open if open_shutter else ensure_shutter_closed
        namespace.setdefault(plan.__name__, plan)
        return f"RE({plan.__name__}({name}, allow_actuation=True, sleep_time=0.0))"

    return QtShutterStatus(re, shutters, command)


def _germ_viewer_options(mock: bool) -> dict[str, Any]:
    """Configure the HEX GeRM viewer, including its mock-frame adapter."""
    options: dict[str, Any] = {
        "raw_waveform": True,
        "image_shape": (192, 4096),
        "colormap": "JET",
        "roi_suffix_pattern": "1:ROI{}:",
        "auto_scale_method": "minmax",
        "log_scale": True,
        "use_ca": True,
        "show_profile_lines": True,
    }
    if mock:
        options["push_frame"] = lambda push: None
    return options


def _build_edxd_view(re_client: Any, _spec: GuiSpec) -> Any:
    """Build the HEX GeRM view, including its mock-frame adapter."""
    import IPython
    from bluesky import RunEngine
    from qtpy.QtWidgets import QHBoxLayout, QTabWidget, QVBoxLayout, QWidget

    from hextools.gui.misc import QtTabbedDetectorsWidget
    from hextools.gui.plan_widget import QtPlanWidget
    from hextools.photon_delivery_system import Shutter

    class HexEDXDView(QWidget):
        def __init__(self):
            super().__init__()
            top = QHBoxLayout()
            detectors = QtTabbedDetectorsWidget(re_client)
            mock = isinstance(re_client, RunEngine) and is_running_in_ci()
            detectors.add_detector(
                "germ",
                "XF:27ID1-ES{GeRM-Det:1}MCA",
                **_germ_viewer_options(mock),
            )
            top.addWidget(detectors, stretch=3)

            plan_tabs = QTabWidget()
            for label, plan in (
                ("2Theta Tilt", edxd_2theta_tilt),
                ("Calibration", edxd_calib_scan),
                ("Scan", edxd_scan),
                ("Grid Scan", edxd_grid_scan),
                ("Custom Pos List Grid", edxd_custom_pos_list_grid),
                ("Count", edxd_count),
                ("Configure Test Pulses", configure_test_pulses),
            ):
                plan_tabs.addTab(QtPlanWidget(re_client, plan), label)
            side = QVBoxLayout()
            side.addWidget(plan_tabs, stretch=1)
            top.addLayout(side, stretch=1)
            layout = QVBoxLayout()
            layout.addLayout(top, stretch=1)
            self.setLayout(layout)

            self._germ_mock = None
            if mock:
                ipython = IPython.get_ipython()
                namespace = ipython.user_ns if ipython is not None else {}
                detector_mock = getattr(namespace.get("germ"), "_mock", None)
                if isinstance(detector_mock, GeRMDetectorMock):
                    detector_mock.set_shutters(
                        *(
                            shutter
                            for name in ("fe_shutter", "photon_shutter")
                            if isinstance(shutter := namespace.get(name), Shutter)
                        )
                    )
                    self._show_germ_frame = lambda frame: detectors.show_frame(
                        "germ", frame
                    )
                    detector_mock.register_frame_callback(self._show_germ_frame)
                    self._germ_mock = detector_mock

        def closeEvent(self, event):  # noqa: N802
            if self._germ_mock is not None:
                self._germ_mock.unregister_frame_callback(self._show_germ_frame)
                self._germ_mock = None
            super().closeEvent(event)

    return HexEDXDView()


def _default_plan_history_file() -> Path:
    state_home = Path(
        os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state"
    )
    filename = "plan_history_mock.json" if is_running_in_ci() else "plan_history.json"
    return state_home / "hextools" / filename


def _nxs_export():
    from hextools.gui.auto_exporter_view import QtAutoExporterView

    return QtAutoExporterView(
        "hextools.exporters.auto_exporter", autostart=not is_running_in_ci()
    )


HEX_GUI = GuiSpec(
    name="HEX",
    techniques=[
        Technique(
            "Beamline",
            plans=[
                ("Motors", move_motor),
                ("Slits", move_slits),
                ("Frontend Aperture", set_fe_aperture),
                ("Filters", set_filters),
                ("Change Beam Mode", change_beam_mode),
                ("Change Energy", change_energy),
                ("Reset Scan ID", reset_scan_id),
                ("Settings Snapshot", store_device_settings_snapshot),
                ("Store Settings", store_device_settings),
                ("Apply Settings", apply_device_settings),
            ],
            detectors=[
                DetectorView("fs_window_cam", "XF:27IDA-BI{FS:1-Cam:1}"),
                DetectorView("diamond_window_cam", "XF:27IDA-BI{FAM:1-Cam:1}"),
                DetectorView("sample_cam", "XF:27ID1-ES{Sample-Cam:1}"),
                DetectorView("f_hutch_cam", "XF:27IDA-BI{GigE-Cam:5}"),
            ],
        ),
        Technique(
            "Tomography",
            plans=[
                ("Dark/Flat", tomo_dark_flat),
                ("Flyscan", tomo_flyscan),
                ("Flyscan Loop", tomo_flyscan_loop),
                ("1D Step", tomo_1d_step_scan),
                ("2D Step", tomo_2d_step_scan),
                ("Radiography", take_radiograph),
                ("Alignment", tomo_alignment_scan),
            ],
            detectors=[
                DetectorView(
                    f"kinetix{i}",
                    "XF:27ID1-BI{Kinetix-Det:" + str(i) + "}",
                    {"show_profile_lines": True},
                )
                for i in range(1, 5)
            ],
            combined=[("Dual Cam", ["kinetix1", "kinetix3"])],
        ),
        Technique("EDXD", widget_factory=_build_edxd_view),
        Technique(
            "XRD",
            detectors=[DetectorView("perkin_elmer", "XF:27ID1-ES{PE-Det:1}")],
        ),
    ],
    # Names must match those defined in the profile / Queue Server namespace.
    expected_devices=[
        ("fe_shutter", PPSShutter),
        ("photon_shutter", Shutter),
        ("a_slits", Slits),
        ("f_slits", Slits),
        ("fe_slit1", FrontEndSlit),
        ("storage_ring", NSLS2StorageRing),
        ("dclm", DCLM),
        ("optics_table", OpticsTable),
        ("sample_tower", SampleTower),
        ("sample_small", SampleSmall),
        ("panda", HDFPanda),
        ("kinetix1", KinetixDetector),
        ("kinetix2", KinetixDetector),
        ("kinetix3", KinetixDetector),
        ("kinetix4", KinetixDetector),
        ("double_obj_camera", FOV_2_4_mm_Camera),
        ("wide_fov_camera", FOV_20_40_mm_Camera),
        ("phantom", PhantomDetector),
        ("sample_cam", VimbaDetector),
        ("f_hutch_cam", VimbaDetector),
    ],
    logger_name="hextools",
    run_specs=ALL_SPECS,
    status_widgets=[_shutters],
    extra_tabs=[("NXS Export", _nxs_export)],
    proposals=ProposalBackend(
        list_proposals=list_proposals,
        get_proposal_info=get_proposal_info,
        switch_proposal=switch_proposal,
        # Outside mock mode RE.md is Redis-backed, so a switch reaches the worker.
        md_is_shared=lambda: not is_running_in_ci(),
    ),
    site=Site(
        "Upton, NY", latitude=40.8677, longitude=-72.8814, timezone="America/New_York"
    ),
    profile_dir=Path(__file__).resolve().parent / "profiles",
    default_profile="collection",
    environment={"BEAMLINE_ACRONYM": "HEX"},
    mock_environment={"HEXTOOLS_RUNNING_IN_CI": "YES"},
    plan_history_env="HEXTOOLS_PLAN_HISTORY_FILE",
    default_plan_history_file=_default_plan_history_file,
    about_packages=(
        "hextools",
        "bluesky",
        "ophyd-async",
        "tiled",
        "bluesky-tiled-plugins",
    ),
    settings_scope=("NSLS2", "hextools-gui"),
)
