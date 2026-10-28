"""What the data-acquisition GUI shows at HEX (27-ID).

Everything HEX-specific about the GUI lives here; ``hextools.gui`` is generic.
Launch with ``python -m hextools.gui`` (or the ``hex-gui`` pixi task).
"""

from __future__ import annotations

from pathlib import Path

from ophyd_async.epics.adkinetix import KinetixDetector
from ophyd_async.epics.advimba import VimbaDetector
from ophyd_async.fastcs.panda import HDFPanda

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
    SampleTower,
    move_motor,
)
from hextools.photon_delivery_system import (
    DCLM,
    Shutter,
    Slits,
    change_energy,
    move_slits,
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

    return QtShutterStatus(
        re,
        namespace,
        device_type=Shutter,
        open_plan=ensure_shutter_open,
        close_plan=ensure_shutter_closed,
        plan_kwargs={"allow_actuation": True},
    )


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
        Technique(
            "EDXD",
            plans=[
                ("2Theta Tilt", edxd_2theta_tilt),
                ("Calibration", edxd_calib_scan),
                ("Scan", edxd_scan),
                ("Grid Scan", edxd_grid_scan),
                ("Custom Pos List Grid", edxd_custom_pos_list_grid),
                ("Count", edxd_count),
                ("Configure Test Pulses", configure_test_pulses),
            ],
            detectors=[
                DetectorView(
                    "germ",
                    "XF:27ID1-ES{GeRM-Det:1}MCA",
                    {
                        "raw_waveform": True,
                        "image_shape": (192, 4096),
                        "colormap": "JET",
                        "roi_suffix_pattern": "1:ROI{}:",
                        "auto_scale_method": "minmax",
                        "log_scale": True,
                        "use_ca": True,
                        "show_profile_lines": True,
                    },
                )
            ],
        ),
        Technique(
            "XRD",
            detectors=[DetectorView("perkin_elmer", "XF:27ID1-ES{PE-Det:1}")],
        ),
    ],
    # Names must match those defined in the profile / Queue Server namespace.
    expected_devices=[
        ("fe_shutter", Shutter),
        ("photon_shutter", Shutter),
        ("a_slits", Slits),
        ("f_slits", Slits),
        ("storage_ring", NSLS2StorageRing),
        ("dclm", DCLM),
        ("optics_table", OpticsTable),
        ("sample_tower", SampleTower),
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
    environment={"BEAMLINE_ACRONYM": "HEX"},
    mock_environment={"HEXTOOLS_RUNNING_IN_CI": "YES"},
    plan_history_env="HEXTOOLS_PLAN_HISTORY_FILE",
    about_packages=(
        "hextools",
        "bluesky",
        "ophyd-async",
        "tiled",
        "bluesky-tiled-plugins",
    ),
    settings_scope=("NSLS2", "hextools-gui"),
)
