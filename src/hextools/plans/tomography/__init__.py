"""
HEX beamline tomography plans — one module per legacy pyepics script,
detector(s) passed as arguments (configure, don't copy, per detector).

Exports
-------
alignment_scan  : alignment step-scan with optional flats
take_dark_flat  : dark + flat-field acquisition
take_radiograph : N frames at a fixed position
scan_1d         : 1-D motor step-scan, one image per point
tomo_flyscan    : hardware-triggered fly-scan tomography (PandA)
run_multiple_scans    : batch runner (motor series or time series + dark/flats)
run_multiple_2d_scans : grid batch runner (two motors, dark/flat per row)
tomo_flyscan_average / run_multiple_scans_average / run_multiple_2d_scans_average :
    frame-averaged variants (Proc-plugin recursive filter)
reset_detector          : recovery — detector/PandA/stage back to defaults
enable_flat_correction  : live-view flat-field correction mode
"""

from .alignment_scan import alignment_scan
from .enable_flat_correction import enable_flat_correction
from .reset_detector import reset_detector
from .run_multiple_2d_scans import run_multiple_2d_scans
from .run_multiple_2d_scans_average import run_multiple_2d_scans_average
from .run_multiple_scans import run_multiple_scans
from .run_multiple_scans_average import run_multiple_scans_average
from .scan_1d import scan_1d
from .take_dark_flat import take_dark_flat
from .take_radiograph import take_radiograph
from .tomo_flyscan import tomo_flyscan
from .tomo_flyscan_average import tomo_flyscan_average

__all__ = [
    "alignment_scan",
    "take_dark_flat",
    "take_radiograph",
    "scan_1d",
    "tomo_flyscan",
    "run_multiple_scans",
    "run_multiple_2d_scans",
    "tomo_flyscan_average",
    "run_multiple_scans_average",
    "run_multiple_2d_scans_average",
    "reset_detector",
    "enable_flat_correction",
]
