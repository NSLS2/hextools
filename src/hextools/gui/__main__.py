"""Launch the HEX data-acquisition GUI: ``python -m hextools.gui``."""

from hextools.gui.window import run

HEX_SPEC = "hextools.gui_config:HEX_GUI"
PLAN_HISTORY_FILE_ENV = "HEXTOOLS_PLAN_HISTORY_FILE"


def main(argv: list[str] | None = None):
    """Launch the GUI with HEX's spec."""
    run(HEX_SPEC, argv)


def launch_local_viewer(end_ipython_on_close: bool = False):
    """Compatibility wrapper for the former HEX launcher helper."""
    from hextools.gui.spec import load_spec
    from hextools.gui.window import launch_local_viewer as launch

    return launch(load_spec(HEX_SPEC), end_ipython_on_close)


def germ_viewer_options(mock: bool):
    """Compatibility wrapper for the former HEX EDXD-viewer helper."""
    from hextools.gui_config import _germ_viewer_options

    return _germ_viewer_options(mock)


def default_plan_history_file():
    """Compatibility wrapper for the former HEX history-path helper."""
    from hextools.gui_config import _default_plan_history_file

    return _default_plan_history_file()


if __name__ == "__main__":
    main()
