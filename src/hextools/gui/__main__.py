"""Launch the HEX data-acquisition GUI: ``python -m hextools.gui``."""

from hextools.gui.window import run

HEX_SPEC = "hextools.gui_config:HEX_GUI"


def main(argv: list[str] | None = None):
    """Launch the GUI with HEX's spec."""
    run(HEX_SPEC, argv)


if __name__ == "__main__":
    main()
