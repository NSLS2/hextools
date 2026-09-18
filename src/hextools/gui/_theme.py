"""BNL brand theming for the HEX GUI.

A dark theme built on the Brookhaven National Laboratory brand palette
(https://www.bnl.gov/brandcenter/palette.php).
"""

from __future__ import annotations

from pathlib import Path

# SVG assets referenced from the stylesheet (Qt needs forward-slash URLs).
_ASSET_URL = (Path(__file__).parent / "assets").as_posix()
# Primary palette
TEAL = "#105C78"
CERULEAN = "#00ADDC"
LIME = "#B2D33B"
ORANGE = "#F68B1F"
FUCHSIA = "#B72467"

# Secondary palette
GOLDENROD = "#FFCD34"
CRIMSON = "#DB3526"
VIOLET = "#51499E"
CORNFLOWER = "#4881C3"
JADE = "#25B574"

# Neutrals & utility grays
BLACK = "#000000"
DARK_GRAY = "#4C515A"
GRAY = "#58595B"
MID_GRAY = "#858889"
LIGHT_GRAY = "#ACB2AE"
SAND = "#BDBDB0"
CREAM = "#DDDCCB"

# Derived tints/shades used for surfaces (dark theme).
TEAL_DARK = "#0C4A61"
TEAL_HOVER = "#16708F"
CERULEAN_DARK = "#0090BC"
BACKGROUND = "#1E2227"
SURFACE = "#282D34"
ALT_ROW = "#2F353D"
BORDER = "#3A424B"
BORDER_STRONG = "#4C555F"
TEXT = "#E4E8EB"
TEXT_MUTED = "#9AA5AD"
# Light foreground used on top of accent-colored (teal/cerulean) surfaces.
ON_ACCENT = "#FFFFFF"

BNL_COLORS = {
    "teal": TEAL,
    "cerulean": CERULEAN,
    "lime": LIME,
    "orange": ORANGE,
    "fuchsia": FUCHSIA,
    "goldenrod": GOLDENROD,
    "crimson": CRIMSON,
    "violet": VIOLET,
    "cornflower": CORNFLOWER,
    "jade": JADE,
}

STYLESHEET = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 12px;
}}
QMainWindow, QDialog {{ background-color: {BACKGROUND}; }}
QLabel {{ background: transparent; }}
QToolTip {{
    background-color: {TEAL};
    color: {ON_ACCENT};
    border: 1px solid {TEAL_DARK};
    padding: 4px 6px;
}}

QGroupBox {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 8px;
    padding: 28px 10px 12px 10px;
    background-color: {SURFACE};
}}
QGroupBox::title {{
    subcontrol-origin: border;
    subcontrol-position: top left;
    left: 12px;
    top: 8px;
    padding: 0 4px;
    color: {CERULEAN};
    font-weight: 600;
}}

QPushButton {{
    background-color: {TEAL};
    color: {ON_ACCENT};
    border: none;
    border-radius: 4px;
    padding: 4px 10px;
    min-width: 44px;
    min-height: 22px;
}}
QPushButton:hover {{ background-color: {TEAL_HOVER}; }}
QPushButton:pressed {{ background-color: {TEAL_DARK}; }}
QPushButton:checked {{ background-color: {CERULEAN}; }}
QPushButton:disabled {{
    background-color: {BORDER};
    color: {TEXT_MUTED};
}}
QToolButton {{
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 3px;
}}
QToolButton:hover {{ background-color: {ALT_ROW}; border-color: {BORDER}; }}
QToolButton:pressed, QToolButton:checked {{
    background-color: {CERULEAN}; color: {ON_ACCENT};
}}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    padding: 4px 6px;
    selection-background-color: {CERULEAN};
    selection-color: {ON_ACCENT};
}}
QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {CERULEAN};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 2px solid {CERULEAN};
    padding: 3px 5px;
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {SURFACE};
    border: 1px solid {BORDER_STRONG};
    selection-background-color: {CERULEAN};
    selection-color: {ON_ACCENT};
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {SURFACE};
}}
QTabBar::tab {{
    background: {BACKGROUND};
    color: {TEXT_MUTED};
    padding: 3px 12px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}}
QTabBar::tab:last, QTabBar::tab:only-one {{
    border-right: 1px solid {BORDER};
}}
QTabBar::tab:selected {{
    background: {SURFACE};
    color: {CERULEAN};
    font-weight: 600;
    border-top: 3px solid {LIME};
    padding-top: 1px;
    margin-bottom: -1px;
}}
QTabBar::tab:hover:!selected {{ color: {CERULEAN}; background: {ALT_ROW}; }}

/* Main viewer uses a vertical (West) tab bar. Scope with '>' so the
   nested plan-editor (North) tab bar does not inherit these rules. */
QTabWidget#mainViewerTabs::pane {{ left: -1px; }}
QTabWidget#mainViewerTabs > QTabBar::tab {{
    padding: 12px 5px;
    border: 1px solid {BORDER};
    border-right: none;
    border-top-left-radius: 5px;
    border-bottom-left-radius: 5px;
    border-top-right-radius: 0;
    border-bottom-right-radius: 0;
    margin-right: 0;
    margin-bottom: 3px;
}}
QTabWidget#mainViewerTabs > QTabBar::tab:selected {{
    border: 1px solid {BORDER};
    border-right: none;
    border-left: 3px solid {LIME};
    margin-bottom: 3px;
}}

QHeaderView::section {{
    background-color: {TEAL};
    color: {ON_ACCENT};
    padding: 5px 6px;
    border: none;
    border-right: 1px solid {TEAL_DARK};
    font-weight: 600;
}}
QTableView, QTableWidget, QTreeView, QListView {{
    background-color: {SURFACE};
    alternate-background-color: {ALT_ROW};
    gridline-color: {BORDER};
    border: 1px solid {BORDER};
    border-radius: 4px;
    selection-background-color: {CERULEAN};
    selection-color: {ON_ACCENT};
    outline: none;
}}
QTableView::item, QTreeView::item, QListView::item {{ padding: 2px 4px; }}
QTableCornerButton::section {{ background-color: {TEAL}; border: none; }}

QMenuBar {{ background-color: {TEAL}; color: {ON_ACCENT}; padding: 2px; }}
QMenuBar::item {{ background: transparent; padding: 5px 12px; border-radius: 4px; }}
QMenuBar::item:selected {{ background: {TEAL_HOVER}; }}
QMenu {{ background-color: {SURFACE}; border: 1px solid {BORDER_STRONG}; padding: 4px; }}
QMenu::item {{ padding: 5px 24px 5px 12px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: {CERULEAN}; color: {ON_ACCENT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}

QProgressBar {{
    border: 1px solid {BORDER_STRONG};
    border-radius: 5px;
    text-align: center;
    background: {SURFACE};
    color: {TEXT};
    min-height: 18px;
}}
QProgressBar::chunk {{ background-color: {CERULEAN}; border-radius: 4px; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 0; }}
QScrollBar::handle:vertical {{
    background: {BORDER_STRONG};
    border-radius: 6px;
    min-height: 24px;
}}
QScrollBar::handle:vertical:hover {{ background: {CERULEAN}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; margin: 0; }}
QScrollBar::handle:horizontal {{
    background: {BORDER_STRONG};
    border-radius: 6px;
    min-width: 24px;
}}
QScrollBar::handle:horizontal:hover {{ background: {CERULEAN}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 14px;
    height: 14px;
    background: {SURFACE};
    border: 1px solid {BORDER_STRONG};
}}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {TEAL}; }}
QCheckBox::indicator:checked {{
    background: {TEAL};
    border-color: {TEAL};
    image: url({_ASSET_URL}/check.svg);
}}
QRadioButton::indicator:checked {{
    background: {TEAL};
    border-color: {TEAL};
    image: url({_ASSET_URL}/radio-dot.svg);
}}
QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 4px; }}
QSplitter::handle:vertical {{ height: 4px; }}
QStatusBar {{ background: {TEAL}; color: {ON_ACCENT}; }}
QStatusBar::item {{ border: none; }}
"""


def apply_bnl_theme(app=None):
    """Apply the BNL-branded stylesheet to the given (or current) QApplication."""
    if app is None:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
    if app is not None:
        app.setStyleSheet(STYLESHEET)
