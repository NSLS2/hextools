import pytest

from hextools.gui import _theme  # noqa: PLC2701


@pytest.mark.parametrize("theme", sorted(_theme.THEMES))
def test_build_stylesheet_fills_every_placeholder(theme):
    stylesheet = _theme.build_stylesheet(theme)
    assert _theme.THEMES[theme]["BACKGROUND"] in stylesheet
    # Unfilled placeholders or unescaped braces would leave single braces behind.
    assert "{{" not in stylesheet
    assert "}}" not in stylesheet


def test_dark_is_default_and_themes_differ():
    assert _theme.DEFAULT_THEME == "dark"
    assert _theme.STYLESHEET == _theme.build_stylesheet("dark")
    assert _theme.build_stylesheet("dark") != _theme.build_stylesheet("light")


def test_unknown_theme_rejected():
    with pytest.raises(ValueError, match="Unknown theme"):
        _theme.build_stylesheet("sepia")


@pytest.fixture
def qt_app(monkeypatch, tmp_path):
    pytest.importorskip("qtpy")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtCore import QSettings
    from qtpy.QtWidgets import QApplication

    QSettings.setPath(
        QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    QSettings.setPath(
        QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path)
    )
    app = QApplication.instance() or QApplication([])
    _theme.apply_bnl_theme(app, _theme.DEFAULT_THEME)
    return app


def test_apply_and_remember_theme(qt_app):
    app = qt_app
    assert _theme.saved_theme() == "dark"

    _theme.apply_bnl_theme(app, "light")
    _theme.save_theme("light")
    assert app.styleSheet() == _theme.build_stylesheet("light")
    assert _theme.current_theme() == "light"
    assert _theme.saved_theme() == "light"

    _theme.apply_bnl_theme(app, "dark")
    assert app.styleSheet() == _theme.STYLESHEET
    assert _theme.current_theme() == "dark"


def test_every_switch_follows_a_theme_change(qt_app):
    # The stylesheet is application-wide, but each window has its own switch.
    # PR 91 review: a second window's switch kept showing the old theme.
    from hextools.gui.theme_switch import QtThemeSwitch

    first, second = QtThemeSwitch(), QtThemeSwitch()
    assert first.isChecked() and second.isChecked()

    first.setChecked(False)
    assert _theme.current_theme() == "light"
    assert qt_app.styleSheet() == _theme.build_stylesheet("light")
    assert not second.isChecked()

    second.setChecked(True)
    assert _theme.current_theme() == "dark"
    assert first.isChecked()

    # A switch made after a change starts in step with it.
    second.setChecked(False)
    assert not QtThemeSwitch().isChecked()
