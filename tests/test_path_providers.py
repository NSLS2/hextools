"""Mock detectors never reuse a file name."""

from hextools.utils.path_providers import mock_path_provider


# Copilot on #109: second-resolution names let two short runs in the same
# second share an asset file.
def test_mock_path_provider_gives_each_call_a_new_file(tmp_path):
    provider = mock_path_provider(tmp_path)
    assert provider("germ").filename != provider("germ").filename
