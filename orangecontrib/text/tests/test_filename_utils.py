"""Tests for the ``url_to_safe_filename`` utility in ``orangecontrib.text.misc``."""
import unittest

from orangecontrib.text.misc import url_to_safe_filename


class UrlToSafeFilenameTest(unittest.TestCase):
    """Tests for the ``url_to_safe_filename`` function."""

    def test_basic_http_url(self):
        """Test conversion of a basic HTTP URL."""
        result = url_to_safe_filename("https://example.com/path")
        self.assertEqual(result, "https___example.com_path")

    def test_url_with_query_params(self):
        """Test conversion of a URL with query parameters (& stays)."""
        result = url_to_safe_filename("https://example.com/path?foo=bar&baz=qux")
        self.assertEqual(result, "https___example.com_path_foo=bar&baz=qux")

    def test_url_with_port(self):
        """Test conversion of a URL with a port number."""
        result = url_to_safe_filename("https://example.com:8080/path")
        self.assertEqual(result, "https___example.com_8080_path")

    def test_url_with_fragment(self):
        """Test conversion of a URL with a fragment (# stays)."""
        result = url_to_safe_filename("https://example.com/path#section")
        self.assertEqual(result, "https___example.com_path#section")

    def test_replaces_invalid_windows_chars(self):
        """Test that all Windows-invalid characters are replaced with underscores.

        Characters < > : " / \\ | ? * (9 total) are all replaced.
        """
        # All 9 invalid chars: < > : " / \ | ? *
        invalid_chars = '<>:' + '"' + '/\\|?*'
        result = url_to_safe_filename(invalid_chars)
        self.assertEqual(result, "_________" )

    def test_replaces_question_mark_and_asterisk(self):
        """Test that '?' and '*' are replaced."""
        result = url_to_safe_filename("https://example.com/path?q=*test*")
        self.assertEqual(result, "https___example.com_path_q=_test_")

    def test_replaces_control_characters(self):
        """Test that control characters (0x00-0x1F) are replaced."""
        result = url_to_safe_filename("https://example.com/path\x00\x01\x1f")
        self.assertEqual(result, "https___example.com_path___")

    def test_handles_windows_reserved_name_CON(self):
        """Test that Windows reserved name 'CON' is prefixed when it's the full string."""
        result = url_to_safe_filename("CON")
        self.assertEqual(result, "_CON")

    def test_handles_windows_reserved_name_PRN(self):
        """Test that Windows reserved name 'PRN' is prefixed when it's the full string."""
        result = url_to_safe_filename("PRN")
        self.assertEqual(result, "_PRN")

    def test_handles_windows_reserved_name_AUX(self):
        """Test that Windows reserved name 'AUX' is prefixed when it's the full string."""
        result = url_to_safe_filename("AUX")
        self.assertEqual(result, "_AUX")

    def test_handles_windows_reserved_name_NUL(self):
        """Test that Windows reserved name 'NUL' is prefixed when it's the full string."""
        result = url_to_safe_filename("NUL")
        self.assertEqual(result, "_NUL")

    def test_handles_windows_reserved_name_COM1(self):
        """Test that Windows reserved name 'COM1' is prefixed when it's the full string."""
        result = url_to_safe_filename("COM1")
        self.assertEqual(result, "_COM1")

    def test_handles_windows_reserved_name_LPT1(self):
        """Test that Windows reserved name 'LPT1' is prefixed when it's the full string."""
        result = url_to_safe_filename("LPT1")
        self.assertEqual(result, "_LPT1")

    def test_handles_case_insensitive_reserved_names(self):
        """Test that reserved names are matched case-insensitively."""
        for name in ("con", "Con", "CoN", "CON"):
            result = url_to_safe_filename(name)
            self.assertEqual(result, "_" + name)

    def test_com10_not_reserved(self):
        """Test that COM10 is NOT treated as a reserved name (regex is COM[1-9])."""
        result = url_to_safe_filename("COM10")
        self.assertEqual(result, "COM10")

    def test_lpt4_not_reserved(self):
        """Test that LPT4 is NOT treated as a reserved name (regex is LPT[1-3])."""
        result = url_to_safe_filename("LPT4")
        self.assertEqual(result, "LPT4")

    def test_reserved_name_embedded_in_longer_string_not_affected(self):
        """Test that reserved names inside a longer path are not prefixed."""
        result = url_to_safe_filename("https://example.com/myCONfile")
        self.assertNotIn("__CON", result)
        self.assertIn("myCONfile", result)

    def test_empty_url_raises_value_error(self):
        """Test that an empty URL raises ValueError."""
        with self.assertRaises(ValueError):
            url_to_safe_filename("")

    def test_whitespace_only_url_raises_value_error(self):
        """Test that a whitespace-only URL raises ValueError."""
        with self.assertRaises(ValueError):
            url_to_safe_filename("   ")

    def test_none_url_raises_value_error(self):
        """Test that None raises ValueError (caught by runtime check)."""
        with self.assertRaises(ValueError):
            url_to_safe_filename(None)  # type: ignore

    def test_unicode_characters_pass_through(self):
        """Test that Unicode characters are not modified."""
        result = url_to_safe_filename("https://example.com/café")
        self.assertIn("cafe", result)

    def test_multiple_consecutive_invalid_chars(self):
        """Test that multiple consecutive invalid chars each become underscores."""
        result = url_to_safe_filename("///")
        self.assertEqual(result, "___")

    def test_mixed_special_chars(self):
        """Test URL with a mix of special characters."""
        result = url_to_safe_filename("https://example.com/path/to/file?v=1.0#top")
        self.assertEqual(result, "https___example.com_path_to_file_v=1.0#top")


if __name__ == "__main__":
    unittest.main()
