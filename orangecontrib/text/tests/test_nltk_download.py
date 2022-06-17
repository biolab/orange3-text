import os
import unittest
from orangecontrib.text.misc.nltk_data_download import _get_proxy_address


class TestNLTKDownload(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_https = os.environ.get("https_proxy")
        os.environ.pop("https_proxy", None)

    def tearDown(self) -> None:
        os.environ.pop("https_proxy", None)
        if self.previous_https is not None:
            os.environ["https_proxy"] = self.previous_https

    def test_get_proxy_address(self):
        self.assertIsNone(_get_proxy_address())

        os.environ["https_proxy"] = "https://test.com"
        self.assertEqual("https://test.com:443", _get_proxy_address())

        os.environ["https_proxy"] = "https://test.com:12"
        self.assertEqual("https://test.com:12", _get_proxy_address())

        os.environ["https_proxy"] = "https://test.com/test"
        self.assertEqual("https://test.com:443/test", _get_proxy_address())

        os.environ["https_proxy"] = "https://test.com/test?a=2"
        self.assertEqual("https://test.com:443/test?a=2", _get_proxy_address())

        os.environ["https_proxy"] = "test.com/test?a=2"
        self.assertEqual("http://test.com:80/test?a=2", _get_proxy_address())


if __name__ == "__main__":
    unittest.main()
