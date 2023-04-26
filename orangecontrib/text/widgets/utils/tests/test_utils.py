import unittest
from datetime import datetime


class TestEnum2Int(unittest.TestCase):
    def test_remove_enum2int(self):
        """
        Happy new year 2024. When this test start to fail:
        - remove enum2int from orangecontrib.text.widgets.utils.__init__
        - change imports to Orange's enum2int in widget that use it
        - remove this test
        - depend orange3-text on orange 3.35
        """
        self.assertLess(datetime.today(), datetime(2024, 1, 1))


if __name__ == "__main__":
    unittest.main()
