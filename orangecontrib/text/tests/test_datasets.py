import os
import unittest

from orangecontrib.text import Corpus


class TestDatasets(unittest.TestCase):
    def test_languages(self):
        cur_file = os.path.dirname(os.path.abspath(__file__))

        for file in os.listdir(os.path.join(cur_file, "..", "datasets")):
            if file.endswith((".tab", ".xlsx", ".pkl", ".csv")):
                c = Corpus.from_file(file)
                self.assertIsNotNone(c.language)


if __name__ == "__main__":
    unittest.main()
