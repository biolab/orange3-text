import unittest

from orangecontrib.text.util import chunks


class ChunksTest(unittest.TestCase):

    def test_results(self):
        self.assertEqual(list(chunks([], 10)), [])
        self.assertEqual(list(chunks([1, 2], 3)), [[1, 2]])
        self.assertEqual(list(chunks([1, 2], 1)), [[1], [2]])

    def test_size(self):
        for chunk in chunks(range(10), 2):
            self.assertEqual(len(chunk), 2)

        for chunk in chunks(range(10), 3):
            pass
        self.assertEqual(len(chunk), 1)
