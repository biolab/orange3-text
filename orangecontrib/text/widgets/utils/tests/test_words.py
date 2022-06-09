import unittest

from Orange.data import StringVariable
from orangecontrib.text.widgets.tests.test_owkeywords import create_words_table


class WordsTable(unittest.TestCase):
    def test_create_words_name(self):
        table = create_words_table(["foo", "bar"])
        self.assertEqual(table.name, "Words")

    def test_create_words_domain(self):
        table = create_words_table(["foo", "bar"])
        self.assertEqual(len(table.domain), 1)

    def test_create_words_var(self):
        table = create_words_table(["foo", "bar"])
        var = table.domain.metas[0]
        self.assertEqual(var.name, "Words")
        self.assertEqual(var.attributes, {"type": "words"})
        self.assertIsInstance(var, StringVariable)

    def test_create_words_data(self):
        words = ["foo", "bar"]
        table = create_words_table(words)
        self.assertEqual(list(table.metas[:, 0]), words)


if __name__ == '__main__':
    unittest.main()
