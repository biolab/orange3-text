# pylint: disable=missing-docstring
import unittest
from typing import List
from unittest.mock import Mock

from Orange.data import StringVariable, Table, Domain
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.widgets.owontology import OWOntology, _run, \
    EditableTreeView


def create_words_table(words: List) -> Table:
    words_var = StringVariable("Words")
    words_var.attributes = {"type": "words"}
    domain = Domain([], metas=[words_var])
    data = [[w] for w in words]
    words = Table.from_list(domain, data)
    words.name = "Words"
    return words


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.words = ["foo", "bar"]
        self.state = Mock()
        self.state.is_interruption_requested = Mock(return_value=False)

    def test_run(self):
        result = _run(self.words, self.state)
        self.assertEqual(result, {'foo': {'bar': {}}})

    def test_run_single_word(self):
        result = _run(["foo"], self.state)
        self.assertEqual(result, {"foo": {}})

    def test_run_empty(self):
        result = _run([], self.state)
        self.assertEqual(result, {})

    def test_run_interrupt(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=True)
        self.assertRaises(Exception, _run, self.words, state)


class TestEditableTreeView(WidgetTest):
    def setUp(self):
        self.data = {"foo": {"bar": {}, "baz": {}}}
        self.view = EditableTreeView()

    def test_set_data(self):
        model = self.view._EditableTreeView__model

        self.view.set_data(self.data)
        self.assertEqual(model.rowCount(), 1)

        self.view.set_data(self.data)
        self.assertEqual(model.rowCount(), 1)

    def test_clear(self):
        model = self.view._EditableTreeView__model

        self.view.set_data(self.data)
        self.assertEqual(model.rowCount(), 1)

        self.view.clear()
        self.assertEqual(model.rowCount(), 0)


class TestOWKeywords(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWOntology)

    def test_input_words(self):
        words = create_words_table(["foo"])
        self.send_signal(self.widget.Inputs.words, words)
        self.assertFalse(self.widget.Warning.no_words_column.is_shown())

    def test_input_words_no_type(self):
        words = Table("zoo")
        self.send_signal(self.widget.Inputs.words, words)
        self.assertTrue(self.widget.Warning.no_words_column.is_shown())
        self.send_signal(self.widget.Inputs.words, None)
        self.assertFalse(self.widget.Warning.no_words_column.is_shown())


if __name__ == "__main__":
    unittest.main()
