import unittest
from unittest.mock import Mock

import numpy as np
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text import Corpus
from orangecontrib.text.widgets.owcorpus import OWCorpus


class TestOWCorpus(WidgetTest):
    def setUp(self):
        self.widget: OWCorpus = self.create_widget(OWCorpus)

    def check_output(self, sel_title):
        """
        This function check whether the `sel_title` variable has a title true
        in the output
        """
        output = self.get_output(self.widget.Outputs.corpus)
        for attr in output.domain.variables + output.domain.metas:
            if str(attr) == sel_title:
                # sel_title attribute must be marked as a title
                self.assertTrue(attr.attributes.get("title", False))
            else:
                # others must not be marked as a title
                self.assertFalse(attr.attributes.get("title", False))

    def test_title_combo(self):
        self.wait_until_finished()
        # default corpus dataset
        self.assertEqual(self.widget.corpus.name, "book-excerpts")

        options = self.widget.title_model[:]
        self.assertIn(self.widget.corpus.domain["Text"], options)
        # for this dataset no title variable is selected
        self.assertEqual(None, self.widget.title_variable)
        self.check_output(None)

    def test_title_already_in_dataset(self):
        """
        This dataset already have the title attribute so the title option
        is set to this attribute by default
        """
        # default corpus dataset
        data = Corpus.from_file("election-tweets-2016")
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()

        self.assertEqual(data.domain["Content"], self.widget.title_variable)
        self.check_output("Content")

    def test_title_selection_strategy_title_heading(self):
        """
        When a there is a title, heading, filename attribute, select this one
        as a default title.
        """
        data = Table(
            Domain([], metas=[StringVariable("title"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["title"], self.widget.title_variable)
        self.check_output("title")

        data = Table(
            Domain([], metas=[StringVariable("Title"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Title"], self.widget.title_variable)
        self.check_output("Title")

        # when title and heading present first select title
        data = Table(
            Domain([], metas=[
                StringVariable("Title"),
                StringVariable("Heading"),
                StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Title"], self.widget.title_variable)
        self.check_output("Title")

        data = Table(
            Domain([], metas=[
                StringVariable("Heading"),
                StringVariable("Title"),
                StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Title"], self.widget.title_variable)
        self.check_output("Title")

        data = Table(
            Domain([], metas=[
                StringVariable("Heading"),
                StringVariable("Filename"),
                StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Heading"], self.widget.title_variable)
        self.check_output("Heading")

    def test_title_selection_strategy_most_unique(self):
        """
        With this test we test whether the selection strategy for a title
        attribute works correctly.
        This one must select the most unique feature.
        """
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b")]),
            np.empty((3, 0)),
            metas=[["a" * 10, "a" * 10],
                   ["a" * 10, "b" * 10],
                   ["a" * 10, "c" * 10]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["b"], self.widget.title_variable)
        self.check_output("b")

    def test_title_selection_strategy_shortness(self):
        """
        With this test we test whether the selection strategy for a title
        attribute works correctly.
        Select the uniquest and also short enough, here attribute a is not
        suitable since it has too long title, and c is more unique than b.
        """
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 10, "a" * 10],
                   ["b" * 100, "a" * 10, "b" * 10],
                   ["c" * 100, "a" * 10, "b" * 10]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["c"], self.widget.title_variable)
        self.check_output("c")

    def test_title_selection_strategy_shortest(self):
        """
        With this test we test whether the selection strategy for a title
        attribute works correctly.
        When no variable is short enough we just select the shortest attribute.
        """
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["c"], self.widget.title_variable)
        self.check_output("c")

    def test_keep_selected_variables(self):
        """
        When domain just slightly changes selected text variables should
        still be same.
        """
        attributes = [
            ContinuousVariable("a"), ContinuousVariable("b"),
            ContinuousVariable("c")]
        metas = [StringVariable("d"), StringVariable("e"),
                 StringVariable("f"), StringVariable("g"),
                 StringVariable("h"), StringVariable("i")]
        data = Table(
            Domain(attributes, metas=metas),
            np.array([[0] * len(attributes)]),
            metas=[["a" * 10] * len(metas)]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        prew_selected = data.domain.metas[1:3]
        self.widget.used_attrs = prew_selected

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()

        data = Table.from_table(
            Domain(attributes[:-1], [], metas=metas), data
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertListEqual(list(prew_selected), self.widget.used_attrs)

    def test_no_text_feature(self):
        """
        Test with data which have empty text_features. Widget should not show
        the error but, should have all features unused.
        """
        # widget already loads book-excerpts from file and store context
        # settings this call restore context settings to default otherwise
        # Text variable is moved to used_attributes by the context
        self.widget.settingsHandler.reset_to_original(self.widget)
        data = Corpus.from_file("book-excerpts")
        data.text_features = []
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertFalse(
            self.widget.Error.corpus_without_text_features.is_shown()
        )
        self.assertEqual(0, len(list(self.widget.used_attrs_model)))
        self.assertListEqual(
            [data.domain["Text"]],
            list(self.widget.unused_attrs_model)
        )

    def test_corpus_without_text_features(self):
        """
        Test if corpus_without_text_features is correctly raised for data
        without text features
        """
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertTrue(
            self.widget.Error.corpus_without_text_features.is_shown()
        )


if __name__ == "__main__":
    unittest.main()