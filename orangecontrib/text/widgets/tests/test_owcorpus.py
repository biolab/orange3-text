import unittest
from unittest.mock import Mock

import numpy as np
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text import Corpus
from orangecontrib.text.widgets.owcorpus import OWCorpus


class TestOWCorpus(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCorpus)

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

    def test_input_status(self):
        """
        Test input, output info
        """
        data = Corpus.from_file("election-tweets-2016")
        input_sum = self.widget.info.set_input_summary = Mock()

        self.send_signal(self.widget.Inputs.data, data)
        input_sum.assert_called_with(
            str(len(data)), f"{len(data)} data instances on input")
        input_sum.reset_mock()
        self.send_signal(self.widget.Inputs.data, data[:1])
        input_sum.assert_called_with("1", "1 data instance on input")
        input_sum.reset_mock()

        self.send_signal(self.widget.Inputs.data, None)
        input_sum.assert_called_with(self.widget.info.NoInput)
        input_sum.reset_mock()

    def test_output_status(self):
        """
        Test input, output info
        """
        # when input signal
        data = Corpus.from_file("election-tweets-2016")
        out_sum = self.widget.info.set_output_summary = Mock()

        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        out_sum.assert_called_with(
            str(len(data)),
            "6444 document(s)\n4 text features(s)\n7 other feature(s)\n"
            "Classification; discrete class with 2 values.")
        out_sum.reset_mock()

        # corpus without class
        data1 = Corpus(Domain(data.domain.attributes, metas=data.domain.metas),
                       data.X, metas=data.metas,
                       text_features=data.text_features)
        self.send_signal(self.widget.Inputs.data, data1)
        self.wait_until_finished()
        out_sum.assert_called_with(
            str(len(data)),
            "6444 document(s)\n4 text features(s)\n7 other feature(s)")
        out_sum.reset_mock()

        # corpus with continuous class
        data1 = Corpus(Domain(data.domain.attributes,
                              ContinuousVariable("a"),
                              metas=data.domain.metas),
                       data.X, np.random.rand(len(data), 1),
                       metas=data.metas,
                       text_features=data.text_features)
        self.send_signal(self.widget.Inputs.data, data1)
        self.wait_until_finished()
        out_sum.assert_called_with(
            str(len(data)),
            "6444 document(s)\n4 text features(s)\n7 other feature(s)\n"
            "Regression; numerical class.")
        out_sum.reset_mock()

        # default dataset is on the output
        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()
        out_sum.assert_called_with(
            "140",
            "140 document(s)\n1 text features(s)\n0 other feature(s)\n"
            "Classification; discrete class with 2 values.")
        out_sum.reset_mock()

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


if __name__ == "__main__":
    unittest.main()