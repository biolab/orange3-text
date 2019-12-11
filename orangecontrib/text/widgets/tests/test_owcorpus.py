import numpy as np
from Orange.data import Table, Domain, StringVariable
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
        self.assertEqual(data.domain["Heading"], self.widget.title_variable)
        self.check_output("Heading")

    def test_title_selection_strategy(self):
        """
        With this test we test whether the selection strategy for a title
        attribute works correctly
        """
        # select the most unique
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b")]),
            np.empty((3, 0)),
            metas=[["a" * 10, "a" * 10],
                   ["a" * 10, "b" * 10],
                   ["a" * 10, "c" * 10]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(data.domain["b"], self.widget.title_variable)
        self.check_output("b")

        # select the uniquest and also short enough, here attribute a is not
        # suitable since it has too long title, and c is more unique than b
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 10, "a" * 10],
                   ["b" * 100, "a" * 10, "b" * 10],
                   ["c" * 100, "a" * 10, "b" * 10]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(data.domain["c"], self.widget.title_variable)
        self.check_output("c")

        # when no variable is short enough we just select the shortest
        # attribute
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.assertEqual(data.domain["c"], self.widget.title_variable)
        self.check_output("c")

