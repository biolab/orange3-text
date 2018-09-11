import unittest
from unittest.mock import Mock

from AnyQt.QtCore import QModelIndex, QItemSelection, Qt
from AnyQt.QtGui import QBrush, QColor

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owconcordance import ConcordanceModel, \
                                                     OWConcordance


class TestConcordanceModel(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')

    def test_data(self):
        """data returns proper text to display"""
        model = ConcordanceModel()
        model.set_width(2)
        self.assertEqual(model.rowCount(QModelIndex()), 0)

        model.set_corpus(self.corpus)
        model.set_word("of")

        # The same document in two rows
        self.assertEqual(model.rowCount(QModelIndex()), 7)
        self.assertEqual(model.data(model.index(0, 0)), "A survey")
        self.assertEqual(model.data(model.index(0, 1)), "of")
        self.assertEqual(model.data(model.index(0, 2)), "user opinion")

        self.assertEqual(model.data(model.index(1, 0)), "user opinion")
        self.assertEqual(model.data(model.index(1, 1)), "of")
        self.assertEqual(model.data(model.index(1, 2)), "computer system")

        # The third column has just a single word
        self.assertEqual(model.data(model.index(2, 0)), "engineering testing")
        self.assertEqual(model.data(model.index(2, 1)), "of")
        self.assertEqual(model.data(model.index(2, 2)), "EPS")

        # The first column has just a single word
        self.assertEqual(model.data(model.index(3, 0)), "Relation")
        self.assertEqual(model.data(model.index(3, 1)), "of")
        self.assertEqual(model.data(model.index(3, 2)), "user perceived")

    def test_data_non_displayroles(self):
        """Other possibly implemented roles return correct types"""
        model = ConcordanceModel()
        model.set_corpus(self.corpus)
        model.set_word("of")
        ind00 = model.index(0, 0)
        self.assertIsInstance(model.data(ind00, Qt.ForegroundRole),
                              (QBrush, type(None)))
        self.assertIsInstance(model.data(ind00, Qt.BackgroundRole),
                              (QColor, type(None)))
        self.assertIsInstance(model.data(ind00, Qt.TextAlignmentRole),
                              (Qt.Alignment, type(None)))

    def test_color_proper_rows(self):
        """Rows are colored corresponding to the document"""
        model = ConcordanceModel()
        model.set_width(2)
        model.set_corpus(self.corpus)
        model.set_word("of")

        color1 = model.data(model.index(0, 0), Qt.BackgroundRole)
        self.assertEqual(model.data(model.index(1, 0), Qt.BackgroundRole),
                         color1)
        self.assertNotEqual(model.data(model.index(2, 0), Qt.BackgroundRole),
                            color1)

    def test_order_doesnt_matter(self):
        """Setting the word or the corpus first works"""
        model = ConcordanceModel()
        model.set_width(2)
        self.assertEqual(model.rowCount(QModelIndex()), 0)
        model.set_corpus(self.corpus)
        self.assertEqual(model.rowCount(QModelIndex()), 0)
        model.set_word("of")
        self.assertEqual(model.rowCount(QModelIndex()), 7)
        model.set_word("")
        self.assertEqual(model.rowCount(QModelIndex()), 0)
        model.set_word(None)
        self.assertEqual(model.rowCount(QModelIndex()), 0)
        model.set_corpus(None)
        self.assertEqual(model.rowCount(QModelIndex()), 0)
        model.set_word("of")
        self.assertEqual(model.rowCount(QModelIndex()), 0)
        model.set_corpus(self.corpus)
        self.assertEqual(model.rowCount(QModelIndex()), 7)
        model.set_corpus(None)
        self.assertEqual(model.rowCount(QModelIndex()), 0)
        model.set_corpus(None)
        self.assertEqual(model.rowCount(QModelIndex()), 0)

    def test_set_word(self):
        """Concondance.set_word resets the indices"""
        # Human machine interface for lab abc computer applications
        model = ConcordanceModel()
        model.set_corpus(self.corpus)
        model.set_width(2)

        model.set_word("of")
        self.assertEqual(model.rowCount(QModelIndex()), 7)
        self.assertEqual(model.data(model.index(0, 0)), "A survey")

        model.set_word("lab")
        self.assertEqual(model.rowCount(QModelIndex()), 1)
        self.assertEqual(model.data(model.index(0, 0)), "interface for")

        model.set_word(None)
        self.assertEqual(model.rowCount(QModelIndex()), 0)

    def test_signals(self):
        """Setting corpus or word sets the proper signals"""
        model = ConcordanceModel()
        model.set_corpus(self.corpus)

        toBeReset = Mock()
        hasBeenReset = Mock()
        model.modelAboutToBeReset.connect(toBeReset)
        model.modelReset.connect(hasBeenReset)
        model.set_corpus(None)
        # toBeReset.assert_called_once() is only available from Py 3.6
        self.assertEqual(toBeReset.call_count, 1)
        self.assertEqual(hasBeenReset.call_count, 1)

        toBeReset.reset_mock()
        hasBeenReset.reset_mock()
        model.set_word(None)
        self.assertEqual(toBeReset.call_count, 1)
        self.assertEqual(hasBeenReset.call_count, 1)

    def test_matching_docs(self):
        model = ConcordanceModel()
        model.set_word("of")
        model.set_corpus(self.corpus)
        self.assertEqual(model.matching_docs(), 6)

    def test_concordance_output(self):
        model = ConcordanceModel()
        model.set_word("of")
        model.set_corpus(self.corpus)
        self.assertEqual(len(model.get_data()), 7)


class TestConcordanceWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWConcordance)  # type: OWConcordance
        self.corpus = Corpus.from_file('deerwester')

    def test_set_corpus(self):
        self.widget.model.set_corpus = set_corpus = Mock()
        self.send_signal("Corpus", self.corpus)
        set_corpus.assert_called_with(self.corpus)

        self.send_signal("Corpus", None)
        set_corpus.assert_called_with(None)

    def test_set_word(self):
        self.widget.model.set_word = set_word = Mock()
        self.widget.controls.word.setText("foo")
        set_word.assert_called_with("foo")
        self.widget.controls.word.setText("")
        set_word.assert_called_with("")

    def test_set_width(self):
        self.widget.model.set_width = set_width = Mock()
        self.widget.controls.context_width.setValue(4)
        set_width.assert_called_with(4)

    def test_selection(self):
        self.send_signal("Corpus", self.corpus)
        widget = self.widget
        widget.controls.word.setText("of")
        view = self.widget.conc_view

        # Select one row, two are selected, one document on the output
        view.selectRow(1)
        self.assertEqual({ind.row() for ind in view.selectedIndexes()},
                         {0, 1})
        self.assertEqual(self.get_output("Selected Documents"),
                         self.corpus[[1]])

        # Select a single row
        view.selectRow(3)
        self.assertEqual({ind.row() for ind in view.selectedIndexes()},
                         {3})
        self.assertEqual(self.get_output("Selected Documents"),
                         self.corpus[[4]])

        # Add a "double" row, three are selected, two documents on the output
        selection_model = view.selectionModel()
        selection = QItemSelection()
        ind00 = widget.model.index(0, 0)
        selection.select(ind00, ind00)
        selection_model.select(selection, selection_model.Select)
        self.assertEqual({ind.row() for ind in view.selectedIndexes()},
                         {0, 1, 3})
        self.assertEqual(self.get_output("Selected Documents"),
                         self.corpus[[1, 4]])

        # Clear selection by clicking outside
        ind_10 = widget.model.index(-1, 0)
        selection_model.select(ind_10, selection_model.Select)
        self.assertIsNone(self.get_output("Selected Documents"))

        # Selected rows emptied after word change
        view.selectRow(3)
        self.assertTrue(view.selectedIndexes())
        widget.controls.word.setText("o")
        self.assertFalse(view.selectedIndexes())

    def test_signal_to_none(self):
        self.send_signal("Corpus", self.corpus)
        widget = self.widget
        widget.controls.word.setText("of")
        view = self.widget.conc_view
        nrows = widget.model.rowCount()
        view.selectRow(1)

        self.send_signal("Corpus", None)
        self.assertIsNone(self.get_output("Selected Documents"))
        self.assertEqual(widget.model.rowCount(), 0)
        self.assertEqual(widget.controls.word.text(), "")

        self.send_signal("Corpus", self.corpus)
        self.assertEqual(widget.model.rowCount(), nrows)


if __name__ == "__main__":
    unittest.main()
