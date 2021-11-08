# pylint: disable=missing-docstring
import unittest
from typing import List
from unittest.mock import Mock, patch

from AnyQt.QtCore import Qt
from AnyQt.QtTest import QSignalSpy
from AnyQt.QtWidgets import QTableView

from Orange.data import StringVariable, Table, Domain
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text import Corpus
from orangecontrib.text.semantic_search import SemanticSearch
from orangecontrib.text.widgets.owsemanticviewer import OWSemanticViewer, \
    run, DisplayDocument


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
        self.corpus = Corpus.from_file("book-excerpts")
        self.state = Mock()
        self.state.is_interruption_requested = Mock(return_value=False)

    def test_run(self):
        words = ["foo", "graph", "minors", "trees"]
        results = run(self.corpus, words, self.state)
        self.assertIsInstance(results.scores, list)
        self.assertEqual(len(results.scores), len(self.corpus.documents))

    def test_run_no_data(self):
        results = run(None, None, Mock())
        self.assertEqual(results.scores, [])

        results = run([], [], Mock())
        self.assertEqual(results.scores, [])

        results = run(self.corpus, [], Mock())
        self.assertEqual(results.scores, [])

        results = run(None, ["foo", "bar"], Mock())
        self.assertEqual(results.scores, [])

    def test_run_interrupt(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=True)
        self.assertRaises(Exception, run, self.corpus, ["foo", "bar"], state)


class TestDisplayDocument(unittest.TestCase):
    def test_purge_consecutive(self):
        collection = ["...", "...", "foo", "...", "...", "..."]
        purged = DisplayDocument._purge(collection)
        self.assertEqual(purged, ["...", "foo", "..."])

        collection = ["...", "...", "foo", "foo", "...", "..."]
        purged = DisplayDocument._purge(collection)
        self.assertEqual(purged, ["...", "foo", "foo", "..."])

        collection = ["foo", "...", "...", "...", "..."]
        purged = DisplayDocument._purge(collection)
        self.assertEqual(purged, ["foo", "..."])

        collection = ["...", "...", "...", "...", "foo"]
        purged = DisplayDocument._purge(collection)
        self.assertEqual(purged, ["...", "foo"])

        collection = ["...", "...", "...", "foo", "...", "...", "...",
                      "bar", "...", "..."]
        purged = DisplayDocument._purge(collection)
        self.assertEqual(purged, ["...", "foo", "...", "bar", "..."])

    def test_tag_text(self):
        text = "Human machine interface for lab abc computer applications." \
               " A survey of user opinion of computer system response time." \
               "\nThe EPS user interface management system."
        matches = [(0, 58), (118, 159)]
        tagged = DisplayDocument._tag_text(text, matches)
        text = "<mark data-markjs='true'>Human machine interface for lab abc" \
               " computer applications.</mark>" \
               " A survey of user opinion of computer system response time." \
               "\n<mark data-markjs='true'>The EPS user interface management" \
               " system.</mark>"
        self.assertEqual(tagged, text)

    def test_display_document_doc(self):
        disp_type = DisplayDocument.Document
        text = "Human machine interface for lab abc computer applications." \
               " A survey of user opinion of computer system response time." \
               "\nThe EPS user interface management system."
        new_text = DisplayDocument(disp_type)(text, [])
        self.assertEqual(new_text, text)

        new_text = DisplayDocument(disp_type)(text, [(0, 58)])
        text1 = "<mark data-markjs='true'>Human machine interface for lab " \
                "abc computer applications.</mark>" \
                " A survey of user opinion of computer system response time." \
                "\nThe EPS user interface management system."
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(118, 159)])
        text1 = "Human machine interface for lab abc" \
                " computer applications." \
                " A survey of user opinion of computer system response time." \
                "\n<mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(0, 58), (118, 159)])
        text1 = "<mark data-markjs='true'>Human machine interface for lab " \
                "abc computer applications.</mark>" \
                " A survey of user opinion of computer system response time." \
                "\n<mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

    def test_display_document_section(self):
        disp_type = DisplayDocument.Section
        text = "Human machine interface for lab abc computer applications." \
               " A survey of user opinion of computer system response time." \
               "\nThe EPS user interface management system."
        new_text = DisplayDocument(disp_type)(text, [])
        self.assertEqual(new_text, "...")

        new_text = DisplayDocument(disp_type)(text, [(0, 58)])
        text1 = "<mark data-markjs='true'>Human machine interface for lab " \
                "abc computer applications.</mark>" \
                " A survey of user opinion of computer system response time." \
                "\n..."
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(59, 117)])
        text1 = "Human machine interface for lab abc computer applications. " \
                "<mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark>\n..."
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(118, 159)])
        text1 = "...\n<mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(0, 58), (59, 117)])
        text1 = "<mark data-markjs='true'>Human machine interface " \
                "for lab abc computer applications.</mark> " \
                "<mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark>\n..."
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(0, 58), (118, 159)])
        text1 = "<mark data-markjs='true'>Human machine interface for lab " \
                "abc computer applications.</mark>" \
                " A survey of user opinion of computer system response time." \
                "\n<mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(59, 117), (118, 159)])
        text1 = "Human machine interface for lab abc computer applications. " \
                "<mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark>" \
                "\n<mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(
            text, [(0, 58), (59, 117), (118, 159)])
        text1 = "<mark data-markjs='true'>Human machine interface " \
                "for lab abc computer applications.</mark> " \
                "<mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark>" \
                "\n<mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        text = "Human machine interface for lab abc computer applications." \
               " A survey of user opinion of computer system response time." \
               "\n\nThe EPS user interface management system."

        new_text = DisplayDocument(disp_type)(
            text, [(0, 58), (59, 117), (119, 160)])
        text1 = "<mark data-markjs='true'>Human machine interface " \
                "for lab abc computer applications.</mark> " \
                "<mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark>" \
                "\n<mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        text = "Human machine interface for lab abc computer applications." \
               " A survey of user opinion of computer system response time:" \
               "\n - survey of user opinion" \
               "\n - survey of user opinion" \
               "\nThe EPS user interface management system."

        new_text = DisplayDocument(disp_type)(text, [(0, 58), (59, 169)])
        text1 = "<mark data-markjs='true'>Human machine interface " \
                "for lab abc computer applications.</mark> " \
                "<mark data-markjs='true'>A survey of user opinion of" \
                " computer system response time:" \
                "\n - survey of user opinion" \
                "\n - survey of user opinion</mark>\n..."
        self.assertEqual(new_text, text1)

    def test_display_document_sentence(self):
        disp_type = DisplayDocument.Sentence
        text = "Human machine interface for lab abc computer applications." \
               " A survey of user opinion of computer system response time." \
               "\nThe EPS user interface management system."
        new_text = DisplayDocument(disp_type)(text, [])
        self.assertEqual(new_text, "...")

        new_text = DisplayDocument(disp_type)(text, [(0, 58)])
        text1 = "<mark data-markjs='true'>Human machine interface for lab " \
                "abc computer applications.</mark> ... \n"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(59, 117)])
        text1 = "... <mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark> \n ..."
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(118, 159)])
        text1 = "... \n <mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(0, 58), (59, 117)])
        text1 = "<mark data-markjs='true'>Human machine interface " \
                "for lab abc computer applications.</mark> " \
                "<mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark> \n ..."
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(0, 58), (118, 159)])
        text1 = "<mark data-markjs='true'>Human machine interface for lab " \
                "abc computer applications.</mark> ... " \
                "\n <mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(text, [(59, 117), (118, 159)])
        text1 = "... <mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark> " \
                "\n <mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)

        new_text = DisplayDocument(disp_type)(
            text, [(0, 58), (59, 117), (118, 159)])
        text1 = "<mark data-markjs='true'>Human machine interface " \
                "for lab abc computer applications.</mark> " \
                "<mark data-markjs='true'>A survey of user opinion of " \
                "computer system response time.</mark> " \
                "\n <mark data-markjs='true'>The EPS user interface " \
                "management system.</mark>"
        self.assertEqual(new_text, text1)


class DummySearch(SemanticSearch):
    def __call__(self, *args, **kwargs):
        return [
            [[[0, 57], 0.79111683368682]],
            [[[0, 57], 0.63561916351318]],
            [[[0, 40], 0.58700573444366]],
            [[[0, 50], 0.60116827487945]],
            [[[0, 61], 0.66357374191284]],
            [[[0, 47], 0.56377774477005]],
            [[[0, 40], 0.37049713730812]],
            [[[0, 55], 0.46417143940925]],
            [[[0, 12], 0.46417143940925]]
        ]


class TestOWSemanticViewer(WidgetTest):
    def setUp(self):
        self.patcher = patch("orangecontrib.text.widgets.owsemanticviewer."
                             "SemanticSearch", new=DummySearch)
        self.patcher.start()
        self.widget = self.create_widget(OWSemanticViewer)
        self.corpus = Corpus.from_file("deerwester")
        self.words = create_words_table(["foo", "graph", "minors", "trees"])

    def tearDown(self):
        self.widget.cancel()
        self.patcher.stop()

    def test_table(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)
        self.wait_until_finished()

        model = self.widget._list_view.model()
        table = [["0", "0.791", "Document 1"],
                 ["0", "0.636", "Document 2"],
                 ["0", "0.587", "Document 3"],
                 ["0", "0.601", "Document 4"],
                 ["0", "0.664", "Document 5"],
                 ["1", "0.564", "Document 6"],
                 ["2", "0.370", "Document 7"],
                 ["3", "0.464", "Document 8"],
                 ["2", "0.464", "Document 9"]]
        for i in range(len(self.corpus)):
            for j in range(model.columnCount()):
                self.assertEqual(model.data(model.index(i, j)), table[i][j])

    def test_webview(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)
        self.wait_until_finished()

        self.process_events()
        spy = QSignalSpy(self.widget._web_view.loadFinished)
        spy.wait()
        html = self.widget._web_view.html()
        text = "Human machine interface for lab abc computer applications"
        self.assertIn(text, html)
        self.assertIn(f'<mark data-markjs="true">{text}', html)

    def test_outputs(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)
        self.wait_until_finished()

        matching_docs = self.get_output(self.widget.Outputs.matching_docs)
        other_docs = self.get_output(self.widget.Outputs.other_docs)
        corpus = self.get_output(self.widget.Outputs.corpus)

        self.assertIsInstance(matching_docs, Corpus)
        self.assertIsInstance(other_docs, Corpus)
        self.assertIsInstance(corpus, Corpus)

        self.assertEqual(len(matching_docs), 1)
        self.assertEqual(len(other_docs), 8)
        self.assertEqual(len(corpus), 9)

    def test_clear(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)
        self.wait_until_finished()

        self.assertEqual(self.widget.selection, [0])
        self.assertIsNotNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.other_docs))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.corpus))

        self.send_signal(self.widget.Inputs.corpus, None)
        self.wait_until_finished()

        self.assertEqual(self.widget.selection, [])
        self.assertIsNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertIsNone(self.get_output(self.widget.Outputs.other_docs))
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()

        self.assertEqual(self.widget.selection, [0])
        self.assertIsNotNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.other_docs))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.corpus))

        self.send_signal(self.widget.Inputs.words, None)
        self.wait_until_finished()

        self.assertEqual(self.widget.selection, [])
        self.assertIsNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.other_docs))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.corpus))

    def test_sorted_table_selection(self):
        self.widget.controls.threshold.setValue(1)

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)
        self.wait_until_finished()

        matching_docs = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(matching_docs.ids, [self.corpus.ids[0]])

        self.process_events()
        spy = QSignalSpy(self.widget._web_view.loadFinished)
        spy.wait()
        text = "Human machine interface for lab abc computer applications"
        self.assertIn(text, self.widget._web_view.html())

        # sort table by Score (asc)
        header = self.widget._list_view.horizontalHeader()
        header.setSortIndicator(1, Qt.AscendingOrder)
        header.sectionClicked.emit(1)

        # selection: [0, 6, 8]
        self.widget._list_view.setSelectionMode(QTableView.MultiSelection)
        self.widget._list_view.selectRow(2)
        self.widget._list_view.selectRow(0)

        matching_docs = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(list(matching_docs.ids),
                         list(self.corpus.ids[[0, 6, 8]]))

        # docs are sorted by Score (asc)
        self.process_events()
        spy = QSignalSpy(self.widget._web_view.loadFinished)
        spy.wait()
        text = "The intersection graph of paths in trees" \
               "</p><hr><p>Graph minors A survey</p><hr><p>" \
               "Human machine interface for lab abc computer applications"
        self.assertIn(text, self.widget._web_view.html())

        # sort table by Score (desc)
        header = self.widget._list_view.horizontalHeader()
        header.setSortIndicator(1, Qt.DescendingOrder)
        header.sectionClicked.emit(1)

        self.process_events()
        spy = QSignalSpy(self.widget._web_view.loadFinished)
        spy.wait()
        text = "Human machine interface for lab abc computer applications" \
               "</p><hr><p>Graph minors A survey</p><hr><p>" \
               "The intersection graph of paths in trees"
        self.assertIn(text, self.widget._web_view.html())

    def test_send_report(self):
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.send_report()

        self.send_signal(self.widget.Inputs.words, self.words)
        self.wait_until_finished()
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
