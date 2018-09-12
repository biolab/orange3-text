import unittest
from AnyQt.QtTest import QSignalSpy
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owcorpusviewer import OWCorpusViewer


class TestCorpusViewerWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCorpusViewer)
        self.corpus = Corpus.from_file('deerwester')

    def test_data(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(self.widget.n_documents, 9)
        out_corpus = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(out_corpus, self.corpus)

    def test_search(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.regexp_filter = "graph"
        self.process_events()
        out_corpus = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(len(out_corpus), 4)

    def test_highlighting(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.regexp_filter = "graph"
        self.process_events()
        self.widget.doc_webview.html()
        spy = QSignalSpy(self.widget.doc_webview.loadFinished)
        spy.wait()
        html = self.widget.doc_webview.html()
        self.assertIn('<mark data-markjs="true">', html)


if __name__ == "__main__":
    unittest.main()
