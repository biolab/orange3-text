import unittest
from AnyQt.QtTest import QSignalSpy
from Orange.widgets.tests.base import WidgetTest
from Orange.data import StringVariable

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owcorpusviewer import OWCorpusViewer


class TestCorpusViewerWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCorpusViewer)
        self.corpus = Corpus.from_file('deerwester')

    def test_data(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(self.widget.n_documents, 9)
        self.widget.doc_list.selectAll()
        out_corpus = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(out_corpus, self.corpus)

    def test_search(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.regexp_filter = "Human"
        self.process_events()
        out_corpus = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(len(out_corpus), 1)

        # first document is selected, when filter with word that is not in
        # selected document out_corpus is None
        self.widget.regexp_filter = "graph"
        self.process_events()
        out_corpus = self.get_output(self.widget.Outputs.matching_docs)
        self.assertIsNone(out_corpus)

    def test_highlighting(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        # no intersection between filter and selection
        self.widget.regexp_filter = "graph"
        self.process_events()
        spy = QSignalSpy(self.widget.doc_webview.loadFinished)
        spy.wait()
        html = self.widget.doc_webview.html()
        self.assertNotIn('<mark data-markjs="true">', html)

        # all documents are selected
        self.widget.regexp_filter = "graph"
        self.widget.doc_list.selectAll()
        spy = QSignalSpy(self.widget.doc_webview.loadFinished)
        spy.wait()
        html = self.widget.doc_webview.html()
        self.assertIn('<mark data-markjs="true">', html)

    def test_highlighting_non_latin(self):
        documents = [
            {
                'content': """царстве есть сад с молодильными яблоками"""
            }
        ]
        metas = [
            (StringVariable('content'), lambda doc: doc.get('content')),
        ]
        dataset_name = 'RussianDocument'
        corpus = Corpus.from_documents(documents, dataset_name, metas=metas)

        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.widget.regexp_filter = "\\bсад\\b"
        self.process_events()
        self.widget.doc_webview.html()
        spy = QSignalSpy(self.widget.doc_webview.loadFinished)
        spy.wait()
        html = self.widget.doc_webview.html()
        self.assertIn('<mark data-markjs="true">', html)

    def test_output(self):
        """ Output is intersection between selection and filter """
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.regexp_filter = "graph"
        self.process_events()
        self.assertIsNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertEqual(
            9, len(self.get_output(self.widget.Outputs.other_docs))
        )

        self.widget.doc_list.selectAll()  # selects current documents in list
        self.assertEqual(
            4, len(self.get_output(self.widget.Outputs.matching_docs))
        )
        self.assertEqual(
            5, len(self.get_output(self.widget.Outputs.other_docs))
        )

        self.widget.regexp_filter = "human"
        self.process_events()
        # empty because none of mathching documents is selected
        self.assertIsNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertEqual(
            9, len(self.get_output(self.widget.Outputs.other_docs))
        )

        self.widget.doc_list.selectAll()
        self.assertEqual(
            5, len(self.get_output(self.widget.Outputs.matching_docs))
        )
        self.assertEqual(
            4, len(self.get_output(self.widget.Outputs.other_docs))
        )

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertIsNone(self.get_output(self.widget.Outputs.other_docs))

    def test_report(self):
        self.widget.send_report()

        self.widget.regexp_filter = "human"
        self.process_events()
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
