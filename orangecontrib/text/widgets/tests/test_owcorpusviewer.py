import unittest
from unittest import TestCase

import numpy as np
from AnyQt.QtCore import QItemSelectionModel, Qt
from AnyQt.QtTest import QSignalSpy

from Orange.data import StringVariable, Domain
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import BASE_TOKENIZER
from orangecontrib.text.widgets.owcorpusviewer import (
    OWCorpusViewer,
    DocumentListModel,
    DocumentsFilterProxyModel,
)


class TestDocumentListModel(TestCase):
    def test_empty(self):
        model = DocumentListModel()
        self.assertEqual(model.rowCount(), 0)
        self.assertListEqual(model.get_filter_content(), [])

    def test_data(self):
        model = DocumentListModel()
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        contents = ["bar", "foo", "bar foo"]
        model.setup_data(documents, contents)

        self.assertListEqual(model.get_filter_content(), contents)
        self.assertEqual(model.rowCount(), 3)

        self.assertEqual(model.data(model.index(0)), documents[0])
        self.assertEqual(model.data(model.index(1)), documents[1])
        self.assertEqual(model.data(model.index(2)), documents[2])

    def test_data_method(self):
        model = DocumentListModel()
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        contents = ["bar", "foo", "bar foo"]
        model.setup_data(documents, contents)

        self.assertEqual(model.data(model.index(0), Qt.DisplayRole), documents[0])
        self.assertEqual(model.data(model.index(1), Qt.DisplayRole), documents[1])
        self.assertEqual(model.data(model.index(2), Qt.DisplayRole), documents[2])

        self.assertEqual(model.data(model.index(0), Qt.UserRole), contents[0])
        self.assertEqual(model.data(model.index(1), Qt.UserRole), contents[1])
        self.assertEqual(model.data(model.index(2), Qt.UserRole), contents[2])

        self.assertIsNone(model.data(model.index(2), Qt.BackgroundRole))

    def test_update_filter_content(self):
        model = DocumentListModel()
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        contents = ["bar", "foo", "bar foo"]
        model.setup_data(documents, contents)

        model.update_filter_content(["a", "b", "c"])
        self.assertEqual(model.data(model.index(0), Qt.UserRole), "a")
        self.assertEqual(model.data(model.index(1), Qt.UserRole), "b")
        self.assertEqual(model.data(model.index(2), Qt.UserRole), "c")

        with self.assertRaises(AssertionError):
            model.update_filter_content(
                [
                    "a",
                    "b",
                ]
            )


class TestFilterModel(TestCase):
    def test_filter_model(self):
        model = DocumentListModel()
        filter_model = DocumentsFilterProxyModel()
        filter_model.setSourceModel(model)
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        contents = ["bar", "foo", "bar foo"]
        model.setup_data(documents, contents)

        # __regex is None - all data shown
        self.assertEqual(filter_model.rowCount(), 3)
        self.assertEqual(filter_model.data(filter_model.index(0, 0)), documents[0])
        self.assertEqual(filter_model.data(filter_model.index(1, 0)), documents[1])
        self.assertEqual(filter_model.data(filter_model.index(2, 0)), documents[2])

        # with regex set
        filter_model.set_filter_string("bar")
        self.assertEqual(filter_model.rowCount(), 2)
        self.assertEqual(filter_model.data(filter_model.index(0, 0)), documents[0])
        self.assertEqual(filter_model.data(filter_model.index(1, 0)), documents[2])

    def test_empty_model(self):
        model = DocumentListModel()
        filter_model = DocumentsFilterProxyModel()
        filter_model.setSourceModel(model)
        self.assertEqual(filter_model.rowCount(), 0)


class TestCorpusViewerWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCorpusViewer)
        self.corpus = Corpus.from_file("deerwester")

    def test_data(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(len(self.widget.corpus), 9)
        self.widget.doc_list.selectAll()
        out_corpus = self.get_output(self.widget.Outputs.matching_docs)
        np.testing.assert_array_equal(out_corpus.X, self.corpus.X)
        np.testing.assert_array_equal(out_corpus.Y, self.corpus.Y)
        np.testing.assert_array_equal(out_corpus.metas, self.corpus.metas)
        np.testing.assert_array_equal(out_corpus._tokens, self.corpus._tokens)

    def test_search(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.regexp_filter = "Human"
        self.widget.refresh_search()
        self.process_events()
        out_corpus = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(len(out_corpus), 1)
        self.assertEqual(int(self.widget.n_matches), 7)

        # first document is selected, when filter with word that is not in
        # selected document, first of shown documents is selected
        self.widget.regexp_filter = "graph"
        self.widget.refresh_search()
        self.process_events()
        self.assertEqual(1, len(self.get_output(self.widget.Outputs.matching_docs)))
        # word count doesn't depend on selection
        self.assertEqual(int(self.widget.n_matches), 7)

        # when filter is removed, matched words is 0
        self.widget.regexp_filter = ""
        self.widget.refresh_search()
        self.process_events()
        self.wait_until_finished()
        self.assertEqual(int(self.widget.n_matches), 0)

    def test_invalid_regex(self):
        # Error is shown when invalid regex is entered
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.regexp_filter = "*"
        self.widget.refresh_search()
        self.process_events()
        self.assertEqual(self.widget.n_matches, "n/a")
        self.assertTrue(self.widget.Error.invalid_regex.is_shown())
        # Error is hidden when valid regex is entered
        self.widget.regexp_filter = "graph"
        self.widget.refresh_search()
        self.process_events()
        self.assertFalse(self.widget.Error.invalid_regex.is_shown())      
   
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
        documents = [{"content": """царстве есть сад с молодильными яблоками"""}]
        metas = [
            (StringVariable("content"), lambda doc: doc.get("content")),
        ]
        dataset_name = "RussianDocument"
        corpus = Corpus.from_documents(documents, dataset_name, metas=metas)

        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.widget.regexp_filter = "\\bсад\\b"
        self.widget.refresh_search()
        self.process_events()
        self.widget.doc_webview.html()
        spy = QSignalSpy(self.widget.doc_webview.loadFinished)
        spy.wait()
        html = self.widget.doc_webview.html()
        self.assertIn('<mark data-markjs="true">', html)

    def test_output(self):
        """Output is intersection between selection and filter"""
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.regexp_filter = "graph"
        self.widget.refresh_search()
        self.process_events()
        # when intersection is empty automatically select first document shown
        mathing = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(1, len(mathing))
        self.assertEqual(
            mathing.get_column("Text")[0],
            "The generation of random binary unordered trees",
        )
        self.assertEqual(8, len(self.get_output(self.widget.Outputs.other_docs)))
        self.assertEqual(
            len(self.corpus.domain.metas) + 1,
            len(self.get_output(self.widget.Outputs.corpus).domain.metas),
        )

        self.widget.doc_list.selectAll()  # selects current documents in list
        self.assertEqual(4, len(self.get_output(self.widget.Outputs.matching_docs)))
        self.assertEqual(5, len(self.get_output(self.widget.Outputs.other_docs)))
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(self.get_output(self.widget.Outputs.matching_docs)),
            sum(output.get_column("Selected")),
        )

        self.widget.regexp_filter = "human"
        self.widget.refresh_search()
        self.process_events()
        # when intersection is empty automatically select first document shown
        mathing = self.get_output(self.widget.Outputs.matching_docs)
        self.assertEqual(1, len(mathing))
        self.assertEqual(
            mathing.get_column("Text")[0],
            "Human machine interface for lab abc computer applications",
        )
        self.assertEqual(8, len(self.get_output(self.widget.Outputs.other_docs)))
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(1, sum(output.get_column("Selected")))

        self.widget.doc_list.selectAll()
        self.assertEqual(5, len(self.get_output(self.widget.Outputs.matching_docs)))
        self.assertEqual(4, len(self.get_output(self.widget.Outputs.other_docs)))
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(self.get_output(self.widget.Outputs.matching_docs)),
            sum(output.get_column("Selected")),
        )

        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertIsNone(self.get_output(self.widget.Outputs.matching_docs))
        self.assertIsNone(self.get_output(self.widget.Outputs.other_docs))
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    def test_empty_corpus(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus[:0])
        self.assertSetEqual(self.widget.selected_documents, set())
        self.assertEqual(self.widget.doc_list.model().rowCount(), 0)

    def test_report(self):
        self.widget.send_report()

        self.widget.regexp_filter = "human"
        self.process_events()
        self.widget.send_report()

    def test_filter_attributes(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.filter_input.setText("graph")
        self.widget.refresh_search()

        # all attributes used for filtering (shown documents with "graph" in Category)
        doc_model = self.widget.doc_list.model()
        doc_shown = [
            doc_model.data(doc_model.index(i, 0)) for i in range(doc_model.rowCount())
        ]
        self.assertListEqual(
            doc_shown, ["Document 6", "Document 7", "Document 8", "Document 9"]
        )

        # only "Text" used for filtering (shown documents with "graph" in Text)
        slv = self.widget.search_listbox
        slv.selectionModel().select(
            slv.model().index(1), QItemSelectionModel.ClearAndSelect
        )
        doc_shown = [
            doc_model.data(doc_model.index(i, 0)) for i in range(doc_model.rowCount())
        ]
        self.assertListEqual(doc_shown, ["Document 7", "Document 8", "Document 9"])

    def test_filters_restored_from_context(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.filter_input.setText("graph")
        self.widget.refresh_search()
        slv = self.widget.search_listbox
        slv.selectionModel().select(
            slv.model().index(1), QItemSelectionModel.ClearAndSelect
        )
        self.assertListEqual(self.widget.search_features, [self.corpus.domain["Text"]])

        # send some other data to change values
        temp_corpus = Corpus.from_file("andersen")
        self.send_signal(self.widget.Inputs.corpus, temp_corpus)
        self.assertListEqual(self.widget.search_features, list(temp_corpus.domain))

        # test if corpus correctly restored for search_features
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertListEqual(self.widget.search_features, [self.corpus.domain["Text"]])
        self.assertEqual(self.widget.regexp_filter, "graph")

        # check if restored values correctly used for filtering
        # filter_conent must include only values from the text column
        self.assertListEqual(
            self.widget.doc_list_model.get_filter_content(),
            self.corpus.get_column("Text").tolist(),
        )
        # only "Text" used for filtering (shown documents with "graph" in Text)
        doc_model = self.widget.doc_list.model()
        doc_shown = [
            doc_model.data(doc_model.index(i, 0)) for i in range(doc_model.rowCount())
        ]
        self.assertListEqual(doc_shown, ["Document 7", "Document 8", "Document 9"])

    def test_data_only_hidden_attributes(self):
        for a in self.corpus.domain:
            a.attributes["hidden"] = True
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        # documents are shown but filter does not work
        self.assertEqual(self.widget.doc_list_model.rowCount(), 9)

    def test_token_checkbox(self):
        corpus_tokens = BASE_TOKENIZER(self.corpus)
        self.send_signal(self.widget.Inputs.corpus, corpus_tokens)
        self.assertTrue(self.widget.show_tokens_checkbox.isEnabled())
        self.assertFalse(self.widget.show_tokens_checkbox.isChecked())

        self.widget.show_tokens_checkbox.setChecked(True)
        self.assertTrue(self.widget.show_tokens_checkbox.isChecked())

        # if corpus without tokens on the input button is dissabled and unchecked
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertFalse(self.widget.show_tokens_checkbox.isChecked())
        self.assertFalse(self.widget.show_tokens_checkbox.isEnabled())

    def test_image(self):
        im_attr = StringVariable("Image")
        im_attr.attributes["origin"] = "/path/to/image"
        im_attr.attributes["type"] = "image"
        domain = self.corpus.domain
        im_corpus = self.corpus.transform(
            Domain(domain.attributes, metas=domain.metas + (im_attr,))
        )
        with im_corpus.unlocked(im_corpus.metas):
            im_corpus[:, im_attr] = np.array(["image_name"] + [""] * 8).reshape(-1, 1)
        self.send_signal(self.widget.Inputs.corpus, im_corpus)
        # tried to get content from the view to test correctness and cannot find
        # a nice way also patching does not work on all systems, just testing
        # that having image in corpus does not fail

    def test_migrate_settings(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        packed_data = self.widget.settingsHandler.pack_data(self.widget)
        context = packed_data["context_settings"][0]
        # we borrow display_features from setting extracted from widget, it
        # contains Category and Text
        context.values["display_indices"] = [0]
        context.values["search_indices"] = [1]
        context.values["__version__"] = 1
        context.attributes = tuple(context.attributes.items())
        context.attributes = context.attributes
        self.widget = self.create_widget(
            OWCorpusViewer,
            stored_settings={"context_settings": [context], "__version__": 1},
        )
        self.send_signal(self.widget.Inputs.corpus, self.corpus, widget=self.widget)
        domain = self.corpus.domain
        self.assertListEqual(self.widget.display_features, [domain["Category"]])
        self.assertListEqual(self.widget.search_features, [domain["Text"]])


if __name__ == "__main__":
    unittest.main()
