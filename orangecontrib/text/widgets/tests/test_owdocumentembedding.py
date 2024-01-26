import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtWidgets import QComboBox, QRadioButton
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError

from orangecontrib.text.language import DEFAULT_LANGUAGE, ISO2LANG
from orangecontrib.text.tests.test_documentembedder import PATCH_METHOD, make_dummy_post
from orangecontrib.text.vectorization.document_embedder import (
    DocumentEmbedder,
    LANGUAGES,
)
from orangecontrib.text.vectorization.sbert import EMB_DIM, SBERT
from orangecontrib.text.widgets.owdocumentembedding import OWDocumentEmbedding
from orangecontrib.text import Corpus


async def none_method(_, __):
    return None

_response_list = str(np.arange(0, EMB_DIM, dtype=float).tolist())
SBERT_RESPONSE = f'{{"embedding": {_response_list}}}'.encode()


class TestOWDocumentEmbedding(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWDocumentEmbedding)
        self.corpus = Corpus.from_file('deerwester')
        self.larger_corpus = Corpus.from_file('book-excerpts')

        # test on fastText, except for tests that change the setting
        self.widget.findChildren(QRadioButton)[1].click()
        SBERT().clear_cache()
        DocumentEmbedder.clear_cache("en")
        DocumentEmbedder.clear_cache("sl")

    def tearDown(self):
        SBERT().clear_cache()
        DocumentEmbedder.clear_cache("en")
        DocumentEmbedder.clear_cache("sl")

    def test_input(self):
        set_data = self.widget.set_data = Mock()
        self.send_signal("Corpus", None)
        set_data.assert_called_with(None)
        sample = self.corpus[:0]
        self.send_signal("Corpus", sample)
        set_data.assert_called_with(sample)
        self.send_signal("Corpus", self.corpus)
        set_data.assert_called_with(self.corpus)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_output(self):
        self.send_signal("Corpus", None)
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

        self.send_signal("Corpus", self.corpus)
        result = self.get_output(self.widget.Outputs.corpus)
        self.wait_until_finished()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(self.corpus), len(result))

    @patch(PATCH_METHOD, make_dummy_post(b''))
    def test_some_failed(self):
        simulate.combobox_activate_index(
            self.widget.controlArea.findChildren(QComboBox)[0], 1
        )
        with self.assertWarns(RuntimeWarning):  # avoid warnings in test logs
            self.send_signal("Corpus", self.corpus)
            self.wait_until_finished()
        result = self.get_output(self.widget.Outputs.corpus)
        skipped = self.get_output(self.widget.Outputs.skipped)
        self.assertIsNone(result)
        self.assertEqual(len(skipped), len(self.corpus))
        self.assertTrue(self.widget.Warning.unsuccessful_embeddings.is_shown())

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_cancel_embedding(self):
        self.send_signal("Corpus", self.larger_corpus)
        self.widget.cancel_button.click()
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    @patch('orangecontrib.text.vectorization.document_embedder' +
           '._ServerEmbedder.embedd_data',
           side_effect=EmbeddingConnectionError)
    def test_connection_error(self, _):
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))
        self.assertTrue(self.widget.Error.no_connection.is_shown())

    @patch(
        "orangecontrib.text.vectorization.document_embedder"
        + ".DocumentEmbedder.transform",
        side_effect=OSError,
    )
    def test_unexpected_error(self, _):
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))
        self.assertTrue(self.widget.Error.unexpected_error.is_shown())

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_rerun_on_new_data(self):
        """ Check if embedding is automatically re-run on new data """
        self.widget._auto_apply = False
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

        self.send_signal(self.widget.Inputs.corpus, self.corpus[:3])
        self.wait_until_finished()
        self.assertEqual(3, len(self.get_output(self.widget.Outputs.corpus)))

        self.send_signal(self.widget.Inputs.corpus, self.corpus[:1])
        self.wait_until_finished()
        self.assertEqual(1, len(self.get_output(self.widget.Outputs.corpus)))

    @patch('orangecontrib.text.vectorization.document_embedder' +
           '._ServerEmbedder._encode_data_instance', none_method)
    def test_skipped_documents(self):
        with self.assertWarns(RuntimeWarning):  # avoid warnings in test logs
            self.send_signal("Corpus", self.corpus)
            self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))
        self.assertEqual(len(self.get_output(self.widget.Outputs.skipped)), len(self.corpus))
        self.assertTrue(self.widget.Warning.unsuccessful_embeddings.is_shown())

    @patch(PATCH_METHOD, make_dummy_post(SBERT_RESPONSE))
    def test_sbert(self):
        self.widget.findChildren(QRadioButton)[0].click()
        SBERT().clear_cache()

        self.send_signal("Corpus", self.corpus)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(self.corpus), len(result))
        self.assertTupleEqual(self.corpus.domain.metas, result.domain.metas)
        self.assertEqual(384, len(result.domain.attributes))

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_corpus_name_preserved(self):
        # test on fasttext
        self.send_signal("Corpus", self.corpus)
        # just to make sure corpus already has a name
        self.assertEqual("deerwester", self.corpus.name)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertIsNotNone(result)
        self.assertEqual("deerwester", result.name)

        # test on sbert
        self.widget.findChildren(QRadioButton)[0].click()
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertIsNotNone(result)
        self.assertEqual("deerwester", result.name)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_fasttext_language(self):
        # english corpus
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("en", self.widget.language)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(9, len(result))

        # slovenian corpus
        self.corpus.attributes["language"] = "sl"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("sl", self.widget.language)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(9, len(result))

        # language none
        self.corpus.attributes["language"] = None
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        # use widgets default language English
        self.assertEqual(DEFAULT_LANGUAGE, self.widget.language)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(9, len(result))

        # language not supported
        self.corpus.attributes["language"] = "be"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        # use widgets default language English
        self.assertEqual(DEFAULT_LANGUAGE, self.widget.language)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(9, len(result))

        # language english
        self.corpus.attributes["language"] = "en"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("en", self.widget.language)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(9, len(result))

        # manually set language
        simulate.combobox_activate_item(
            self.widget.controlArea.findChildren(QComboBox)[0], "French"
        )
        self.assertEqual("fr", self.widget.language)
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(9, len(result))

        # providing new corpus should reset language
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual("en", self.widget.language)

    def test_language_from_settings(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        simulate.combobox_activate_item(
            self.widget.controlArea.findChildren(QComboBox)[0], "French"
        )
        self.assertEqual("fr", self.widget.language)
        settings = self.widget.settingsHandler.pack_data(self.widget)

        widget = self.create_widget(OWDocumentEmbedding, stored_settings=settings)
        self.send_signal(widget.Inputs.corpus, self.corpus, widget=widget)
        self.assertEqual("fr", widget.language)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    @patch("orangecontrib.text.widgets.owdocumentembedding.OWDocumentEmbedding.report_items")
    def test_report(self, mocked_items: Mock):
        self.widget.findChildren(QRadioButton)[0].click()
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.send_report()
        mocked_items.assert_called_once()
        mocked_items.reset_mock()

        self.widget.findChildren(QRadioButton)[1].click()
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.send_report()
        mocked_items.assert_called_once()
        mocked_items.reset_mock()

    def test_migrate_settings(self):
        for iso_lang in LANGUAGES:
            settings = {"__version__": 2, "language": ISO2LANG[iso_lang]}
            widget = self.create_widget(OWDocumentEmbedding, stored_settings=settings)
            self.assertEqual(iso_lang, widget.language)


if __name__ == "__main__":
    unittest.main()
