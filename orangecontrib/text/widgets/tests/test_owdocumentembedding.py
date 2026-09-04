import json
import unittest
from unittest.mock import Mock, patch, MagicMock

import openai

import numpy as np
from AnyQt.QtWidgets import QComboBox, QRadioButton
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError

from orangecontrib.text.language import DEFAULT_LANGUAGE, ISO2LANG
from orangecontrib.text.tests.test_documentembedder import PATCH_METHOD, make_dummy_post
from orangecontrib.text.vectorization import document_embedder
from orangecontrib.text.vectorization.document_embedder import (
    DocumentEmbedder,
    LANGUAGES,
)
from orangecontrib.text.vectorization.sbert import EMB_DIM, SBERT
from orangecontrib.text.vectorization.onnx_embedder import ONNXEmbedder
from orangecontrib.text.widgets.owdocumentembedding import OWDocumentEmbedding, Methods, OnnxModel
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
        self.widget.set_method(Methods.FastText)
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
        self.widget.set_method(Methods.SBERT)
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
        self.widget.set_method(Methods.SBERT)
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
        self.widget.set_method(Methods.SBERT)
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.send_report()
        mocked_items.assert_called_once()
        mocked_items.reset_mock()

        self.widget.set_method(Methods.FastText)
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


class TestOWDocumentEmbeddingOAI(WidgetTest):
    def setUp(self):
        super().setUp()
        self.widget = self.create_widget(OWDocumentEmbedding, stored_settings={
            "base_url": "localhost:8000", "model": "embedder", "method": 2
        })
        self.corpus = Corpus.from_file('deerwester')
        self.larger_corpus = Corpus.from_file('book-excerpts')

        # Disable embedder caches.
        cache = MagicMock()
        cache.md5_hash = lambda _: b""
        cache.get_cached_result_or_none = lambda _: None
        self._patch = patch.object(
            document_embedder, "EmbedderCache", MagicMock(return_value=cache)
        )
        self._patch.__enter__()

    def tearDown(self):
        self._patch.__exit__(None, None, None)
        super().tearDown()

    @patch("openai.OpenAI")
    def test_openai_output(self, mock_openai):
        """Test that OpenAI embedder (method 2) produces correct output."""
        # Mock the OpenAI client response
        mock_embedding = MagicMock()
        mock_embedding.embedding = np.arange(EMB_DIM, dtype=float).tolist()
        mock_response = MagicMock()
        mock_response.data = [mock_embedding] * len(self.corpus)
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        mock_openai.return_value = mock_embedding

        # Select OpenAI embedder (fourth radio button, index 3)
        self.widget.set_method(Methods.OpenAIEmbedder)
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        result = self.get_output(self.widget.Outputs.corpus)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(self.corpus), len(result))

    @patch("openai.OpenAI")
    def test_openai_authentication_error(self, mock_openai):
        """Test authentication error handling for OpenAI embedder."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {"error": {"message": "Invalid API key"}}
        ).encode()
        error = openai.AuthenticationError(
            "Invalid API key", response=mock_response, body=None
        )
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = error
        mock_openai.return_value = mock_client

        self.widget.set_method(Methods.OpenAIEmbedder)
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))
        self.assertTrue(self.widget.Error.authentication_error.is_shown())

    @patch("openai.OpenAI")
    def test_openai_api_spec_error(self, mock_openai):
        """Test API spec error handling for OpenAI embedder."""
        mock_response = MagicMock()
        mock_response.content = json.dumps(
            {"error": {"message": "Invalid model: bad-model"}}
        ).encode()
        error = openai.BadRequestError(
            "Bad request", response=mock_response, body=None
        )
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = error
        mock_openai.return_value = mock_client

        self.widget.set_method(Methods.OpenAIEmbedder)
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))
        self.assertTrue(self.widget.Error.api_spec_error.is_shown())

    @patch("openai.OpenAI")
    def test_openai_connection_error(self, mock_openai):
        """Test connection error handling for OpenAI embedder."""
        mock_request = MagicMock()
        error = openai.APIConnectionError(message="Connection refused", request=mock_request)
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = error
        mock_openai.return_value = mock_client

        self.widget.set_method(Methods.OpenAIEmbedder)
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))
        self.assertTrue(self.widget.Error.connection_error.is_shown())

    def test_report(self):
        self.widget.set_method(Methods.OpenAIEmbedder)
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.send_report()

    def test_onnx_radio_button_exists(self):
        """Test that the ONNX radio button is present."""
        rbs = self.widget.findChildren(QRadioButton)
        # Should have 4 radio buttons: SBERT, FastText, OpenAI, ONNX
        self.assertEqual(len(rbs), 4)
        # ONNX is the 4th radio button (index 3)
        self.assertEqual(rbs[3].text(), "Local ONNX Embedder:")

    @patch("huggingface_hub.hf_hub_download")
    @patch("onnxruntime.InferenceSession")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_onnx_init_method(self, mock_tokenizer, mock_session, mock_download):
        """Test that ONNX embedder is correctly configured via init_method."""
        # Mock the tokenizer
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3]] * len(self.corpus)),
            "attention_mask": np.array([[1, 1, 1]] * len(self.corpus)),
        }
        # Mock the ONNX session
        mock_session.return_value.get_inputs.return_value = [MagicMock(name="input_ids"), MagicMock(name="attention_mask")]
        mock_session.return_value.get_outputs.return_value = [MagicMock()]
        mock_session.return_value.get_outputs()[0].shape = (1, 1, 384)
        mock_session.return_value.run.return_value = [
            np.random.randn(len(self.corpus), 256, 384).astype(np.float32)
        ]

        # Select ONNX embedder
        self.widget.set_method(Methods.ONNXEmbedder)
        # Verify init_method returns an ONNXEmbedder instance
        method = self.widget.init_method()
        self.assertIsInstance(method, ONNXEmbedder)
        # Verify default model is sentence-transformers/all-MiniLM-L6-v2
        self.assertEqual(method.model_id, "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(method.model_filename, "onnx/model_quint8_avx2.onnx")

        # Switch to IBM Granite (index 1)
        self.widget.onnx_model = OnnxModel.IBM_GRANITE_97M_MULTILINGUAL.value
        self.widget.on_change()
        method = self.widget.init_method()
        self.assertEqual(method.model_id, "ibm-granite/granite-embedding-97m-multilingual-r2")
        self.assertEqual(method.model_filename, "onnx/model_quint8_avx2.onnx")

        # Switch to Snowflake (index 2)
        self.widget.onnx_model = OnnxModel.SNOWFLAKE_ARCTIC_EMBED_XS.value
        self.widget.on_change()
        method = self.widget.init_method()
        self.assertEqual(method.model_id, "Snowflake/snowflake-arctic-embed-xs")
        self.assertEqual(method.model_filename, "onnx/model_uint8.onnx")

    def test_onnx_report(self):
        """Test ONNX embedder report content."""
        self.widget.set_method(Methods.ONNXEmbedder)
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.wait_until_finished()
        self.widget.send_report()


if __name__ == "__main__":
    unittest.main()
