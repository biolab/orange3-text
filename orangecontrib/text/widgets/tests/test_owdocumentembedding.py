import unittest
from unittest.mock import Mock, patch

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError

from orangecontrib.text.tests.test_documentembedder import PATCH_METHOD, make_dummy_post
from orangecontrib.text.widgets.owdocumentembedding import OWDocumentEmbedding
from orangecontrib.text import Corpus


async def none_method(_, __):
    return None


class TestOWDocumentEmbedding(WidgetTest):

    def setUp(self):
        self.widget = self.create_widget(OWDocumentEmbedding)
        self.corpus = Corpus.from_file('deerwester')
        self.larger_corpus = Corpus.from_file('book-excerpts')

    def test_input(self):
        set_data = self.widget.set_data = Mock()
        self.send_signal("Corpus", None)
        set_data.assert_called_with(None)
        self.send_signal("Corpus", self.corpus[:0])
        set_data.assert_called_with(self.corpus[:0])
        self.send_signal("Corpus", self.corpus)
        set_data.assert_called_with(self.corpus)

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_output(self):
        self.send_signal("Corpus", None)
        self.assertIsNone(self.get_output(self.widget.Outputs.new_corpus))

        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        result = self.get_output(self.widget.Outputs.new_corpus)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(self.corpus), len(result))

    def test_input_summary(self):
        input_summary = self.widget.info.set_input_summary = Mock()
        self.send_signal("Corpus", None)
        input_summary.assert_called_with(self.widget.info.NoInput)

        self.send_signal("Corpus", self.corpus)
        input_summary.assert_called_with(str(len(self.corpus)), "9 documents.")

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_output_summary(self):
        output_summary = self.widget.info.set_output_summary = Mock()
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        output_summary.assert_called_with(
            f"{str(int(len(self.corpus)))}|0",
            "Successful: {}, Unsuccessful: {}".format(
                int(len(self.corpus)), int(0)))

    @patch(PATCH_METHOD, make_dummy_post(b''))
    def test_some_failed(self):
        simulate.combobox_activate_index(self.widget.controls.aggregator, 1)
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        result = self.get_output(self.widget.Outputs.new_corpus)
        skipped = self.get_output(self.widget.Outputs.skipped)
        self.assertIsNone(result)
        self.assertEqual(len(skipped), len(self.corpus))
        self.assertTrue(self.widget.Warning.unsuccessful_embeddings.is_shown())

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_cancel_embedding(self):
        self.send_signal("Corpus", self.larger_corpus)
        self.widget.cancel_button.click()
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.new_corpus))

    @patch('orangecontrib.text.vectorization.document_embedder' +
           '._ServerEmbedder.embedd_data',
           side_effect=EmbeddingConnectionError)
    def test_connection_error(self, _):
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.new_corpus))
        self.assertTrue(self.widget.Error.no_connection.is_shown())

    @patch('orangecontrib.text.vectorization.document_embedder' +
           '.DocumentEmbedder.__call__',
           side_effect=OSError)
    def test_unexpected_error(self, _):
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.new_corpus))
        self.assertTrue(self.widget.Error.unexpected_error.is_shown())

    @patch(PATCH_METHOD, make_dummy_post(b'{"embedding": [1.3, 1]}'))
    def test_rerun_on_new_data(self):
        """ Check if embedding is automatically re-run on new data """
        self.widget._auto_apply = False
        self.assertIsNone(self.get_output(self.widget.Outputs.new_corpus))

        self.send_signal(self.widget.Inputs.corpus, self.corpus[:3])
        self.wait_until_finished()
        self.assertEqual(
            3, len(self.get_output(self.widget.Outputs.new_corpus))
        )

        self.send_signal(self.widget.Inputs.corpus, self.corpus[:1])
        self.wait_until_finished()
        self.assertEqual(
            1, len(self.get_output(self.widget.Outputs.new_corpus))
        )

    @patch('orangecontrib.text.vectorization.document_embedder' +
           '._ServerEmbedder._encode_data_instance', none_method)
    def test_skipped_documents(self):
        self.send_signal("Corpus", self.corpus)
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.new_corpus))
        self.assertEqual(len(self.get_output(self.widget.Outputs.skipped)), len(self.corpus))
        self.assertTrue(self.widget.Warning.unsuccessful_embeddings.is_shown())


if __name__ == "__main__":
    unittest.main()
