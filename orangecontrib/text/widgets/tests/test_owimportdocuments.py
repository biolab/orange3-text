import os
import unittest
from unittest.mock import patch, Mock

from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.widgets.owimportdocuments import OWImportDocuments


class TestOWImportDocuments(WidgetTest):
    def setUp(self) -> None:
        self.widget: OWImportDocuments = self.create_widget(OWImportDocuments)
        path = os.path.join(os.path.dirname(__file__), "data/documents")
        self.widget.setCurrentPath(path)
        self.widget.reload()
        self.wait_until_finished()

    def test_current_path(self):
        path = os.path.join(os.path.dirname(__file__), "data/documents")
        self.assertEqual(path, self.widget.currentPath)

    def test_no_skipped(self):
        path = os.path.join(os.path.dirname(__file__), "data/documents", "good")
        self.widget.setCurrentPath(path)
        self.widget.reload()
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.skipped_documents))

    def test_output(self):
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(4, len(output))
        self.assertEqual(3, len(output.domain.metas))
        names = output.get_column_view("name")[0]
        self.assertListEqual(
            # ž in sample_text_ž must be unicode char 0x17E not decomposed
            # 0x7A + 0x30C as it is in file name
            ["sample_docx", "sample_odt", "sample_pdf", "sample_txt_ž"],
            sorted(names.tolist()),
        )
        texts = output.get_column_view("content")[0]
        self.assertListEqual(
            # ž in sample_text_ž must be unicode char 0x17E not decomposed
            # 0x7A + 0x30C as it is in file name
            [
                f"This is a test {x} file"
                for x in ["docx", "odt", "pdf", "txt_ž"]
            ],
            sorted([x.strip() for x in texts.tolist()]),
        )
        self.assertEqual("content", output.text_features[0].name)

        skipped_output = self.get_output(self.widget.Outputs.skipped_documents)
        self.assertEqual(1, len(skipped_output))
        self.assertEqual(2, len(skipped_output.domain.metas))
        names = skipped_output.get_column_view("name")[0]
        self.assertListEqual(
            ["sample_pdf_corrupted.pdf"],
            sorted(names.tolist()),
        )

    def test_could_not_be_read_warning(self):
        """
        sample_pdf_corrupted.pdf is corrupted file and cannot be loaded
        correctly - widget must show the warning
        """
        self.assertTrue(self.widget.Warning.read_error.is_shown())
        self.assertEqual(
            "One file couldn't be read.",
            str(self.widget.Warning.read_error),
        )

    def test_send_report(self):
        self.widget.send_report()

    def test_conllu_cb(self):
        path = os.path.join(os.path.dirname(__file__), "data/conllu")
        self.widget.setCurrentPath(path)
        self.widget.reload()
        self.wait_until_finished()
        # default has only lemmas
        corpus = self.get_output(self.widget.Outputs.data)
        self.assertTrue(corpus.has_tokens())
        # check pos tags are on the output
        self.widget.controls.pos_cb.setChecked(True)
        corpus = self.get_output(self.widget.Outputs.data)
        self.assertTrue(len(corpus.pos_tags))
        # check named entities are on the output
        self.widget.controls.ner_cb.setChecked(True)
        corpus = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(corpus.domain.metas), 5)
        # check only corpus is on the output when all boxes unchecked
        self.widget.controls.lemma_cb.setChecked(False)
        self.widget.controls.pos_cb.setChecked(False)
        self.widget.controls.ner_cb.setChecked(False)
        corpus = self.get_output(self.widget.Outputs.data)
        self.assertFalse(corpus.has_tokens())
        self.assertIsNone(corpus.pos_tags)
        self.assertEqual(len(corpus.domain.metas), 4)

    def test_info_box(self):
        self.assertEqual(
            "4 documents, 1 skipped", self.widget.info_area.text()
        )

        # empty widget
        self.widget: OWImportDocuments = self.create_widget(OWImportDocuments)
        self.assertEqual(
            "No document set selected", self.widget.info_area.text()
        )

    @patch("orangecontrib.text.import_documents.ImportDocuments.scan",
           Mock(return_value=[]))
    def test_load_empty_folder(self):
        widget = self.create_widget(OWImportDocuments)
        path = os.path.join(os.path.dirname(__file__), "data/documents")
        widget.setCurrentPath(path)
        widget.reload()
        self.wait_until_finished(widget=widget)
        self.assertIsNone(self.get_output(widget.Outputs.data))


if __name__ == "__main__":
    unittest.main()
