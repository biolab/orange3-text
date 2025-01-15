import os
import unittest
from unittest.mock import patch, Mock

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.text.widgets.owimportdocuments import OWImportDocuments


DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tests", "data", "documents")
)


class TestOWImportDocuments(WidgetTest):
    def setUp(self) -> None:
        self.widget: OWImportDocuments = self.create_widget(OWImportDocuments)
        self.path = os.path.join(os.path.dirname(__file__), DATA_PATH)
        self.widget.setCurrentPath(self.path)
        self.widget.reload()
        self.wait_until_finished()

    def test_current_path(self):
        self.assertEqual(self.path, self.widget.currentPath)

    def test_no_skipped(self):
        path = os.path.join(DATA_PATH, "good")
        self.widget.setCurrentPath(path)
        self.widget.reload()
        self.wait_until_finished()
        self.assertIsNone(self.get_output(self.widget.Outputs.skipped_documents))

    def test_output(self):
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(5, len(output))
        self.assertEqual(3, len(output.domain.metas))
        names = output.get_column("name")
        self.assertListEqual(
            # ž in sample_text_ž must be unicode char 0x17E not decomposed
            # 0x7A + 0x30C as it is in file name
            [
                "minimal-document",
                "sample_docx",
                "sample_odt",
                "sample_pdf",
                "sample_txt_ž",
            ],
            sorted(names.tolist()),
        )
        # skip first document - it contains different text
        texts = output.get_column("content")[1:]
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
        names = skipped_output.get_column("name")
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
        self.assertEqual(len(corpus.domain.metas), 23)
        # check only corpus is on the output when all boxes unchecked
        self.widget.controls.lemma_cb.setChecked(False)
        self.widget.controls.pos_cb.setChecked(False)
        self.widget.controls.ner_cb.setChecked(False)
        corpus = self.get_output(self.widget.Outputs.data)
        self.assertFalse(corpus.has_tokens())
        self.assertIsNone(corpus.pos_tags)
        self.assertEqual(len(corpus.domain.metas), 22)

    def test_info_box(self):
        self.assertEqual("5 documents, 1 skipped", self.widget.info_area.text())

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

    def tests_context(self):
        self.widget: OWImportDocuments = self.create_widget(OWImportDocuments)
        # change default to something else to see if language is changed
        self.widget.language = "sl"

        path = os.path.join(DATA_PATH, "good")
        self.widget.setCurrentPath(path)
        self.widget.reload()
        self.wait_until_finished()

        # english is recognized for selected documents
        self.assertEqual(self.widget.language, "en")
        self.assertEqual("en", self.get_output(self.widget.Outputs.data).language)
        simulate.combobox_activate_item(self.widget.controls.language, "Dutch")

        self.assertEqual(self.widget.language, "nl")
        self.assertEqual("nl", self.get_output(self.widget.Outputs.data).language)

        # read something else
        path1 = os.path.join(os.path.dirname(__file__), "data/conllu")
        self.widget.setCurrentPath(path1)
        self.widget.reload()
        self.wait_until_finished()

        # read same data again and observe if context is restored
        self.widget.setCurrentPath(path)
        self.widget.reload()
        self.wait_until_finished()
        self.assertEqual(self.widget.language, "nl")
        self.assertEqual("nl", self.get_output(self.widget.Outputs.data).language)

    def test_migrate_settings(self):
        packed_data = self.widget.settingsHandler.pack_data(self.widget)
        packed_data["context_settings"][0].values["language"] = "French"
        packed_data["context_settings"][0].values["__version__"] = 1

        widget = self.create_widget(OWImportDocuments, stored_settings=packed_data)
        widget.setCurrentPath(self.path)
        widget.reload()
        self.wait_until_finished(widget=widget)
        self.assertEqual("fr", widget.language)

        packed_data["context_settings"][0].values["language"] = "Ancient greek"
        widget = self.create_widget(OWImportDocuments, stored_settings=packed_data)
        widget.setCurrentPath(self.path)
        widget.reload()
        self.wait_until_finished(widget=widget)
        self.assertEqual("grc", widget.language)

        packed_data["context_settings"][0].values["language"] = None
        widget = self.create_widget(OWImportDocuments, stored_settings=packed_data)
        widget.setCurrentPath(self.path)
        widget.reload()
        self.wait_until_finished(widget=widget)
        self.assertIsNone(widget.language)


if __name__ == "__main__":
    unittest.main()
