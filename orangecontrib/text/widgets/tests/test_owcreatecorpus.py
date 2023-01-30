import unittest

import numpy as np
from Orange.data import StringVariable
from Orange.widgets.tests.base import WidgetTest
from AnyQt.QtWidgets import QPushButton, QComboBox
from orangewidget.tests.utils import simulate

from orangecontrib.text.widgets.owcreatecorpus import OWCreateCorpus


class TestOWCreateCorpus(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWCreateCorpus)
        self.add_document_btn = self.widget.buttonsArea.findChild(QPushButton)

    def test_add_remove_editors(self):
        self.assertEqual(3, len(self.widget.editors))
        self.assertEqual(3, len(self.widget.texts))
        self.assertListEqual([("", "")] * 3, self.widget.texts)

        self.add_document_btn.click()
        self.assertEqual(4, len(self.widget.editors))
        self.assertEqual(4, len(self.widget.texts))
        self.assertListEqual([("", "")] * 4, self.widget.texts)

        self.add_document_btn.click()
        self.assertEqual(5, len(self.widget.editors))
        self.assertEqual(5, len(self.widget.texts))
        self.assertListEqual([("", "")] * 5, self.widget.texts)

        # click x button for first editor
        self.widget.editors[0].findChild(QPushButton).click()
        self.assertEqual(4, len(self.widget.editors))
        self.assertEqual(4, len(self.widget.texts))
        self.assertListEqual([("", "")] * 4, self.widget.texts)

        self.widget.editors[0].findChild(QPushButton).click()
        self.assertEqual(3, len(self.widget.editors))
        self.assertEqual(3, len(self.widget.texts))
        self.assertListEqual([("", "")] * 3, self.widget.texts)

        self.widget.editors[0].findChild(QPushButton).click()
        self.assertEqual(2, len(self.widget.editors))
        self.assertEqual(2, len(self.widget.texts))
        self.assertListEqual([("", "")] * 2, self.widget.texts)

        self.widget.editors[0].findChild(QPushButton).click()
        self.assertEqual(1, len(self.widget.editors))
        self.assertEqual(1, len(self.widget.texts))
        self.assertListEqual([("", "")], self.widget.texts)

        # last editor cannot be removed
        self.widget.editors[0].findChild(QPushButton).click()
        self.assertEqual(1, len(self.widget.editors))
        self.assertEqual(1, len(self.widget.texts))
        self.assertListEqual([("", "")], self.widget.texts)

    def test_add_text(self):
        # start with 1 editor
        self.widget.editors[-1].findChild(QPushButton).click()
        self.widget.editors[-1].findChild(QPushButton).click()

        editor = self.widget.editors[0]
        self.assertListEqual([("", "")], self.widget.texts)
        editor.title_le.setText("Beautiful document")
        editor.title_le.editingFinished.emit()
        self.assertListEqual([("Beautiful document", "")], self.widget.texts)
        editor.text_area.setPlainText("I am a beautiful document")
        editor.text_area.editingFinished.emit()
        self.assertListEqual(
            [("Beautiful document", "I am a beautiful document")], self.widget.texts
        )

        self.add_document_btn.click()
        editor = self.widget.editors[1]
        self.assertListEqual(
            [("Beautiful document", "I am a beautiful document"), ("", "")],
            self.widget.texts,
        )
        editor.title_le.setText("Another another document")
        editor.title_le.editingFinished.emit()
        self.assertListEqual(
            [
                ("Beautiful document", "I am a beautiful document"),
                ("Another another document", ""),
            ],
            self.widget.texts,
        )
        editor.text_area.setPlainText("I am another beautiful document")
        editor.text_area.editingFinished.emit()
        self.assertListEqual(
            [
                ("Beautiful document", "I am a beautiful document"),
                ("Another another document", "I am another beautiful document"),
            ],
            self.widget.texts,
        )

        # remove first document
        self.widget.editor_vbox.findChild(QPushButton).click()
        self.assertListEqual(
            [("Another another document", "I am another beautiful document")],
            self.widget.texts,
        )

        # change the only document
        editor = self.widget.editors[0]
        editor.title_le.setText("Modified document")
        editor.title_le.editingFinished.emit()
        self.assertListEqual(
            [("Modified document", "I am another beautiful document")],
            self.widget.texts,
        )
        editor.text_area.setPlainText("Test")
        editor.text_area.editingFinished.emit()
        self.assertListEqual([("Modified document", "Test")], self.widget.texts)

        self.add_document_btn.click()
        self.assertListEqual(
            [("Modified document", "Test"), ("", "")], self.widget.texts
        )

    def test_output(self):
        # start with 1 editor
        self.widget.editors[-1].findChild(QPushButton).click()
        self.widget.editors[-1].findChild(QPushButton).click()

        corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(0, len(corpus.domain.attributes))
        self.assertTupleEqual(
            (StringVariable("Title"), StringVariable("Document")), corpus.domain.metas
        )
        np.testing.assert_array_equal(["?"], corpus.titles)
        self.assertListEqual(["?"], corpus.documents)
        np.testing.assert_array_equal([["", ""]], corpus.metas)

        self.add_document_btn.click()
        self.add_document_btn.click()
        editor1, editor2, editor3 = self.widget.editors
        editor1.title_le.setText("Document 1")
        editor2.title_le.setText("Document 2")
        editor3.title_le.setText("Document 3")
        editor1.text_area.setPlainText("Test 1")
        editor2.text_area.setPlainText("Test 2")
        editor3.text_area.setPlainText("Test 3")
        editor1.text_area.editingFinished.emit()
        editor2.text_area.editingFinished.emit()
        editor3.text_area.editingFinished.emit()

        corpus = self.get_output(self.widget.Outputs.corpus)
        np.testing.assert_array_equal(
            ["Document 1", "Document 2", "Document 3"], corpus.titles
        )
        self.assertListEqual(["Test 1", "Test 2", "Test 3"], corpus.documents)
        np.testing.assert_array_equal(
            [
                ["Document 1", "Test 1"],
                ["Document 2", "Test 2"],
                ["Document 3", "Test 3"],
            ],
            corpus.metas,
        )

        editor2.findChild(QPushButton).click()
        corpus = self.get_output(self.widget.Outputs.corpus)
        np.testing.assert_array_equal(["Document 1", "Document 3"], corpus.titles)
        self.assertListEqual(["Test 1", "Test 3"], corpus.documents)
        np.testing.assert_array_equal(
            [
                ["Document 1", "Test 1"],
                ["Document 3", "Test 3"],
            ],
            corpus.metas,
        )

        self.add_document_btn.click()
        corpus = self.get_output(self.widget.Outputs.corpus)
        np.testing.assert_array_equal(["Document 1", "Document 3", "?"], corpus.titles)
        self.assertListEqual(["Test 1", "Test 3", "?"], corpus.documents)
        np.testing.assert_array_equal(
            [["Document 1", "Test 1"], ["Document 3", "Test 3"], ["", ""]],
            corpus.metas,
        )

        self.widget.editors[0].findChild(QPushButton).click()
        corpus = self.get_output(self.widget.Outputs.corpus)
        np.testing.assert_array_equal(["Document 3", "?"], corpus.titles)
        self.assertListEqual(["Test 3", "?"], corpus.documents)
        np.testing.assert_array_equal(
            [["Document 3", "Test 3"], ["", ""]],
            corpus.metas,
        )

        self.widget.editors[-1].findChild(QPushButton).click()
        corpus = self.get_output(self.widget.Outputs.corpus)
        np.testing.assert_array_equal(["Document 3"], corpus.titles)
        self.assertListEqual(["Test 3"], corpus.documents)
        np.testing.assert_array_equal([["Document 3", "Test 3"]], corpus.metas)

    def test_language(self):
        corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual("en", corpus.language)

        combo = self.widget.controlArea.findChild(QComboBox)
        simulate.combobox_activate_index(combo, 2)
        corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual("am", corpus.language)


if __name__ == "__main__":
    unittest.main()
