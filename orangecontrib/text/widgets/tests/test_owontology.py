# pylint: disable=missing-docstring
import os
import pickle
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from AnyQt.QtCore import Qt, QItemSelectionModel, QItemSelection, \
    QItemSelectionRange
from AnyQt.QtWidgets import QFileDialog
from AnyQt.QtTest import QTest
from owlready2 import get_ontology, Thing, types

from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest
from orangecontrib.text.ontology import OntologyHandler
from orangecontrib.text.widgets.owontology import OWOntology, _run, \
    EditableTreeView, _tree_to_html
from orangecontrib.text.widgets.utils.words import create_words_table


SBERT_PATCH_METHOD = "orangecontrib.text.ontology.SBERT.__call__"


class TestUtils(unittest.TestCase):
    def test_tree_to_html(self):
        tree = {"foo": {"bar": {},
                        "bar1": {"bar2": {}, "bar3": {}}}, "baz": {}}
        html = "<ul><li>foo<ul><li>bar</li><li>bar1<ul><li>bar2</li>" \
               "<li>bar3</li></ul></li></ul></li><li>baz</li></ul>"
        self.assertEqual(html, _tree_to_html(tree))


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.handler = OntologyHandler()
        self.words = ["foo", "bar"]
        self.state = Mock()
        self.state.is_interruption_requested = Mock(return_value=False)

    def test_run(self):
        result = _run(self.handler.generate, (self.words,), self.state)
        self.assertEqual(result, ({"bar": {"foo": {}}}, 0))

    def test_run_single_word(self):
        result = _run(self.handler.generate, (["foo"],), self.state)
        self.assertEqual(result, ({"foo": {}}, 0))

    def test_run_empty(self):
        result = _run(self.handler.generate, ([],), self.state)
        self.assertEqual(result, ({}, 0))

    def test_run_interrupt(self):
        state = Mock()
        state.is_interruption_requested = Mock(return_value=True)
        self.assertRaises(Exception, _run, self.handler.generate,
                          (self.words,), state)


class TestEditableTreeView(WidgetTest):
    def setUp(self):
        self.data = {"foo": {"bar": {}, "baz": {}}}
        self.view = EditableTreeView()

    def test_set_data(self):
        model = self.view._EditableTreeView__model

        self.view.set_data(self.data)
        self.assertEqual(model.rowCount(), 1)

        self.view.set_data(self.data)
        self.assertEqual(model.rowCount(), 1)

    def test_clear(self):
        model = self.view._EditableTreeView__model

        self.view.set_data(self.data)
        self.assertEqual(model.rowCount(), 1)

        self.view.clear()
        self.assertEqual(model.rowCount(), 0)

    def test_on_remove(self):
        self.view.set_data(self.data)
        model = self.view._EditableTreeView__model
        sel_model = self.view._EditableTreeView__tree.selectionModel()
        sel_model.select(model.index(0, 0), QItemSelectionModel.ClearAndSelect)

        self.assertEqual(self.view.get_data(), {"foo": {"bar": {}, "baz": {}}})
        self.view._EditableTreeView__on_remove()
        self.assertEqual(self.view.get_data(), {"bar": {}, "baz": {}})

    def test_on_remove_recursive(self):
        self.view.set_data(self.data)
        model = self.view._EditableTreeView__model
        sel_model = self.view._EditableTreeView__tree.selectionModel()
        sel_model.select(model.index(0, 0), QItemSelectionModel.ClearAndSelect)

        self.assertEqual(self.view.get_data(), {"foo": {"bar": {}, "baz": {}}})
        self.view._EditableTreeView__on_remove_recursive()
        self.assertEqual(self.view.get_data(), {})

    def test_on_remove_with_delete_key(self):
        # test with delete button
        self.view.set_data(self.data)
        model = self.view._EditableTreeView__model
        sel_model = self.view._EditableTreeView__tree.selectionModel()
        sel_model.select(model.index(0, 0), QItemSelectionModel.ClearAndSelect)

        self.assertEqual(self.view.get_data(), {"foo": {"bar": {}, "baz": {}}})
        QTest.keyClick(self.view, Qt.Key_Delete)
        self.assertDictEqual({}, self.view.get_data())

    def test_on_remove_with_backspace_key(self):
        self.view.set_data(self.data)
        model = self.view._EditableTreeView__model
        sel_model = self.view._EditableTreeView__tree.selectionModel()
        sel_model.select(model.index(0, 0), QItemSelectionModel.ClearAndSelect)

        self.assertEqual(self.view.get_data(), {"foo": {"bar": {}, "baz": {}}})
        QTest.keyClick(self.view, Qt.Key_Backspace)
        self.assertDictEqual({}, self.view.get_data())

    def test_get_words(self):
        self.view.set_data(self.data)
        self.assertEqual(self.view.get_words(), ["foo", "bar", "baz"])

    def test_get_selected_words(self):
        self.view.set_data(self.data)
        model = self.view._EditableTreeView__model
        sel_model = self.view._EditableTreeView__tree.selectionModel()
        sel_model.select(model.index(0, 0), QItemSelectionModel.ClearAndSelect)
        self.assertEqual(self.view.get_selected_words(), {"foo"})

    def test_get_selected_words_with_children(self):
        self.view.set_data(self.data)
        model = self.view._EditableTreeView__model
        sel_model = self.view._EditableTreeView__tree.selectionModel()
        sel_model.select(model.index(0, 0), QItemSelectionModel.ClearAndSelect)
        self.assertEqual(self.view.get_selected_words_with_children(),
                         {"foo", "bar", "baz"})

    def test_get_data_with_selection(self):
        self.view.set_data(self.data)
        model = self.view._EditableTreeView__model
        sel_model = self.view._EditableTreeView__tree.selectionModel()
        sel_model.select(model.index(0, 0), QItemSelectionModel.ClearAndSelect)
        self.assertEqual(self.view.get_data(), {"foo": {"bar": {}, "baz": {}}})
        self.assertEqual(self.view.get_data(with_selection=True),
                         ({"foo": ({"bar": ({}, False),
                                    "baz": ({}, False)}, True)}, False))

    def test_set_data_with_selection(self):
        data = ({"foo": ({"bar": ({}, False),
                          "baz": ({}, False)}, True)}, False)
        self.view.set_data(data)
        self.assertEqual(self.view.get_data(with_selection=True), data)


class TestOWOntology(WidgetTest):
    @patch(SBERT_PATCH_METHOD, Mock(return_value=[np.ones(300)] * 3))
    def setUp(self):
        self._ontology_1 = {"foo1": {"bar1": {}, "baz1": {}}}
        self._ontology_2 = {"foo2": {"bar2": {}, "baz2": {}}}
        settings = {
            "ontology_library": [
                {"name": "Ontology 1", "ontology": self._ontology_1},
                {"name": "Ontology 2", "ontology": self._ontology_2}
            ],
            "ontology": ({"foo1": ({"bar1": ({}, False),
                                    "baz1": ({}, False)}, False)}, False)
        }
        self.widget = self.create_widget(OWOntology, stored_settings=settings)

    @patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)] * 3)
    def test_input_words(self, _):
        get_ontology_data = self.widget._OWOntology__ontology_view.get_data

        words = create_words_table(["foo"])
        self.send_signal(self.widget.Inputs.words, words)

        self.assertEqual(self.widget._OWOntology__get_selected_row(), 0)
        self.assertEqual(get_ontology_data(), self._ontology_1)

        self.widget._OWOntology__set_selected_row(1)
        self.assertEqual(self.widget._OWOntology__get_selected_row(), 1)
        self.assertEqual(get_ontology_data(), self._ontology_2)

        self.widget._OWOntology__set_selected_row(0)
        self.assertEqual(self.widget._OWOntology__get_selected_row(), 0)
        self.assertEqual(get_ontology_data(), self._ontology_1)

    def test_input_words_no_type(self):
        words = Table("zoo")
        self.send_signal(self.widget.Inputs.words, words)
        self.assertTrue(self.widget.Warning.no_words_column.is_shown())
        self.send_signal(self.widget.Inputs.words, None)
        self.assertFalse(self.widget.Warning.no_words_column.is_shown())

    def test_output_words(self):
        def select_words(indices):
            onto_view = self.widget._OWOntology__ontology_view
            model = onto_view._EditableTreeView__model
            tree = onto_view._EditableTreeView__tree
            selection = QItemSelection()
            sel_model = tree.selectionModel()
            for i in indices:
                selection.append(QItemSelectionRange(model.index(i, 0)))
            sel_model.select(selection, QItemSelectionModel.ClearAndSelect)

        self.assertIsNone(self.get_output(self.widget.Outputs.words))

        select_words(range(1))
        words = create_words_table(["bar1", "baz1", "foo1"])
        output = self.get_output(self.widget.Outputs.words)
        self.assert_table_equal(words, output)

    @patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)] * 3)
    def test_library_sel_changed(self, _):
        get_ontology_data = self.widget._OWOntology__ontology_view.get_data
        self.assertEqual(get_ontology_data(), self._ontology_1)
        self.widget._OWOntology__set_selected_row(1)
        self.assertEqual(self.widget._OWOntology__get_selected_row(), 1)
        self.assertEqual(get_ontology_data(), self._ontology_2)

    @patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)] * 3)
    def test_library_add(self, _):
        get_ontology_data = self.widget._OWOntology__ontology_view.get_data

        self.widget._OWOntology__on_add()
        self.assertEqual(self.widget._OWOntology__get_selected_row(), 2)
        self.assertEqual(get_ontology_data(), self._ontology_1)

        self.widget._OWOntology__set_selected_row(1)
        self.assertEqual(get_ontology_data(), self._ontology_2)

    @patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)] * 3)
    def test_library_remove(self, _):
        get_ontology_data = self.widget._OWOntology__ontology_view.get_data

        self.widget._OWOntology__on_remove()
        self.assertEqual(self.widget._OWOntology__model.rowCount(), 1)
        self.assertEqual(get_ontology_data(), self._ontology_2)
        self.assertEqual(self.widget._OWOntology__get_selected_row(), 0)

        self.widget._OWOntology__on_remove()
        self.assertEqual(self.widget._OWOntology__model.rowCount(), 0)
        self.assertEqual(get_ontology_data(), self._ontology_2)
        self.assertIsNone(self.widget._OWOntology__get_selected_row())

    @patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)] * 3)
    def test_library_update(self, _):
        self.assertEqual(self.widget._OWOntology__get_selected_row(), 0)
        model = self.widget._OWOntology__ontology_view._EditableTreeView__model
        model.setData(model.index(0, 0), "foo3", role=Qt.EditRole)

        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual(settings["ontology_library"][0]["ontology"],
                         self._ontology_1)

        self.widget._OWOntology__on_update()
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual(settings["ontology_library"][0]["ontology"],
                         {"foo3": {"bar1": {}, "baz1": {}}})

    @patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)] * 3)
    def test_library_import(self, _):
        ontology = {"foo3": {"bar3": {}, "baz3": {}}}
        get_ontology_data = self.widget._OWOntology__ontology_view.get_data

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as f:
            pickle.dump(ontology, f)

        with patch.object(QFileDialog, "getOpenFileName",
                          Mock(return_value=(f.name, None))):
            self.widget._OWOntology__on_import_file()
            self.assertEqual(self.widget._OWOntology__get_selected_row(), 2)
            self.assertEqual(self.widget._OWOntology__model[2].name,
                             os.path.basename(f.name))
            self.assertEqual(get_ontology_data(), ontology)

    def test_library_import_owl(self):
        # create ontology
        onto = get_ontology("http://test.org/onto.owl")
        with onto:
            # foo3 will be missing the label attribute - widget display name
            foo3 = types.new_class("foo3", (Thing,))
            # other two classes will have label attribute - widget display label
            for c, label in (("bar3", "Bar 3"), ("baz3", "Baz 3")):
                thing = types.new_class(c, (foo3,))
                thing.label = label

        get_ontology_data = self.widget._OWOntology__ontology_view.get_data
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # safe to file
            path = os.path.join(tmp_dir_name, "onto.owl")
            onto.save(file=path, format="rdfxml")

            # open with widget
            with patch.object(
                QFileDialog, "getOpenFileName", Mock(return_value=(path, None))
            ):
                self.widget._OWOntology__on_import_file()
                self.assertEqual(self.widget._OWOntology__get_selected_row(), 2)
                name = self.widget._OWOntology__model[2].name
                self.assertEqual(name, os.path.basename(path))
                expected = {"foo3": {"Bar 3": {}, "Baz 3": {}}}
                self.assertEqual(get_ontology_data(), expected)

    def test_library_save(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pkl",
                                         delete=False) as f:
            pass

        with patch.object(QFileDialog, "getSaveFileName",
                          Mock(return_value=(f.name, None))):
            self.widget._OWOntology__on_save()

        with open(f.name, "rb") as dummy_f:
            self.assertEqual(pickle.load(dummy_f), self._ontology_1)

    def test_send_report(self):
        self.widget.send_report()

    def test_skipped_words_generate(self):
        """
        Test case when embedding fails when generating the ontology. It results
        in exclusion of non-embedded terms and warning.
        """
        get_ontology_data = self.widget._OWOntology__ontology_view.get_data
        self.assertDictEqual(get_ontology_data(), {"foo1": {"bar1": {}, "baz1": {}}})

        # generate with embedding error - two skipped
        with patch(SBERT_PATCH_METHOD, return_value=[np.ones(300), None, None]):
            self.widget._OWOntology__run_button.click()
            self.wait_until_finished()
            self.assertDictEqual(get_ontology_data(), {"foo1": {}})
            self.assertTrue(self.widget.Warning.skipped_words.is_shown())
            self.assertEqual(
                str(self.widget.Warning.skipped_words),
                "2 terms are skipped due to server connection error.",
            )

        # generate without embedding error
        with patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)]):
            self.widget._OWOntology__run_button.click()
            self.wait_until_finished()
            self.assertDictEqual(get_ontology_data(), {"foo1": {}})
            self.assertFalse(self.widget.Warning.skipped_words.is_shown())

    def test_skipped_words_insert(self):
        """
        Test case when embedding fails when inserting the term. It results
        in exclusion of non-embedded terms and warning.
        """
        words = create_words_table(["foo2", "foo3"])
        self.send_signal(self.widget.Inputs.words, words)

        # insert with an embedding error
        with patch(
            SBERT_PATCH_METHOD,
            side_effect=[
                [np.ones(300), np.ones(300), np.ones(300), None],
                [np.ones(300), np.ones(300), np.ones(300)],
            ],
        ):
            get_ontology_data = self.widget._OWOntology__ontology_view.get_data
            self.assertDictEqual(
                get_ontology_data(), {"foo1": {"bar1": {}, "baz1": {}}}
            )

            self.widget._OWOntology__input_view.setCurrentIndex(
                self.widget._OWOntology__input_model.index(0, 0)
            )
            self.widget._OWOntology__inc_button.click()
            self.wait_until_finished()
            self.assertDictEqual(
                get_ontology_data(), {"foo1": {"bar1": {}, "baz1": {}}}
            )
            self.assertTrue(self.widget.Warning.skipped_words.is_shown())
            self.assertEqual(
                str(self.widget.Warning.skipped_words),
                "1 terms are skipped due to server connection error.",
            )

        # insert without embedding error
        with patch(SBERT_PATCH_METHOD, return_value=[np.ones(300)] * 4):
            self.widget._OWOntology__inc_button.click()
            self.wait_until_finished()
            self.assertDictEqual(
                get_ontology_data(), {"foo1": {"bar1": {}, "baz1": {}, "foo2": {}}}
            )
            self.assertFalse(self.widget.Warning.skipped_words.is_shown())


if __name__ == "__main__":
    unittest.main()
