# pylint: disable=missing-docstring,protected-access
import unittest
from unittest.mock import Mock, patch
import tempfile
import os

from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QGroupBox, QFileDialog

from Orange.data import Table, StringVariable, Domain
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.widgets.owwordlist import OWWordList, UpdateRules, \
    WordList
from orangewidget.tests.utils import simulate


class TestWordList(unittest.TestCase):
    def test_generate_name(self):
        name = WordList.generate_word_list_name([])
        self.assertEqual(name, "Untitled 1")

        name = WordList.generate_word_list_name(["foo"])
        self.assertEqual(name, "Untitled 1")

        name = WordList.generate_word_list_name(["foo", "Untitled 1"])
        self.assertEqual(name, "Untitled 2")

        name = WordList.generate_word_list_name(
            ["foo", "Untitled 1", "Untitled A"])
        self.assertEqual(name, "Untitled 2")

        name = WordList.generate_word_list_name(["Untitled 2"])
        self.assertEqual(name, "Untitled 1")

        name = WordList.generate_word_list_name(
            ["Untitled", "Untitled 2", "Untitled 3"])
        self.assertEqual(name, "Untitled 1")

        name = WordList.generate_word_list_name(
            ["Untitled 1", "Untitled 2", "Untitled 4"])
        self.assertEqual(name, "Untitled 3")


class TestUpdateRule(unittest.TestCase):
    def test_update_intersect(self):
        rule = UpdateRules.INTERSECT
        model = Mock()
        model.wrap = Mock()
        UpdateRules.update(model, ["foo"], ["bar"], rule)
        model.wrap.assert_called_once_with([])

        model.wrap.reset_mock()
        UpdateRules.update(model, ["foo"], ["foo"], rule)
        model.wrap.assert_called_once_with(["foo"])

        model.wrap.reset_mock()
        UpdateRules.update(model, ["foo", "foo"], ["foo"], rule)
        model.wrap.assert_called_once_with(["foo"])

        model.wrap.reset_mock()
        UpdateRules.update(model, ["foo"], ["foo", "foo"], rule)
        model.wrap.assert_called_once_with(["foo"])

    def test_update_union(self):
        rule = UpdateRules.UNION
        model = Mock()
        model.wrap = Mock()
        UpdateRules.update(model, ["foo"], ["bar"], rule)
        model.wrap.assert_called_once_with(["foo", "bar"])

        model.wrap.reset_mock()
        UpdateRules.update(model, ["foo"], ["bar", "foo"], rule)
        model.wrap.assert_called_once_with(["foo", "bar"])

        model.wrap.reset_mock()
        UpdateRules.update(model, ["foo", "foo"], ["foo"], rule)
        model.wrap.assert_called_once_with(["foo"])

        model.wrap.reset_mock()
        UpdateRules.update(model, ["foo"], ["foo", "foo"], rule)
        model.wrap.assert_called_once_with(["foo"])

    def test_update_input(self):
        rule = UpdateRules.INPUT
        model = Mock()
        model.wrap = Mock()
        UpdateRules.update(model, ["foo"], ["bar"], rule)
        model.wrap.assert_called_once_with(["bar"])

    def test_update_input(self):
        rule = UpdateRules.LIBRARY
        model = Mock()
        model.wrap = Mock()
        UpdateRules.update(model, ["foo"], ["bar"], rule)
        model.wrap.assert_called_once_with(["foo"])


class TestOWWordList(WidgetTest):
    def setUp(self):
        self._word_list_1 = ["foo", "bar", "baz"]
        self._word_list_2 = ["word 1", "word 2", "word 3", "word 4"]
        settings = {"word_list_library": [{"name": "Hello world",
                                           "words": self._word_list_1},
                                          {"name": "Word list 1",
                                           "words": self._word_list_2}],
                    "words": self._word_list_1}
        self.widget = self.create_widget(OWWordList, stored_settings=settings)

        words_var = [StringVariable("S1"), StringVariable("S2")]
        lst = [["foo", "A"], ["bar", "B"], ["foobar", "C"]]
        self._input_table = Table.from_list(Domain([], metas=words_var), lst)

    def test_default(self):
        self.assertEqual(self.widget.library_model[0].name, "Hello world")
        self.assertEqual(self.widget.library_model[1].name, "Word list 1")
        self.assertEqual(self.widget._get_selected_word_list_index(), 0)
        self.assertListEqual(self.widget._get_selected_words_indices(), [])
        self.assertListEqual(self.widget.words_model[:], self._word_list_1)

    def test_outputs(self):
        self.widget._set_selected_words([0, 1])

        words = self.get_output(self.widget.Outputs.words)
        self.assertEqual(words.name, "Words")
        self.assertEqual(len(words.domain.attributes), 0)
        self.assertEqual(len(words.domain.class_vars), 1)
        self.assertEqual(words.domain.class_vars[0].name, "Selected")
        self.assertEqual(len(words.domain.metas), 1)
        self.assertEqual(words.domain.metas[0].name, "Words")
        self.assertDictEqual(words.domain.metas[0].attributes,
                             {"type": "words"})
        self.assertListEqual([1, 1, 0], list(words.Y))
        self.assertListEqual(list(words.metas[:, 0]), self._word_list_1)

        selected_words = self.get_output(self.widget.Outputs.selected_words)
        self.assertEqual(selected_words.name, "Words")
        self.assertEqual(len(selected_words.domain.attributes), 0)
        self.assertEqual(len(selected_words.domain.class_vars), 0)
        self.assertEqual(len(selected_words.domain.metas), 1)
        self.assertEqual(selected_words.domain.metas[0].name, "Words")
        self.assertDictEqual(selected_words.domain.metas[0].attributes,
                             {"type": "words"})
        self.assertListEqual(["foo", "bar"], list(selected_words.metas[:, 0]))

    def test_library_sel_changed(self):
        self.widget._set_selected_word_list(1)
        self.assertEqual(self.widget._get_selected_word_list_index(), 1)
        words = self.get_output(self.widget.Outputs.words)
        self.assertListEqual(list(words.metas[:, 0]),
                             ["word 1", "word 2", "word 3", "word 4"])

    def test_library_add(self):
        self.widget._OWWordList__on_add_word_list()
        sel_wlist = self.widget._get_selected_word_list_index()
        sel_words = self.widget._get_selected_words_indices()
        self.assertEqual(sel_wlist, 2)
        self.assertListEqual(sel_words, [])
        self.assertListEqual(self.widget.words_model[:], self._word_list_1)

        self.widget._set_selected_word_list(1)
        self.assertListEqual(self.widget.words_model[:], self._word_list_2)

    def test_library_remove(self):
        self.widget._OWWordList__on_remove_word_list()
        self.assertEqual(self.widget.library_model.rowCount(), 1)
        self.assertEqual(self.widget.words_model.rowCount(), 4)
        sel_wlist = self.widget._get_selected_word_list_index()
        sel_words = self.widget._get_selected_words_indices()
        self.assertEqual(sel_wlist, 0)
        self.assertListEqual(sel_words, [])

        self.widget._OWWordList__on_remove_word_list()
        self.assertEqual(self.widget.library_model.rowCount(), 0)
        self.assertEqual(self.widget.words_model.rowCount(), 4)
        sel_wlist = self.widget._get_selected_word_list_index()
        sel_words = self.widget._get_selected_words_indices()
        self.assertIsNone(sel_wlist, 0)
        self.assertListEqual(sel_words, [])

    def test_library_update(self):
        self.assertEqual(self.widget._get_selected_word_list_index(), 0)
        model = self.widget.words_model
        model.setItemData(model.index(2, 0), {Qt.EditRole: "foobar"})
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertListEqual(settings["word_list_library"][0]["words"],
                             ["foo", "bar", "baz"])
        self.widget._OWWordList__on_update_word_list()
        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertListEqual(settings["word_list_library"][0]["words"],
                             ["foo", "bar", "foobar"])

    def test_library_import(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"traffic \n")
            f.write(b"control car bike bus\n")
        with patch.object(QFileDialog, "getOpenFileName",
                          Mock(return_value=(f.name, None))):
            self.widget._OWWordList__on_import_word_list()
            self.assertEqual(self.widget._get_selected_word_list_index(), 2)
            self.assertEqual(self.widget.library_model[2].name,
                             os.path.basename(f.name))
            self.assertListEqual(self.widget.words_model[:],
                                 ["traffic", "control car bike bus"])

    def test_library_save(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                         delete=False) as f:
            pass
        with patch.object(QFileDialog, "getSaveFileName",
                          Mock(return_value=(f.name, None))):
            self.widget._OWWordList__on_save_word_list()
        with open(f.name) as dummy_f:
            self.assertListEqual(dummy_f.read().strip().split("\n"),
                                 self._word_list_1)

    def test_words_sel_changed(self):
        self.widget._set_selected_words([0, 2])
        output = self.get_output(self.widget.Outputs.selected_words)
        self.assertListEqual(list(output.metas[:, 0]), ["foo", "baz"])

    def test_words_sel_changed_commit_invoked_once(self):
        self.widget.commit = Mock()
        self.widget._set_selected_words([0, 2])
        self.widget.commit.assert_called_once()

    def test_add_word(self):
        self.widget.commit = Mock()
        self.widget._OWWordList__on_add_word()
        self.widget.commit.assert_called_once()

    def test_remove_word(self):
        self.widget._set_selected_words([0])
        self.widget._OWWordList__on_remove_word()
        output = self.get_output(self.widget.Outputs.words)
        self.assertListEqual(list(output.metas[:, 0]), ["bar", "baz"])

        self.widget._OWWordList__on_remove_word()
        self.widget._OWWordList__on_remove_word()
        self.assertIsNone(self.get_output(self.widget.Outputs.words))

    def test_remove_word_commit_invoked_once(self):
        self.widget._set_selected_words([0, 1])
        self.widget.commit = Mock()
        self.widget._OWWordList__on_remove_word()
        self.widget.commit.assert_called_once()

    def test_remove_word_no_selection(self):
        self.widget.commit = Mock()
        self.widget._OWWordList__on_remove_word()
        self.widget.commit.assert_not_called()

    def test_sort_words(self):
        self.widget._set_selected_words([0, 1])
        self.widget._OWWordList__on_apply_sorting()
        selected_words = self.get_output(self.widget.Outputs.selected_words)
        self.assertListEqual(["bar", "foo"], list(selected_words.metas[:, 0]))

    def test_sort_words_commit_invoked_once(self):
        self.widget.commit = Mock()
        self.widget._OWWordList__on_apply_sorting()
        self.widget.commit.assert_called_once()

    def test_input_controls_enabled(self):
        box: QGroupBox = self.widget._OWWordList__input_box
        self.assertFalse(box.isEnabled())

        iris = Table("iris")
        self.send_signal(self.widget.Inputs.words, iris)
        self.assertFalse(box.isEnabled())

        zoo = Table("zoo")
        self.send_signal(self.widget.Inputs.words, zoo)
        self.assertTrue(box.isEnabled())

        self.send_signal(self.widget.Inputs.words, None)
        self.assertFalse(box.isEnabled())

    def test_input_data_warning(self):
        self.assertFalse(self.widget.Warning.no_string_vars.is_shown())

        iris = Table("iris")
        self.send_signal(self.widget.Inputs.words, iris)
        self.assertTrue(self.widget.Warning.no_string_vars.is_shown())

        zoo = Table("zoo")
        self.send_signal(self.widget.Inputs.words, zoo)
        self.assertFalse(self.widget.Warning.no_string_vars.is_shown())

        self.send_signal(self.widget.Inputs.words, None)
        self.assertFalse(self.widget.Warning.no_string_vars.is_shown())

    def test_input_data(self):
        self.send_signal(self.widget.Inputs.words, self._input_table)
        self.assertEqual(self.widget.words_var.name, "S1")
        self.assertListEqual(self.widget.words_model[:], ["foo", "bar"])

    def test_input_var_changed(self):
        self.send_signal(self.widget.Inputs.words, self._input_table)
        simulate.combobox_activate_index(self.widget.controls.words_var, 1)
        self.assertEqual(self.widget.words_var.name, "S2")
        self.assertListEqual(self.widget.words_model[:], [])

    def test_update_rule_changed(self):
        buttons = self.widget.controls.update_rule_index.buttons
        self.send_signal(self.widget.Inputs.words, self._input_table)

        buttons[UpdateRules.UNION].click()
        self.assertListEqual(self.widget.words_model[:],
                             ["foo", "bar", "baz", "foobar"])

        buttons[UpdateRules.INPUT].click()
        self.assertListEqual(self.widget.words_model[:],
                             ["foo", "bar", "foobar"])

        buttons[UpdateRules.LIBRARY].click()
        self.assertListEqual(self.widget.words_model[:],
                             ["foo", "bar", "baz"])

    @patch("orangecontrib.text.widgets.owwordlist.OWWordList.commit")
    def test_commit_invoked_once(self, commit: Mock):
        self.create_widget(OWWordList, stored_settings=[])
        commit.assert_called_once()

        settings = {"word_list_library": [{"name": "Hello world",
                                           "words": self._word_list_1},
                                          {"name": "Word list 1",
                                           "words": self._word_list_2}]}
        commit.reset_mock()
        self.create_widget(OWWordList, stored_settings=settings)
        commit.assert_called_once()

        settings = {"word_list_library": [{"name": "Hello world",
                                           "words": self._word_list_1},
                                          {"name": "Word list 1",
                                           "words": self._word_list_2}],
                    "selected_words": set(self._word_list_1)}
        commit.reset_mock()
        widget = self.create_widget(OWWordList, stored_settings=settings)
        commit.assert_called_once()

        commit.reset_mock()
        words_var = [StringVariable("S1"), StringVariable("S2")]
        lst = [["foo", "A"], ["bar", "B"], ["foobar", "C"]]
        words = Table.from_list(Domain([], metas=words_var), lst)
        self.send_signal(widget.Inputs.words, words)
        commit.assert_called_once()

    def test_saved_settings(self):
        self.widget._set_selected_word_list(1)
        self.widget.library_model[1].name = "New title"
        model = self.widget.words_model
        model.setData(model.index(0, 0), "changed", Qt.EditRole)
        self.widget._set_selected_words([0, 2])

        settings = self.widget.settingsHandler.pack_data(self.widget)
        self.assertEqual(settings["word_list_index"], 1)
        self.assertEqual(settings["word_list_library"][1]["name"], "New title")
        self.assertSetEqual(settings["selected_words"], {"changed", "word 3"})
        self.assertListEqual(settings["words"],
                             ["changed", "word 2", "word 3", "word 4"])

    def test_saved_workflow(self):
        settings = {"word_list_library": [{"name": "Hello world",
                                           "words": self._word_list_1},
                                          {"name": "Word list 1",
                                           "words": self._word_list_2}]}
        widget = self.create_widget(OWWordList, stored_settings=settings)
        words = self.get_output(widget.Outputs.words, widget=widget)
        self.assertListEqual(list(words.metas[:, 0]), ["foo", "bar", "baz"])
        swords = self.get_output(widget.Outputs.selected_words, widget=widget)
        self.assertIsNone(swords)

    def test_saved_workflow_with_changed_words(self):
        settings = {"word_list_library": [{"name": "Hello world",
                                           "words": self._word_list_1},
                                          {"name": "Word list 1",
                                           "words": self._word_list_2}],
                    "words": ["bar", "foo"]}
        widget = self.create_widget(OWWordList, stored_settings=settings)
        words = self.get_output(widget.Outputs.words, widget=widget)
        self.assertListEqual(list(words.metas[:, 0]), ["bar", "foo"])
        swords = self.get_output(widget.Outputs.selected_words, widget=widget)
        self.assertIsNone(swords)

    def test_saved_workflow_with_selection(self):
        settings = {"word_list_library": [{"name": "Hello world",
                                           "words": self._word_list_1},
                                          {"name": "Word list 1",
                                           "words": self._word_list_2}],
                    "words": ["bar", "foo"],
                    "selected_words": {"foo"}}
        widget = self.create_widget(OWWordList, stored_settings=settings)
        words = self.get_output(widget.Outputs.words, widget=widget)
        self.assertListEqual(list(words.metas[:, 0]), ["bar", "foo"])
        swords = self.get_output(widget.Outputs.selected_words, widget=widget)
        self.assertListEqual(list(swords.metas[:, 0]), ["foo"])

    def test_saved_workflow_with_input(self):
        settings = {"word_list_library": [{"name": "Hello world",
                                           "words": self._word_list_1},
                                          {"name": "Word list 1",
                                           "words": self._word_list_2}],
                    "words": ["bar", "foo"],
                    "selected_words": {"foo"}}
        widget = self.create_widget(OWWordList, stored_settings=settings)
        self.send_signal(widget.Inputs.words, self._input_table, widget=widget)
        words = self.get_output(widget.Outputs.words, widget=widget)
        self.assertListEqual(list(words.metas[:, 0]), ["bar", "foo"])
        swords = self.get_output(widget.Outputs.selected_words, widget=widget)
        self.assertListEqual(list(swords.metas[:, 0]), ["foo"])


if __name__ == "__main__":
    unittest.main()
