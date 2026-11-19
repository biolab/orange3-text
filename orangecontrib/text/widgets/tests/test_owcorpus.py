import os
import tempfile
import unittest
import shutil
import pickle

import numpy as np
from Orange.data import Table, Domain, StringVariable, ContinuousVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.text import Corpus
from orangecontrib.text.preprocess import RegexpTokenizer
from orangecontrib.text.widgets.owcorpus import OWCorpus


class TestOWCorpus(WidgetTest):
    def setUp(self):
        self.widget: OWCorpus = self.create_widget(OWCorpus)

    def check_output(self, sel_title):
        """
        This function check whether the `sel_title` variable has a title true
        in the output
        """
        output = self.get_output(self.widget.Outputs.corpus)
        for attr in output.domain.variables + output.domain.metas:
            if str(attr) == sel_title:
                # sel_title attribute must be marked as a title
                self.assertTrue(attr.attributes.get("title", False))
            else:
                # others must not be marked as a title
                self.assertFalse(attr.attributes.get("title", False))

    def test_title_combo(self):
        self.wait_until_finished()
        # default corpus dataset
        self.assertEqual(self.widget.corpus.name, "book-excerpts")

        options = self.widget.title_model[:]
        self.assertIn(self.widget.corpus.domain["Text"], options)
        # for this dataset no title variable is selected
        self.assertEqual(None, self.widget.title_variable)
        self.check_output(None)

    def test_title_already_in_dataset(self):
        """
        This dataset already have the title attribute so the title option
        is set to this attribute by default
        """
        # default corpus dataset
        data = Corpus.from_file("election-tweets-2016")
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()

        self.assertEqual(data.domain["Content"], self.widget.title_variable)
        self.check_output("Content")

    def test_title_selection_strategy_title_heading(self):
        """
        When a there is a title, heading, filename attribute, select this one
        as a default title.
        """
        data = Table(
            Domain([], metas=[StringVariable("title"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["title"], self.widget.title_variable)
        self.check_output("title")

        data = Table(
            Domain([], metas=[StringVariable("Title"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Title"], self.widget.title_variable)
        self.check_output("Title")

        # when title and heading present first select title
        data = Table(
            Domain([], metas=[
                StringVariable("Title"),
                StringVariable("Heading"),
                StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Title"], self.widget.title_variable)
        self.check_output("Title")

        data = Table(
            Domain([], metas=[
                StringVariable("Heading"),
                StringVariable("Title"),
                StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Title"], self.widget.title_variable)
        self.check_output("Title")

        data = Table(
            Domain([], metas=[
                StringVariable("Heading"),
                StringVariable("Filename"),
                StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["Heading"], self.widget.title_variable)
        self.check_output("Heading")

    def test_title_selection_strategy_most_unique(self):
        """
        With this test we test whether the selection strategy for a title
        attribute works correctly.
        This one must select the most unique feature.
        """
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b")]),
            np.empty((3, 0)),
            metas=[["a" * 10, "a" * 10],
                   ["a" * 10, "b" * 10],
                   ["a" * 10, "c" * 10]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["b"], self.widget.title_variable)
        self.check_output("b")

    def test_title_selection_strategy_shortness(self):
        """
        With this test we test whether the selection strategy for a title
        attribute works correctly.
        Select the uniquest and also short enough, here attribute a is not
        suitable since it has too long title, and c is more unique than b.
        """
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 10, "a" * 10],
                   ["b" * 100, "a" * 10, "b" * 10],
                   ["c" * 100, "a" * 10, "b" * 10]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["c"], self.widget.title_variable)
        self.check_output("c")

    def test_title_selection_strategy_shortest(self):
        """
        With this test we test whether the selection strategy for a title
        attribute works correctly.
        When no variable is short enough we just select the shortest attribute.
        """
        data = Table(
            Domain([], metas=[StringVariable("a"), StringVariable("b"),
                              StringVariable("c")]),
            np.empty((3, 0)),
            metas=[["a" * 100, "a" * 40, "a" * 40],
                   ["b" * 100, "a" * 40, "b" * 30],
                   ["c" * 100, "a" * 40, "b" * 40]]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual(data.domain["c"], self.widget.title_variable)
        self.check_output("c")

    def test_keep_selected_variables(self):
        """
        When domain just slightly changes selected text variables should
        still be same.
        """
        attributes = [
            ContinuousVariable("a"), ContinuousVariable("b"),
            ContinuousVariable("c")]
        metas = [StringVariable("d"), StringVariable("e"),
                 StringVariable("f"), StringVariable("g"),
                 StringVariable("h"), StringVariable("i")]
        data = Table(
            Domain(attributes, metas=metas),
            np.array([[0] * len(attributes)]),
            metas=[["a" * 10] * len(metas)]
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        prew_selected = data.domain.metas[1:3]
        self.widget.used_attrs = prew_selected

        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_finished()

        data = Table.from_table(
            Domain(attributes[:-1], [], metas=metas), data
        )
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertListEqual(list(prew_selected), self.widget.used_attrs)

    def test_multiple_text_features(self):
        """
        Test whether the widget properly stores multiple text_features. It
        must store them both in text_features and in the attributes of
        attributes.
        """
        data = Corpus.from_file("grimm-tales")
        old_features = len(data.text_features)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        # add one more text feature, namely Title
        self.widget.used_attrs_model.append(data.domain.metas[0])
        self.widget.update_feature_selection()
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertNotEqual(old_features, len(output.text_features))
        self.assertEqual(len(output.text_features), 2)
        self.assertTrue(output.domain.metas[0].attributes["include"])
        # remove one text feature, namely Content
        self.widget.used_attrs_model.remove(data.domain.metas[2])
        self.widget.update_feature_selection()
        output = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(len(output.text_features), 1)


    def test_no_text_feature(self):
        """
        Test with data which have empty text_features. Widget should not show
        the error but, should have all features unused.
        """
        # widget already loads book-excerpts from file and store context
        # settings this call restore context settings to default otherwise
        # Text variable is moved to used_attributes by the context
        self.widget.settingsHandler.reset_to_original(self.widget)
        data = Corpus.from_file("book-excerpts")
        data.text_features = []
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertFalse(
            self.widget.Error.corpus_without_text_features.is_shown()
        )
        self.assertEqual(0, len(list(self.widget.used_attrs_model)))
        self.assertListEqual(
            [data.domain["Text"]],
            list(self.widget.unused_attrs_model)
        )

    def test_corpus_without_text_features(self):
        """
        Test if corpus_without_text_features is correctly raised for data
        without text features
        """
        data = Table("iris")
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertTrue(
            self.widget.Error.corpus_without_text_features.is_shown()
        )

    def test_context(self):
        data = Table(Corpus.from_file("book-excerpts"))
        data.attributes["language"] = "sl"
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual("sl", self.widget.language)
        self.assertEqual("sl", self.get_output(self.widget.Outputs.corpus).language)

        # change language to see if context work later when reopened
        simulate.combobox_activate_item(self.widget.controls.language, "Dutch")
        self.assertEqual("nl", self.widget.language)
        self.assertEqual("nl", self.get_output(self.widget.Outputs.corpus).language)

        data1 = Table(Corpus.from_file("deerwester"))
        self.send_signal(self.widget.Inputs.data, data1)
        self.wait_until_finished()
        self.assertEqual("en", self.widget.language)
        self.assertEqual("en", self.get_output(self.widget.Outputs.corpus).language)

        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual("nl", self.widget.language)
        self.assertEqual("nl", self.get_output(self.widget.Outputs.corpus).language)

        # when corpus on input in different language do not match
        data.attributes["language"] = "sk"
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual("sk", self.widget.language)
        self.assertEqual("sk", self.get_output(self.widget.Outputs.corpus).language)

        # different documents in corpus (should not match the context)
        data2 = data[:10]
        data2.attributes["language"] = "sl"
        self.send_signal(self.widget.Inputs.data, data2)
        self.wait_until_finished()
        self.assertEqual("sl", self.widget.language)
        self.assertEqual("sl", self.get_output(self.widget.Outputs.corpus).language)

    def test_guess_language(self):
        data = Table(Corpus.from_file("book-excerpts"))
        # since Table is made from Corpus language attribute is in attributes
        # drop it
        data.attributes = {}
        # change default to something that is not corpus's language
        self.widget.language = "sl"
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual("en", self.widget.language)
        self.assertEqual("en", self.get_output(self.widget.Outputs.corpus).language)

        # change language to see if context work later when reopened
        simulate.combobox_activate_item(self.widget.controls.language, "Dutch")
        self.assertEqual("nl", self.widget.language)
        self.assertEqual("nl", self.get_output(self.widget.Outputs.corpus).language)

        data1 = Table(Corpus.from_file("deerwester"))
        self.send_signal(self.widget.Inputs.data, data1)
        self.wait_until_finished()
        self.assertEqual("en", self.widget.language)
        self.assertEqual("en", self.get_output(self.widget.Outputs.corpus).language)

        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_finished()
        self.assertEqual("nl", self.widget.language)
        self.assertEqual("nl", self.get_output(self.widget.Outputs.corpus).language)

        # different documents in corpus (should not match the context)
        data2 = data[:10]
        data2.attributes["language"] = None
        self.send_signal(self.widget.Inputs.data, data2)
        self.wait_until_finished()
        self.assertEqual("en", self.widget.language)
        self.assertEqual("en", self.get_output(self.widget.Outputs.corpus).language)

    def test_language_unpickle(self):
        path = os.path.dirname(__file__)
        file = os.path.abspath(os.path.join(path, "..", "..", "tests",
                                            "data", "book-excerpts.pkl"))
        corpus = Corpus.from_file(file)
        self.send_signal(self.widget.Inputs.data, corpus)
        self.wait_until_finished()
        self.assertEqual(self.widget.language, "en")

    def test_preserve_preprocessing(self):
        """When preprocessed corpus on input preprocessing should be retained"""
        corpus = Corpus.from_file("andersen")
        corpus = RegexpTokenizer()(corpus)

        # preprocessing should be maintained
        self.send_signal(self.widget.Inputs.data, corpus)
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertTrue(res.has_tokens())

        # add additional text feature - preprocessing should be reset
        self.widget.used_attrs_model.append(corpus.domain.metas[0])
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertFalse(res.has_tokens())

        # remove previously added feature - preprocessing should be kept again
        self.widget.used_attrs_model.remove(corpus.domain.metas[0])
        res = self.get_output(self.widget.Outputs.corpus)
        self.assertTrue(res.has_tokens())

    def test_preserve_preprocessing_from_file(self):
        """When preprocessed corpus loaded preprocessing should be retained"""
        corpus = Corpus.from_file("andersen")
        corpus = RegexpTokenizer()(corpus)
        with tempfile.TemporaryDirectory() as tmp_dir:
            file = os.path.join(tmp_dir, "andersen.pkl")
            corpus.save(file)
            self.widget.file_widget.open_file(file)

            # preprocessing should be maintained
            self.send_signal(self.widget.Inputs.data, corpus)
            res = self.get_output(self.widget.Outputs.corpus)
            self.assertTrue(res.has_tokens())

            # add additional text feature - preprocessing should be reset
            self.widget.used_attrs_model.append(corpus.domain.metas[0])
            res = self.get_output(self.widget.Outputs.corpus)
            self.assertFalse(res.has_tokens())

            # remove previously added feature - preprocessing should be kept again
            self.widget.used_attrs_model.remove(corpus.domain.metas[0])
            res = self.get_output(self.widget.Outputs.corpus)
            self.assertTrue(res.has_tokens())

    def test_migrate_settings(self):
        corpus = Corpus.from_file("book-excerpts")
        self.send_signal(self.widget.Inputs.data, corpus)
        self.wait_until_finished()
        packed_data = self.widget.settingsHandler.pack_data(self.widget)
        packed_data["context_settings"][0].values["language"] = ("French", -2)
        packed_data["context_settings"][0].values["__version__"] = 1

        widget = self.create_widget(OWCorpus, stored_settings=packed_data)
        self.send_signal(self.widget.Inputs.data, corpus, widget=widget)
        self.wait_until_finished(widget=widget)
        self.assertEqual("fr", widget.language)

        packed_data["context_settings"][0].values["language"] = ("Ancient greek", -2)
        widget = self.create_widget(OWCorpus, stored_settings=packed_data)
        self.send_signal(self.widget.Inputs.data, corpus, widget=widget)
        self.wait_until_finished(widget=widget)
        self.assertEqual("grc", widget.language)

        packed_data["context_settings"][0].values["language"] = (None, -2)
        widget = self.create_widget(OWCorpus, stored_settings=packed_data)
        self.send_signal(self.widget.Inputs.data, corpus, widget=widget)
        self.wait_until_finished(widget=widget)
        self.assertIsNone(widget.language)

    def test_relative_corpus_path_serialization(self):
        """
        Test if relative paths are properly saved and reloaded.
        """
        # Create a dummy corpus file
        with tempfile.TemporaryDirectory() as tmp_dir:
            corpus = Corpus.from_file("book-excerpts")
            corpus_path = os.path.join(tmp_dir, "test.corpus")
            with open(corpus_path, "wb") as f:
                pickle.dump(corpus, f)

            # Simulate loading the file into widget
            self.widget.workflow_file = os.path.join(tmp_dir, "workflow.ows")
            self.widget.corpus_path = corpus_path

            settings = {}
            self.widget.save_settings(settings)

            # Simulate moving workflow and corpus to new directory
            with tempfile.TemporaryDirectory() as new_dir:
                new_corpus = os.path.join(new_dir, "test.corpus")
                new_workflow = os.path.join(new_dir, "workflow.ows")
                shutil.copy2(corpus_path, new_corpus)

                # Simulate loading settings in new widget
                restored = self.create_widget(OWCorpus)
                restored.workflow_file = new_workflow
                settings["corpus_path"] = os.path.relpath(new_corpus, new_dir)
                restored.load_settings(settings)

                self.assertTrue(os.path.exists(restored.corpus_path))
                self.assertTrue(os.path.isabs(restored.corpus_path))

if __name__ == "__main__":
    unittest.main()
