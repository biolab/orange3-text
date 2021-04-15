import unittest
from math import isclose
from typing import List, Union
from unittest.mock import patch

import numpy as np
from AnyQt.QtCore import Qt
from Orange.widgets.tests.base import WidgetTest
from Orange.data import Table, StringVariable, Domain, ContinuousVariable
from Orange.widgets.tests.utils import simulate
from Orange.misc.collections import natural_sorted

from orangecontrib.text import Corpus
from orangecontrib.text import preprocess
from orangecontrib.text.vectorization.document_embedder import DocumentEmbedder
from orangecontrib.text.widgets.owscoredocuments import OWScoreDocuments


def embedding_mock(_, corpus, __):
    if isinstance(corpus, list):
        return np.ones((len(corpus), 10))
    else:  # corpus is Corpus
        return (
            Corpus(
                domain=Domain([ContinuousVariable(str(i)) for i in range(10)]),
                X=np.ones((len(corpus), 10)),
            ),
            None,
        )


class TestOWScoreDocuments(WidgetTest):
    @staticmethod
    def create_words_table(words: List[str]) -> Table:
        w = StringVariable("Words")
        w.attributes["type"] = "words"
        return Table(
            Domain([], metas=[w]),
            np.empty((len(words), 0)),
            metas=np.array(words).reshape((-1, 1)),
        )

    def setUp(self) -> None:
        self.widget: OWScoreDocuments = self.create_widget(OWScoreDocuments)

        # create corpus
        self.corpus = Corpus.from_file("book-excerpts")
        pp_list = [
            preprocess.LowercaseTransformer(),
            preprocess.StripAccentsTransformer(),
            preprocess.SnowballStemmer(),
        ]
        for p in pp_list:
            self.corpus = p(self.corpus)

        # create words table
        words = ["house", "doctor", "boy", "way", "Rum"]
        self.words = self.create_words_table(words)

    def test_set_data(self):
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(
            [x[0] for x in self.widget.model], self.corpus.titles.tolist()
        )
        self.assertTrue(self.widget.Warning.missing_words.is_shown())

        self.send_signal(self.widget.Inputs.words, self.words)
        self.assertFalse(self.widget.Warning.missing_words.is_shown())
        self.wait_until_finished()
        self.assertEqual(len(self.widget.model), len(self.corpus))
        self.assertTrue(all(len(x) == 2 for x in self.widget.model))

        output = self.get_output(self.widget.Outputs.corpus)
        self.assertTupleEqual(
            output.domain.variables, self.corpus.domain.variables
        )
        self.assertTupleEqual(
            output.domain.metas[:-1], self.corpus.domain.metas
        )
        self.assertEqual(str(output.domain.metas[-1]), "Word count")
        self.assertEqual(len(output), len(self.corpus))

    def test_corpus_not_normalized(self):
        # send non-normalized corpus
        non_normalized_corpus = Corpus.from_file("book-excerpts")
        self.send_signal(self.widget.Inputs.corpus, non_normalized_corpus)
        self.assertTrue(self.widget.Warning.corpus_not_normalized.is_shown())

        # when sending normalized corpus error should disappear
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertFalse(self.widget.Warning.corpus_not_normalized.is_shown())

    def test_guess_word_attribute(self):
        """
        Test if correct column with words was selected:
        - if type: words attribute is present select the first column with
          type: words attribute
        - else select the attribute with shortest average sequence length
          measured in the number of words
        """
        w = StringVariable("Words")
        w.attributes["type"] = "words"
        w1 = StringVariable("Words 1")
        words = np.array(["house", "doctor", "boy", "way", "Rum"]).reshape(
            (-1, 1)
        )
        words1 = np.array(["house", "doctor1", "boy", "way", "Rum"]).reshape(
            (-1, 1)
        )

        # guess by attribute type
        self.words = Table(
            Domain([], metas=[w, w1]),
            np.empty((len(words), 0)),
            metas=np.hstack([words, words1]),
        )
        self.send_signal(self.widget.Inputs.words, self.words)
        self.assertListEqual(self.widget.words, words.flatten().tolist())

        # reversed order
        self.words = Table(
            Domain([], metas=[w1, w]),
            np.empty((len(words), 0)),
            metas=np.hstack([words1, words]),
        )
        self.send_signal(self.widget.Inputs.words, self.words)
        self.assertListEqual(self.widget.words, words.flatten().tolist())

        # guess by length
        w2 = StringVariable("Words 2")
        words2 = np.array(["house 1", "doctor 1", "boy", "way", "Rum"]).reshape(
            (-1, 1)
        )
        self.words = Table(
            Domain([], metas=[w2, w1]),
            np.empty((len(words), 0)),
            metas=np.hstack([words2, words]),
        )
        self.send_signal(self.widget.Inputs.words, self.words)
        self.assertListEqual(self.widget.words, words.flatten().tolist())

        self.send_signal(self.widget.Inputs.words, None)
        self.assertIsNone(self.widget.words)

    def test_missing_corpus(self):
        self.assertFalse(self.widget.Warning.missing_corpus.is_shown())
        self.send_signal(self.widget.Inputs.words, self.words)
        self.assertTrue(self.widget.Warning.missing_corpus.is_shown())
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertFalse(self.widget.Warning.missing_corpus.is_shown())
        self.assertIsNotNone(self.get_output(self.widget.Outputs.corpus))
        self.send_signal(self.widget.Inputs.corpus, None)
        self.assertTrue(self.widget.Warning.missing_corpus.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.corpus))

    @patch.object(DocumentEmbedder, "__call__", new=embedding_mock)
    def test_change_scorer(self):
        model = self.widget.model
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)
        self.wait_until_finished()
        self.assertEqual(len(model[0]), 2)
        self.assertEqual(model.headerData(1, Qt.Horizontal), "Word count")

        self.widget.controls.word_appearance.click()
        self.wait_until_finished()
        self.assertEqual(len(self.widget.model[0]), 3)
        self.assertEqual(model.headerData(1, Qt.Horizontal), "Word count")
        self.assertEqual(model.headerData(2, Qt.Horizontal), "Word in text")

        self.widget.controls.embedding_similarity.click()
        self.wait_until_finished()
        self.assertEqual(len(self.widget.model[0]), 4)
        self.assertEqual(model.headerData(1, Qt.Horizontal), "Word count")
        self.assertEqual(model.headerData(2, Qt.Horizontal), "Word in text")
        self.assertEqual(model.headerData(3, Qt.Horizontal), "Similarity")

        self.widget.controls.word_frequency.click()
        self.wait_until_finished()
        self.assertEqual(len(self.widget.model[0]), 3)
        self.assertEqual(model.headerData(1, Qt.Horizontal), "Word in text")
        self.assertEqual(model.headerData(2, Qt.Horizontal), "Similarity")

    @staticmethod
    def create_corpus(texts: List[str]) -> Corpus:
        """ Create sample corpus with texts passed """
        text_var = StringVariable("Text")
        domain = Domain([], metas=[text_var])
        c = Corpus(
            domain,
            metas=np.array(texts).reshape(-1, 1),
            text_features=[text_var],
        )
        return preprocess.LowercaseTransformer()(c)

    def test_word_frequency(self):
        corpus = self.create_corpus(
            [
                "Lorem ipsum dolor sit ipsum, consectetur adipiscing elit.",
                "Sed eu sollicitudin velit lorem.",
                "lorem ipsum eu",
            ]
        )
        words = self.create_words_table(["lorem", "ipsum", "eu"])
        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.send_signal(self.widget.Inputs.words, words)
        self.wait_until_finished()
        self.assertListEqual([x[1] for x in self.widget.model], [1, 2 / 3, 1])

        cb_aggregation = self.widget.controls.aggregation
        simulate.combobox_activate_item(cb_aggregation, "Max")
        self.wait_until_finished()
        self.assertListEqual([x[1] for x in self.widget.model], [2, 1, 1])

        simulate.combobox_activate_item(cb_aggregation, "Min")
        self.wait_until_finished()
        self.assertListEqual([x[1] for x in self.widget.model], [0, 0, 1])

        simulate.combobox_activate_item(cb_aggregation, "Median")
        self.wait_until_finished()
        self.assertListEqual([x[1] for x in self.widget.model], [1, 1, 1])

    def test_word_appearance(self):
        corpus = self.create_corpus(
            [
                "Lorem ipsum dolor sit ipsum, consectetur adipiscing elit.",
                "Sed eu sollicitudin velit lorem.",
                "lorem ipsum eu",
            ]
        )
        words = self.create_words_table(["lorem", "ipsum", "eu"])
        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.send_signal(self.widget.Inputs.words, words)
        # unselect word_frequency and select word_ratio
        self.widget.controls.word_frequency.click()
        self.widget.controls.word_appearance.click()
        self.wait_until_finished()
        self.assertListEqual(
            [x[1] for x in self.widget.model], [2 / 3, 2 / 3, 1]
        )

        cb_aggregation = self.widget.controls.aggregation
        simulate.combobox_activate_item(cb_aggregation, "Max")
        self.wait_until_finished()
        self.assertListEqual([x[1] for x in self.widget.model], [1, 1, 1])

        simulate.combobox_activate_item(cb_aggregation, "Min")
        self.wait_until_finished()
        self.assertListEqual([x[1] for x in self.widget.model], [0, 0, 1])

        simulate.combobox_activate_item(cb_aggregation, "Median")
        self.wait_until_finished()
        self.assertListEqual([x[1] for x in self.widget.model], [1, 1, 1])

    @patch.object(DocumentEmbedder, "__call__", new=embedding_mock)
    def test_embedding_similarity(self):
        corpus = self.create_corpus(
            [
                "Lorem ipsum dolor sit ipsum, consectetur adipiscing elit.",
                "Sed eu sollicitudin velit lorem.",
                "lorem ipsum eu",
            ]
        )
        words = self.create_words_table(["lorem", "ipsum", "eu"])
        self.send_signal(self.widget.Inputs.corpus, corpus)
        self.send_signal(self.widget.Inputs.words, words)
        # unselect word_frequency and select embedding_similarity
        self.widget.controls.word_frequency.click()
        self.widget.controls.embedding_similarity.click()
        self.wait_until_finished()
        self.assertTrue(all(isclose(x[1], 1) for x in self.widget.model))

    def test_filter(self):
        view = self.widget.view
        model = view.model()
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)

        self.widget._filter_line_edit.setText("Document 15")
        data = [model.data(model.index(i, 0)) for i in range(model.rowCount())]
        self.assertListEqual(data, ["Document 15"])

    def test_sort_table(self):
        """
        Test if first column of the table is sorted naturally
        """
        view = self.widget.view
        model = view.model()
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.send_signal(self.widget.Inputs.words, self.words)

        view.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        data = [model.data(model.index(i, 0)) for i in range(model.rowCount())]
        self.assertListEqual(data, natural_sorted(self.corpus.titles))

        view.horizontalHeader().setSortIndicator(0, Qt.DescendingOrder)
        data = [model.data(model.index(i, 0)) for i in range(model.rowCount())]
        self.assertListEqual(data, natural_sorted(self.corpus.titles)[::-1])


if __name__ == "__main__":
    unittest.main()
