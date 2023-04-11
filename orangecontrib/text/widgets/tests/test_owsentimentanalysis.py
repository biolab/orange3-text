import os
import unittest
from unittest import mock, skip
from unittest.mock import patch

import numpy as np
from AnyQt.QtWidgets import QRadioButton
from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text import preprocess
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owsentimentanalysis import OWSentimentAnalysis

MS_FILES = [
    ("negative_words_de.txt",),
    ("negative_words_en.txt",),
    ("negative_words_es.txt",),
    ("negative_words_fr.txt",),
    ("negative_words_sl.txt",),
    ("positive_words_de.txt",),
    ("positive_words_en.txt",),
    ("positive_words_es.txt",),
    ("positive_words_fr.txt",),
    ("positive_words_sl.txt",),
]
LILAH_FILES = [("LiLaH-HR.pickle",), ("LiLaH-NL.pickle",), ("LiLaH-SL.pickle",)]
SENTI_FILES = [("SentiArt_DE.pickle",), ("SentiArt_EN.pickle",)]
LISTFILES = {
    "http://file.biolab.si/files/sentiment/": MS_FILES,
    "http://file.biolab.si/files/sentiart/": SENTI_FILES,
    "http://file.biolab.si/files/sentiment-lilah/": LILAH_FILES,
}
MOCK_FUN = "orangecontrib.text.sentiment.serverfiles.ServerFiles.listfiles"

def dummy_listfiles(sf):
    return LISTFILES[sf.server]


@patch(MOCK_FUN, dummy_listfiles)
class TestSentimentWidget(WidgetTest):
    @patch(MOCK_FUN, dummy_listfiles)
    def setUp(self):
        self.widget = self.create_widget(OWSentimentAnalysis)
        self.corpus = Corpus.from_file("deerwester")

    def test_set_corpus(self):
        """
        Just a basic test.
        """
        self.send_signal("Corpus", self.corpus)

    def test_output(self):
        """Test if new column on the output"""
        self.send_signal(self.widget.Inputs.corpus, self.corpus)

        # test default settings
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(out_corpus.domain.variables), len(self.corpus.domain.variables) + 4
        )

        # test multisentiment
        self.widget.findChildren(QRadioButton)[2].click()  # multi senti
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(out_corpus.domain.variables), len(self.corpus.domain.variables) + 1
        )

        # test SentiArt
        self.widget.findChildren(QRadioButton)[3].click()  # senti art
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(out_corpus.domain.variables), len(self.corpus.domain.variables) + 7
        )

        # test Lilah sentiment
        self.widget.lilah_sent.click()
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(out_corpus.domain.variables), len(self.corpus.domain.variables) + 10
        )

        # test liu hu
        self.widget.findChildren(QRadioButton)[0].click()  # liu hu
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(out_corpus.domain.variables), len(self.corpus.domain.variables) + 1
        )

        # test custom files
        self.widget.pos_file = os.path.join(
            os.path.dirname(__file__), "data/sentiment/pos.txt"
        )
        self.widget.neg_file = os.path.join(
            os.path.dirname(__file__), "data/sentiment/neg.txt"
        )
        self.widget.findChildren(QRadioButton)[5].click()  # custom dictionary
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(out_corpus.domain.variables), len(self.corpus.domain.variables) + 1
        )
        res = np.array(
            [
                [12.5],
                [10],
                [16.66666667],
                [12.5],
                [11.11111111],
                [-14.28571429],
                [0],
                [-10],
                [0],
            ]
        )
        np.testing.assert_array_almost_equal(out_corpus.X, res, decimal=8)

        # test Lilah sentiment
        self.corpus.attributes["language"] = "sl"
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.widget.findChildren(QRadioButton)[4].click()  # Lilah
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(
            len(out_corpus.domain.variables), len(self.corpus.domain.variables) + 10
        )

    def test_sentiment_offline(self):
        """Test if sentiment works with offline lexicons"""
        with patch(
            "orangecontrib.text.sentiment.SentimentDictionaries.online",
            new_callable=mock.PropertyMock,
            return_value=False,
        ):
            widget = self.create_widget(OWSentimentAnalysis)
            self.send_signal(widget.Inputs.corpus, self.corpus)
            widget.multi_sent.click()
            self.assertTrue(widget.Warning.senti_offline.is_shown())

    def test_no_file_warnings(self):
        widget = self.create_widget(OWSentimentAnalysis)
        self.send_signal(widget.Inputs.corpus, self.corpus)
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())
        widget.findChildren(QRadioButton)[5].click()  # custom dictionary
        self.assertTrue(widget.Warning.no_dicts_loaded.is_shown())
        widget.pos_file = os.path.join(
            os.path.dirname(__file__), "data/sentiment/pos.txt"
        )
        widget.commit.now()
        self.assertTrue(widget.Warning.one_dict_only.is_shown())
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())
        widget.neg_file = os.path.join(
            os.path.dirname(__file__), "data/sentiment/neg.txt"
        )
        widget.commit.now()
        self.assertFalse(widget.Warning.one_dict_only.is_shown())
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())
        widget.findChildren(QRadioButton)[1].click()  # vader
        self.assertFalse(widget.Warning.one_dict_only.is_shown())
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())

    def test_none_type_input(self):
        # this should not raise an exception
        self.send_signal("Corpus", None)

    def test_migrates_settings(self):
        settings = {"method_idx": 4}
        OWSentimentAnalysis.migrate_settings(settings, version=None)
        self.assertTrue(settings.get("method_idx", 5))

    def test_preprocessed(self):
        widget = self.create_widget(OWSentimentAnalysis)
        corpus = self.corpus.copy()
        pp_list = [preprocess.LowercaseTransformer(), preprocess.WordPunctTokenizer()]
        for pp in pp_list:
            corpus = pp(corpus)
        self.send_signal(widget.Inputs.corpus, corpus)
        self.assertTrue(widget.pp_corpus)
        widget.findChildren(QRadioButton)[0].click()  # vader
        self.assertTrue(widget.pp_corpus)
        self.send_signal(widget.Inputs.corpus, None)
        self.assertIsNone(widget.pp_corpus)


if __name__ == "__main__":
    unittest.main()
