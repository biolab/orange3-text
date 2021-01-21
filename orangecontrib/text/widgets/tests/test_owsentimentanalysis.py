import os
import numpy as np

from unittest import mock, skip
from unittest.mock import patch

from numpy import array_equal

from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.widgets.owsentimentanalysis import OWSentimentAnalysis


class TestSentimentWidget(WidgetTest):
    def setUp(self):
        self.widget = self.create_widget(OWSentimentAnalysis)
        self.corpus = Corpus.from_file('deerwester')

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
        self.assertEqual(len(out_corpus.domain), len(self.corpus.domain) + 4)

        # test multisentiment
        self.widget.multi_sent.click()
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(len(out_corpus.domain), len(self.corpus.domain) + 1)

        # test SentiArt
        self.widget.senti_art.click()
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(len(out_corpus.domain), len(self.corpus.domain) + 7)

        # test liu hu
        self.widget.liu_hu.click()
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(len(out_corpus.domain), len(self.corpus.domain) + 1)

        # test custom files
        self.widget.pos_file = os.path.join(os.path.dirname(__file__),
                                            "data/sentiment/pos.txt")
        self.widget.neg_file = os.path.join(os.path.dirname(__file__),
                                            "data/sentiment/neg.txt")
        self.widget.custom_list.click()
        out_corpus = self.get_output(self.widget.Outputs.corpus)
        self.assertEqual(len(out_corpus.domain), len(self.corpus.domain) + 1)
        res = np.array([[12.5], [10], [16.66666667], [12.5], [11.11111111],
                        [-14.28571429], [0], [-10], [0]])
        np.testing.assert_array_almost_equal(out_corpus.X, res, decimal=8)

    def test_language_changed(self):
        """Test if output changes on language change"""
        self.send_signal(self.widget.Inputs.corpus, self.corpus)
        self.assertEqual(self.widget.multi_box.count(),
                         len(self.widget.MULTI_LANG))

        # default for Liu Hu should be English
        self.widget.liu_hu.click()
        simulate.combobox_activate_item(self.widget.liu_lang, "English")
        output_eng = self.get_output(self.widget.Outputs.corpus)

        simulate.combobox_activate_item(self.widget.liu_lang, "Slovenian")
        output_slo = self.get_output(self.widget.Outputs.corpus)
        self.assertFalse(array_equal(output_eng.X, output_slo.X))

    @skip("Re-enable when possible. Currently fails on Travis.")
    def test_sentiment_offline(self):
        """Test if sentiment works with offline lexicons"""
        with patch("orangecontrib.text.sentiment.SentimentDictionaries.online",
                   new_callable=mock.PropertyMock, return_value=False):
            widget = self.create_widget(OWSentimentAnalysis)
            self.send_signal(widget.Inputs.corpus, self.corpus)
            widget.multi_sent.click()
            self.assertTrue(widget.Warning.senti_offline.is_shown())

    def test_no_file_warnings(self):
        widget = self.create_widget(OWSentimentAnalysis)
        self.send_signal(widget.Inputs.corpus, self.corpus)
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())
        widget.custom_list.click()
        self.assertTrue(widget.Warning.no_dicts_loaded.is_shown())
        widget.pos_file = os.path.join(os.path.dirname(__file__),
                                       "data/sentiment/pos.txt")
        widget.commit()
        self.assertTrue(widget.Warning.one_dict_only.is_shown())
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())
        widget.neg_file = os.path.join(os.path.dirname(__file__),
                                       "data/sentiment/neg.txt")
        widget.commit()
        self.assertFalse(widget.Warning.one_dict_only.is_shown())
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())
        widget.vader.click()
        self.assertFalse(widget.Warning.one_dict_only.is_shown())
        self.assertFalse(widget.Warning.no_dicts_loaded.is_shown())
