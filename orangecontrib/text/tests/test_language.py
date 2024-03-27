import unittest
from unittest import TestCase

import numpy as np
from Orange.data import StringVariable, Domain

from orangecontrib.text import Corpus
from orangecontrib.text.language import detect_language, ISO2LANG, LanguageModel


class TestLanguageModel(TestCase):
    def test_model_without_languages(self):
        # no None, all languages
        lm = LanguageModel()
        self.assertEqual(len(ISO2LANG) - 1, lm.rowCount())
        all_langs = [lm.data(lm.index(i)) for i in range(lm.rowCount())]
        expected = sorted(list(ISO2LANG.values())[:-1])
        self.assertEqual(expected, all_langs)

        lm = LanguageModel(include_none=True)
        self.assertEqual(len(ISO2LANG), lm.rowCount())
        all_langs = [lm.data(lm.index(i)) for i in range(lm.rowCount())]
        expected = sorted(list(ISO2LANG.values())[:-1])
        self.assertEqual(["(no language)"] + expected, all_langs)

    def test_model_with_languages(self):
        lm = LanguageModel(include_none=True, languages=["en", "ar", "it"])
        self.assertEqual(4, lm.rowCount())
        all_langs = [lm.data(lm.index(i)) for i in range(lm.rowCount())]
        self.assertEqual(["(no language)", "Arabic", "English", "Italian"], all_langs)

        lm = LanguageModel(languages=["en", "ar", "it"])
        self.assertEqual(3, lm.rowCount())
        all_langs = [lm.data(lm.index(i)) for i in range(lm.rowCount())]
        self.assertEqual(["Arabic", "English", "Italian"], all_langs)


class TestLanguage(TestCase):
    def test_language_detect(self):
        data = Corpus.from_numpy(
            Domain([], metas=[StringVariable("Text")]),
            np.empty((3, 0)),
            metas=np.array(
                [["Today is a nice day"], ["It is a sunny day"], ["Monday"]]
            ),
        )
        self.assertEqual(detect_language(data), "en")

        data = Corpus.from_numpy(
            Domain([], metas=[StringVariable("Text")]),
            np.empty((3, 0)),
            metas=np.array(
                [["Danes je lep dan"], ["Danes je sončen dan"], ["Ponedeljek"]]
            ),
        )
        self.assertEqual(detect_language(data), "sl")

        # the case where language detector would yiled "so" which is not supported
        # by Orange, it must then yield a language that is next on the list
        # in thi case the only detected language is "so" - result in None
        data = Corpus.from_numpy(
            Domain([], metas=[StringVariable("Text")]),
            np.empty((3, 0)),
            metas=np.array([["aaaaa"], ["aaaaa"], ["aaaa"]]),
        )
        self.assertIsNone(detect_language(data))

        # in thi case "so" and "hu" are detected - result in "hu""
        data = Corpus.from_numpy(
            Domain([], metas=[StringVariable("Text")]),
            np.empty((3, 0)),
            metas=np.array([["aaaaa"], ["bbbbb"], ["aaaa"]]),
        )
        self.assertEqual(detect_language(data), "hu")

    def test_languages_sorted(self):
        """ Test whether language dictionary is sorted alphabetically"""
        languages = list(ISO2LANG)[:-1]  # remove last element which is None
        self.assertListEqual(languages, sorted(languages))


if __name__ == "__main__":
    unittest.main()
