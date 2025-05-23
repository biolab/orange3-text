import pickle
import shutil
import tempfile
import unittest
import os.path
import copy
import itertools
from unittest.mock import patch, Mock

import nltk
from gensim import corpora
from lemmagen3 import Lemmatizer
from nltk.corpus import stopwords
from requests.exceptions import ConnectionError
import numpy as np

from orangecontrib.text import preprocess, tag
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import (
    BASE_TOKENIZER,
    PreprocessorList,
    StopwordsFilter,
)
from orangecontrib.text.preprocess.normalize import UDPipeModels


SF_LIST = "orangecontrib.text.preprocess.normalize.serverfiles.ServerFiles.listfiles"
SF_DOWNLOAD = "orangecontrib.text.preprocess.normalize.serverfiles.ServerFiles.download"
SERVER_FILES = [
    ("slovenian-sst-ud-2.0-170801.udpipe",),
    ("slovenian-ud-2.0-170801.udpipe",),
    ("english-lines-ud-2.0-170801.udpipe",),
    ("english-ud-2.0-170801.udpipe",),
    ("english-partut-ud-2.0-170801.udpipe",),
    ("portuguese-ud-2.0-170801.udpipe",),
    ("lithuanian-ud-2.0-170801.udpipe",),
]


def download_patch(_, *path, **kwargs):
    to_ = kwargs["target"]
    from_ = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    shutil.copyfile(os.path.join(from_, path[0]), to_)


class PreprocessTests(unittest.TestCase):
    sentence = "Human machine interface for lab abc computer applications"

    def setUp(self):
        self.corpus = Corpus.from_file("deerwester")
        self.pp_list = [preprocess.LowercaseTransformer(),
                        preprocess.WordPunctTokenizer(),
                        preprocess.SnowballStemmer(),
                        preprocess.NGrams(),
                        tag.AveragedPerceptronTagger()]

    def test_preprocess(self):
        corpus = self.corpus  # type: Corpus
        self.assertIsNone(corpus._tokens)
        self.assertIsNone(corpus.pos_tags)
        for pp in self.pp_list:
            corpus = pp(corpus)
        self.assertEqual([8, 10, 6, 8, 9, 7, 7, 10, 4],
                         list(map(len, corpus._tokens)))
        self.assertIsNotNone(corpus._tokens)
        self.assertIsNotNone(corpus.pos_tags)
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 5)

    def test_used_preprocessors(self):
        corpus1 = self.corpus.copy()
        for pp in self.pp_list:
            corpus1 = pp(corpus1)
        self.assertEqual(len(self.corpus.used_preprocessor.preprocessors), 0)
        self.assertEqual(len(corpus1.used_preprocessor.preprocessors), 5)

        self.assertEqual([8, 10, 6, 8, 9, 7, 7, 10, 4],
                         list(map(len, corpus1._tokens)))

        corpus2 = PreprocessorList(self.pp_list)(self.corpus)
        np.testing.assert_array_equal(corpus1.tokens, corpus2.tokens)

    def test_apply_preprocessors(self):
        corpus = PreprocessorList(self.pp_list)(self.corpus)
        self.assertEqual([8, 10, 6, 8, 9, 7, 7, 10, 4],
                         list(map(len, corpus._tokens)))
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 5)

    def test_apply_base_preprocessors(self):
        self.assertEqual([8, 10, 6, 8, 9, 7, 7, 10, 4],
                         list(map(len, self.corpus.tokens)))

    def test_string_processor(self):
        p = preprocess.LowercaseTransformer()
        tokens2 = self.corpus.tokens.copy()
        tokens = p(self.corpus).tokens
        np.testing.assert_equal(
            tokens,
            np.array([[t.lower() for t in doc] for doc in tokens2], dtype="object")
        )

    def test_tokenizer(self):
        class SpaceTokenizer(preprocess.BaseTokenizer):
            @classmethod
            def _preprocess(cls, string):
                return string.split()

        p = SpaceTokenizer()
        array = np.array([sent.split() for sent in self.corpus.documents],
                         dtype=object)
        np.testing.assert_equal(p(self.corpus).tokens, array)

    def test_token_normalizer(self):
        class CapTokenNormalizer(preprocess.BaseNormalizer):
            @classmethod
            def _preprocess(cls, token):
                return token.capitalize()

        p = CapTokenNormalizer()
        tokens2 = self.corpus.tokens.copy()
        tokens = p(self.corpus).tokens

        np.testing.assert_equal(
            tokens,
            np.array([[t.capitalize() for t in doc] for doc in tokens2], dtype="object")
        )

    def test_token_filter(self):
        class LengthFilter(preprocess.BaseTokenFilter):
            def _check(self, token):
                return len(token) < 4

        p = LengthFilter()
        tokens = np.array([[token for token in doc.split() if len(token) < 4]
                           for doc in self.corpus.documents], dtype=object)
        np.testing.assert_equal(p(self.corpus).tokens, tokens)

    def test_inplace(self):
        p = preprocess.RegexpTokenizer('\w')
        corpus = p(self.corpus)
        self.assertIsNot(corpus, self.corpus)

    def test_retain_ids(self):
        corpus = self.corpus
        for pp in self.pp_list:
            corpus = pp(corpus)
        self.assertTrue((corpus.ids == self.corpus.ids).all())

    def test_filter_pos_tags(self):
        pp_list = [preprocess.LowercaseTransformer(),
                   preprocess.WordPunctTokenizer(),
                   tag.AveragedPerceptronTagger(),
                   preprocess.StopwordsFilter()]
        corpus = self.corpus
        with corpus.unlocked():
            corpus.metas[0, 0] = "This is the most beautiful day in the world"
        for pp in pp_list:
            corpus = pp(corpus)
        self.assertEqual(len(corpus.tokens), len(corpus.pos_tags))
        self.assertEqual(len(corpus.tokens[0]), len(corpus.pos_tags[0]))
        self.assertEqual(corpus.tokens[0], ["beautiful", "day", "world"])
        self.assertEqual(corpus.pos_tags[0], ["JJ", "NN", "NN"])


class TransformationTests(unittest.TestCase):
    def setUp(self):
        class ReverseStringTransformer(preprocess.BaseTransformer):
            name = 'reverse'

            def _preprocess(self, string):
                return string[::-1]

        self.transformer = ReverseStringTransformer()
        self.corpus = Corpus.from_file("deerwester")

    def test_transform(self):
        trans = self.transformer
        self.assertEqual(trans._preprocess('abracadabra'), 'arbadacarba')

    def test_call(self):
        corpus = self.transformer(self.corpus)
        text = 'snoitacilppa retupmoc cba bal rof ecafretni enihcam namuH'
        self.assertEqual(corpus.pp_documents[0], text)
        self.assertFalse(corpus.has_tokens())
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 1)

    def test_call_with_tokens(self):
        corpus = preprocess.WordPunctTokenizer()(self.corpus)
        corpus = self.transformer(corpus)
        tokens = ['namuH', 'enihcam', 'ecafretni', 'rof', 'bal', 'cba',
                  'retupmoc', 'snoitacilppa']
        self.assertEqual(corpus.tokens[0], tokens)
        self.assertTrue(corpus.has_tokens())
        text = 'Human machine interface for lab abc computer applications'
        self.assertEqual(corpus.documents[0], text)
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_str(self):
        self.assertIn('reverse', str(self.transformer))

    def test_lowercase(self):
        transformer = preprocess.LowercaseTransformer()
        self.assertEqual(transformer._preprocess('Abra'), 'abra')
        self.assertEqual(transformer._preprocess('\u00C0bra'), '\u00E0bra')

    def test_strip_accents(self):
        transformer = preprocess.StripAccentsTransformer()
        self.assertEqual(transformer._preprocess('Abra'), 'Abra')
        self.assertEqual(transformer._preprocess('\u00C0bra'), 'Abra')

    def test_html(self):
        transformer = preprocess.HtmlTransformer()
        self.assertEqual(transformer._preprocess('<p>abra<b>cadabra</b><p>'),
                         'abracadabra')

    def test_url_remover(self):
        remover = preprocess.UrlRemover()
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = 'some link to https://google.com/'
            self.corpus.metas[1, 0] = 'some link to google.com'
        corpus = remover(self.corpus)
        self.assertListEqual(corpus.pp_documents[:2],
                             ['some link to ', 'some link to google.com'])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 1)

    def test_can_deepcopy(self):
        transformer = preprocess.UrlRemover()
        copied = copy.deepcopy(transformer)
        self.assertEqual(copied.urlfinder, transformer.urlfinder)

    def test_can_pickle(self):
        transformer = preprocess.UrlRemover()
        loaded = pickle.loads(pickle.dumps(transformer))
        self.assertEqual(loaded.urlfinder, transformer.urlfinder)


@patch(SF_LIST, new=Mock(return_value=SERVER_FILES))
@patch(SF_DOWNLOAD, download_patch)
class TokenNormalizerTests(unittest.TestCase):
    def setUp(self):
        self.stemmer = nltk.PorterStemmer().stem
        self.corpus = Corpus.from_file('deerwester')

    def test_str(self):
        stemmer = preprocess.PorterStemmer()
        self.assertEqual('Porter Stemmer', str(stemmer))

    def test_call_porter(self):
        pp = preprocess.PorterStemmer()
        self.assertFalse(self.corpus.has_tokens())
        corpus = pp(self.corpus)
        self.assertTrue(corpus.has_tokens())
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_call_snowball(self):
        pp = preprocess.SnowballStemmer()
        self.assertFalse(self.corpus.has_tokens())
        corpus = pp(self.corpus)
        self.assertTrue(corpus.has_tokens())
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_call_word_net(self):
        pp = preprocess.WordNetLemmatizer()
        self.assertFalse(self.corpus.has_tokens())
        corpus = pp(self.corpus)
        self.assertTrue(corpus.has_tokens())
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_call_UDPipe(self):
        pp = preprocess.UDPipeLemmatizer(language="lt")
        self.assertFalse(self.corpus.has_tokens())
        corpus = pp(self.corpus)
        self.assertTrue(corpus.has_tokens())
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_call_lemmagen(self):
        pp = preprocess.LemmagenLemmatizer()
        self.assertFalse(self.corpus.has_tokens())
        corpus = pp(self.corpus)
        self.assertTrue(corpus.has_tokens())
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_function(self):
        stemmer = preprocess.BaseNormalizer()
        stemmer.normalizer = lambda x: x[:-1]
        self.assertEqual(stemmer._preprocess('token'), 'toke')

    def test_snowball(self):
        stemmer = preprocess.SnowballStemmer('fr')
        token = 'voudrais'
        self.assertEqual(
            stemmer._preprocess(token),
            nltk.SnowballStemmer(language='french').stem(token))

    def test_snowball_all_langs(self):
        for language in preprocess.SnowballStemmer.supported_languages:
            normalizer = preprocess.SnowballStemmer(language)
            tokens = normalizer(self.corpus).tokens
            self.assertEqual(len(self.corpus), len(tokens))
            self.assertTrue(all(tokens))

    def test_udpipe(self):
        """Test udpipe token lemmatization"""
        normalizer = preprocess.UDPipeLemmatizer("lt")
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = "esu"
        corpus = normalizer(self.corpus)
        self.assertListEqual(list(corpus.tokens[0]), ["būti"])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_udpipe_doc(self):
        """Test udpipe lemmatization with its own tokenization"""
        normalizer = preprocess.UDPipeLemmatizer("lt", True)
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = "Ant kalno dega namas"
        corpus = normalizer(self.corpus)
        self.assertListEqual(list(corpus.tokens[0]), ["ant", "kalno", "degas", "namas"])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 1)

    def test_udpipe_pickle(self):
        normalizer = preprocess.UDPipeLemmatizer("lt", True)
        # udpipe store model after first call - model is not picklable
        normalizer(self.corpus)
        loaded = pickle.loads(pickle.dumps(normalizer))
        self.assertEqual(normalizer._UDPipeLemmatizer__language,
                         loaded._UDPipeLemmatizer__language)
        self.assertEqual(normalizer._UDPipeLemmatizer__use_tokenizer,
                         loaded._UDPipeLemmatizer__use_tokenizer)
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = "Ant kalno dega namas"
        self.assertEqual(
            list(loaded(self.corpus).tokens[0]), ["ant", "kalno", "degas", "namas"]
        )

    def test_udpipe_deepcopy(self):
        normalizer = preprocess.UDPipeLemmatizer("lt", True)
        copied = copy.deepcopy(normalizer)
        self.assertEqual(normalizer._UDPipeLemmatizer__language,
                         copied._UDPipeLemmatizer__language)
        self.assertEqual(normalizer._UDPipeLemmatizer__use_tokenizer,
                         copied._UDPipeLemmatizer__use_tokenizer)
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = "Ant kalno dega namas"
        self.assertEqual(
            list(copied(self.corpus).tokens[0]), ["ant", "kalno", "degas", "namas"]
        )

    def test_lemmagen(self):
        normalizer = preprocess.LemmagenLemmatizer("sl")
        sentence = "Gori na gori hiša gori"
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = sentence
        self.assertEqual(
            [Lemmatizer("sl").lemmatize(t) for t in sentence.split()],
            normalizer(self.corpus).tokens[0],
        )

    def test_lemmagen_all_langs(self):
        for language in preprocess.LemmagenLemmatizer.supported_languages:
            normalizer = preprocess.LemmagenLemmatizer(language)
            tokens = normalizer(self.corpus).tokens
            self.assertEqual(len(self.corpus), len(tokens))
            self.assertTrue(all(tokens))

    def test_normalizers_picklable(self):
        """ Normalizers must be picklable, tests if it is true"""
        for nm in set(preprocess.normalize.__all__) - {"BaseNormalizer"}:
            normalizer = getattr(preprocess.normalize, nm)
            normalizer = (
                normalizer(language="lt")
                if normalizer is preprocess.UDPipeLemmatizer
                else normalizer()
            )
            normalizer(self.corpus)
            loaded = pickle.loads(pickle.dumps(normalizer))
            loaded(self.corpus)

    def test_cache(self):
        normalizer = preprocess.UDPipeLemmatizer("lt")
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = "esu"
        normalizer(self.corpus)
        self.assertEqual(normalizer._normalization_cache["esu"], "būti")
        self.assertEqual(40, len(normalizer._normalization_cache))

        # cache should not be pickled
        loaded_normalizer = pickle.loads(pickle.dumps(normalizer))
        self.assertEqual(0, len(loaded_normalizer._normalization_cache))


class TokenNormalizerNotPatched(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')

    @unittest.skip("Slow tests")
    def test_udpipe_all_langs(self):
        for _, language in UDPipeModels().supported_languages:
            normalizer = preprocess.UDPipeLemmatizer(language)
            tokens = normalizer(self.corpus).tokens
            self.assertEqual(len(self.corpus), len(tokens))
            self.assertTrue(all(tokens))


@patch(SF_LIST, return_value=SERVER_FILES)
class UDPipeModelsTests(unittest.TestCase):
    def test_label_transform(self, _):
        """Test helper functions for label transformation"""
        fun = UDPipeModels()._UDPipeModels__file_to_language
        res = fun("slovenian-sst-ud-2.0-170801.udpipe")
        self.assertTupleEqual(res, ("Slovenian (sst)", "sl_sst"))
        res = fun("norwegian_bokmaal-sst-ud-2.0-170801.udpipe")
        self.assertTupleEqual(res, ("Norwegian Bokmål (sst)", "nb_sst"))

    @patch(SF_DOWNLOAD, download_patch)
    def test_udpipe_model(self, _):
        """Test udpipe models loading from server"""
        models = UDPipeModels()
        self.assertIn(('Lithuanian', 'lt'), models.supported_languages)
        self.assertEqual(7, len(models.supported_languages))

        local_file = os.path.join(models.local_data, "lithuanian-ud-2.0-170801.udpipe")
        model = models["lt"]
        self.assertEqual(model, local_file)
        self.assertTrue(os.path.isfile(local_file))

    @patch(SF_DOWNLOAD, download_patch)
    def test_udpipe_local_models(self, sf_mock):
        """Test if UDPipe works offline and uses local models"""
        models = UDPipeModels()
        [models.localfiles.remove(f[0]) for f in models.localfiles.listfiles()]
        # use Uyghur, it is the smallest model, we can have it in the repository
        _ = models["lt"]
        sf_mock.side_effect = ConnectionError()
        exp = {"lt": ('Lithuanian', 'lithuanian-ud-2.0-170801.udpipe')}
        self.assertDictEqual(exp, models.model_files)
        self.assertListEqual([('Lithuanian', 'lt')], models.supported_languages)

    def test_udpipe_offline(self, sf_mock):
        """Test if UDPipe works offline"""
        self.assertTrue(UDPipeModels().online)
        sf_mock.side_effect = ConnectionError()
        self.assertFalse(UDPipeModels().online)

    def test_language_to_iso(self, _):
        models = UDPipeModels()
        self.assertEqual("en", models.language_to_iso("English"))
        self.assertEqual("en_lines", models.language_to_iso("English (lines)"))


class FilteringTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')
        self.regexp = preprocess.RegexpFilter('foo')

    def test_str(self):
        self.assertEqual('Regexp', str(self.regexp))

    def test_preprocess(self):
        class DigitsFilter(preprocess.BaseTokenFilter):
            def _check(self, token):
                return not token.isdigit()

        df = DigitsFilter()
        filtered = list(itertools.compress([], df._preprocess([])))
        self.assertEqual(filtered, [])
        filtered = list(itertools.compress(['a', '1'],
                                           df._preprocess(['a', '1'])))
        self.assertEqual(filtered, ['a'])

    def test_stopwords(self):
        f = preprocess.StopwordsFilter("en")
        self.assertFalse(f._check('a'))
        self.assertTrue(f._check('filter'))
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = 'a snake is in a house'
        corpus = f(self.corpus)
        self.assertListEqual(["snake", "house"], corpus.tokens[0])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_stopwords_slovene(self):
        f = preprocess.StopwordsFilter("sl")
        self.assertFalse(f._check('in'))
        self.assertTrue(f._check('abeceda'))
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = 'kača je v hiši'
        corpus = f(self.corpus)
        self.assertListEqual(["kača", "hiši"], corpus.tokens[0])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_supported_languages(self):
        langs = preprocess.StopwordsFilter.supported_languages
        self.assertIsInstance(langs, set)
        # just testing few of most important languages since I want for test to be
        # resistant for any potentially newly introduced languages by NLTK
        self.assertIn("en", langs)
        self.assertIn("sl", langs)
        self.assertIn("fr", langs)
        self.assertIn("sv", langs)
        self.assertIn("fi", langs)
        self.assertIn("de", langs)
        self.assertNotIn(None, langs)

    def test_lang_to_iso(self):
        self.assertEqual("en", StopwordsFilter.lang_to_iso("English"))
        self.assertEqual("sl", StopwordsFilter.lang_to_iso("Slovene"))

    def test_custom_list(self):
        f = tempfile.NamedTemporaryFile("w", delete=False,
                                        encoding='utf-8-sig')
        # test if BOM removed
        f.write('human\n')
        f.write('user\n')
        f.flush()
        f.close()
        stopwords = preprocess.StopwordsFilter(None, f.name)
        self.assertIn('human', stopwords._lexicon)
        self.assertIn('user', stopwords._lexicon)
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = 'human user baz'
        processed = stopwords(self.corpus)
        self.assertEqual(["baz"], processed.tokens[0])
        f.close()
        os.unlink(f.name)

    def test_langauge_missing(self):
        """
        When NLTK adds language that is not in dict of languages module
        should raise an error. If this test fall add missing langauge to LANG2ISO
        """
        for file in os.listdir(stopwords._get_root()):
            if file.islower():
                self.assertIsNotNone(
                    StopwordsFilter.lang_to_iso(file.title()),
                    f"Missing language {file.title()} in StopwordsFilter.LANG2ISO"
                )

    def test_lexicon(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(b'filter\n')
        f.flush()
        f.close()
        lexicon = preprocess.LexiconFilter(f.name)
        self.assertFalse(lexicon._check('false'))
        self.assertTrue(lexicon._check('filter'))
        f.close()
        os.unlink(f.name)

    def test_keep_n(self):
        ff = preprocess.MostFrequentTokensFilter(keep_n=5)
        processed = ff(self.corpus)
        self.assertEqual(len(set(itertools.chain(*processed.tokens))), 5)
        self.assertEqual(len(processed.used_preprocessor.preprocessors), 2)

    def test_min_df(self):
        ff = preprocess.FrequencyFilter(min_df=.5)
        processed = ff(self.corpus)
        size = len(processed.documents)
        self.assertFrequencyRange(processed, size * .5, size)
        self.assertEqual(len(processed.used_preprocessor.preprocessors), 2)

        ff = preprocess.FrequencyFilter(min_df=2)
        processed = ff(self.corpus)
        size = len(processed.documents)
        self.assertFrequencyRange(processed, 2, size)
        self.assertEqual(len(processed.used_preprocessor.preprocessors), 2)

    def test_max_df(self):
        ff = preprocess.FrequencyFilter(max_df=.3)
        size = len(self.corpus.documents)

        corpus = ff(self.corpus)
        self.assertFrequencyRange(corpus, 1, size * .3)
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

        ff = preprocess.FrequencyFilter(max_df=2)
        corpus = ff(self.corpus)
        self.assertFrequencyRange(corpus, 1, 2)
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

        ff = preprocess.FrequencyFilter(min_df=5, max_df=5)
        corpus = ff(self.corpus)
        self.assertFrequencyRange(corpus, 5, size)
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def assertFrequencyRange(self, corpus, min_fr, max_fr):
        dictionary = corpora.Dictionary(corpus.tokens)
        self.assertTrue(all(min_fr <= fr <= max_fr
                            for fr in dictionary.dfs.values()))

    def test_word_list(self):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(b'hello\nworld\n')
        f.flush()
        f.close()
        lexicon = preprocess.LexiconFilter(f.name)
        self.assertIn('hello', lexicon._lexicon)
        self.assertIn('world', lexicon._lexicon)
        f.close()
        os.unlink(f.name)

    def test_filter_numbers(self):
        f = preprocess.NumbersFilter()
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = '1 2foo bar3 baz'
        corpus = f(self.corpus)
        self.assertEqual(["2foo", "bar3", "baz"], corpus.tokens[0])

    def test_filter_tokens_with_numbers(self):
        f = preprocess.WithNumbersFilter()
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = '1 2foo bar3 baz'
        corpus = f(self.corpus)
        self.assertEqual(["baz"], corpus.tokens[0])

    def test_regex_filter(self):
        self.assertFalse(preprocess.RegexpFilter.validate_regexp('?'))
        self.assertTrue(preprocess.RegexpFilter.validate_regexp('\?'))

        reg_filter = preprocess.RegexpFilter(r'.')
        filtered = reg_filter(self.corpus)
        self.assertEqual(0, len(filtered.tokens[0]))
        self.assertEqual(len(filtered.used_preprocessor.preprocessors), 2)

        reg_filter = preprocess.RegexpFilter('foo')
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = 'foo bar'
        filtered = reg_filter(self.corpus)
        self.assertEqual(filtered.tokens[0], ['bar'])
        self.assertEqual(len(filtered.used_preprocessor.preprocessors), 2)

        reg_filter = preprocess.RegexpFilter('^http')
        corpus = BASE_TOKENIZER(self.corpus)
        corpus._tokens[0] = ['https', 'http', ' http']
        filtered = reg_filter(corpus)
        self.assertEqual(filtered.tokens[0], [' http'])
        self.assertEqual(len(filtered.used_preprocessor.preprocessors), 2)

    def test_pos_filter(self):
        pos_filter = preprocess.PosTagFilter("NN")
        pp_list = [preprocess.WordPunctTokenizer(),
                   tag.AveragedPerceptronTagger()]
        corpus = self.corpus
        for pp in pp_list:
            corpus = pp(corpus)
        filtered = pos_filter(corpus)
        self.assertTrue(len(filtered.pos_tags))
        self.assertEqual(len(filtered.pos_tags[0]), 5)
        self.assertEqual(len(filtered.tokens[0]), 5)

    def test_can_deepcopy(self):
        copied = copy.deepcopy(self.regexp)
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = 'foo bar'
        self.assertEqual(copied(self.corpus).tokens[0], ['bar'])

    def test_can_pickle(self):
        loaded = pickle.loads(pickle.dumps(self.regexp))
        with self.corpus.unlocked():
            self.corpus.metas[0, 0] = 'foo bar'
        self.assertEqual(loaded(self.corpus).tokens[0], ['bar'])


class TokenizerTests(unittest.TestCase):
    def test_tokenize(self):
        class DashTokenizer(preprocess.BaseTokenizer):
            def _preprocess(self, string):
                return string.split('-')

        tokenizer = DashTokenizer()
        self.assertEqual(list(tokenizer._preprocess('dashed-sentence')),
                         ['dashed', 'sentence'])

    def test_call(self):
        class SpaceTokenizer(preprocess.BaseTokenizer):
            def _preprocess(self, string):
                return string.split(' ')

        corpus = Corpus.from_file("deerwester")
        tokens = ['Human', 'machine', 'interface', 'for', 'lab', 'abc',
                  'computer', 'applications']
        corpus = SpaceTokenizer()(corpus)
        self.assertEqual(corpus.tokens[0], tokens)
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 1)

    def test_call_with_bad_input(self):
        pattern = '\w+'
        tokenizer = preprocess.RegexpTokenizer(pattern=pattern)
        tokenizer.tokenizer = tokenizer.tokenizer_cls(pattern)
        self.assertRaises(TypeError, tokenizer._preprocess, 1)
        self.assertRaises(TypeError, tokenizer._preprocess, ['1', 2])

    def test_valid_regexp(self):
        self.assertTrue(preprocess.RegexpTokenizer.validate_regexp('\w+'))

    def test_invalid_regex(self):
        for expr in ['\\', '[', ')?']:
            self.assertFalse(preprocess.RegexpTokenizer.validate_regexp(expr))

    def test_str(self):
        tokenizer = preprocess.RegexpTokenizer(pattern=r'\S+')
        self.assertEqual('Regexp', str(tokenizer))

    def test_skip_empty_strings(self):
        pattern = r'[^h ]*'
        tokenizer = preprocess.RegexpTokenizer(pattern=pattern)
        tokenizer.tokenizer = tokenizer.tokenizer_cls(pattern)
        tokens = tokenizer._preprocess('whatever')
        self.assertNotIn('', tokens)

    def test_can_deepcopy(self):
        tokenizer = preprocess.RegexpTokenizer(pattern=r'\w')
        copied = copy.deepcopy(tokenizer)
        corpus = Corpus.from_file('deerwester')
        self.assertTrue(all(tokenizer(corpus).tokens == copied(corpus).tokens))

    def test_can_pickle(self):
        tokenizer = preprocess.RegexpTokenizer(pattern=r'\w')
        pickle.loads(pickle.dumps(tokenizer))

    def test_reset_pos_tags(self):
        corpus = Corpus.from_file('deerwester')
        tagger = tag.AveragedPerceptronTagger()
        tagged_corpus = tagger(corpus)
        self.assertTrue(len(tagged_corpus.pos_tags))
        tokenizer = preprocess.RegexpTokenizer(pattern=r'\w')
        tokenized_corpus = tokenizer(corpus)
        self.assertFalse(tokenized_corpus.pos_tags)


class NGramsTests(unittest.TestCase):
    def setUp(self):
        self.pp = preprocess.NGrams((2, 3))
        self.corpus = Corpus.from_file('deerwester')

    def test_call(self):
        corpus = self.pp(self.corpus)
        self.assertEqual(next(corpus.ngrams)[0],
                         " ".join(corpus.tokens[0][:2]))
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_retain_old_data(self):
        corpus = self.pp(self.corpus)
        self.assertIsNot(corpus, self.corpus)

    def test_str(self):
        self.assertEqual('N-grams Range', str(self.pp))

    def test_can_deepcopy(self):
        copied = copy.deepcopy(self.pp)
        self.assertEqual(copied._NGrams__range, self.pp._NGrams__range)

    def test_can_pickle(self):
        loaded = pickle.loads(pickle.dumps(self.pp))
        self.assertEqual(loaded._NGrams__range, self.pp._NGrams__range)


if __name__ == "__main__":
    unittest.main()
