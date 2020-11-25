import pickle
import tempfile
import unittest
import os.path
import copy
import itertools
from unittest import mock

import nltk
from gensim import corpora
from requests.exceptions import ConnectionError
import numpy as np

from orangecontrib.text import preprocess, tag
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import BASE_TOKENIZER, PreprocessorList
from orangecontrib.text.preprocess.normalize import file_to_language, \
    file_to_name, language_to_name, UDPipeModels


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
        self.assertEqual(corpus1, corpus2)

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
        np.testing.assert_equal(tokens,
                                [[t.lower() for t in doc] for doc in tokens2])

    def test_tokenizer(self):
        class SpaceTokenizer(preprocess.BaseTokenizer):
            @classmethod
            def _preprocess(cls, string):
                return string.split()

        p = SpaceTokenizer()
        array = np.array([sent.split() for sent in self.corpus.documents])
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
            tokens, [[t.capitalize() for t in doc] for doc in tokens2])

    def test_token_filter(self):
        class LengthFilter(preprocess.BaseTokenFilter):
            def _check(self, token):
                return len(token) < 4

        p = LengthFilter()
        tokens = np.array([[token for token in doc.split() if len(token) < 4]
                           for doc in self.corpus.documents])
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
        pp = preprocess.UDPipeLemmatizer()
        self.assertFalse(self.corpus.has_tokens())
        corpus = pp(self.corpus)
        self.assertTrue(corpus.has_tokens())
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_function(self):
        stemmer = preprocess.BaseNormalizer()
        stemmer.normalizer = lambda x: x[:-1]
        self.assertEqual(stemmer._preprocess('token'), 'toke')

    def test_snowball(self):
        stemmer = preprocess.SnowballStemmer('french')
        token = 'voudrais'
        self.assertEqual(
            stemmer._preprocess(token),
            nltk.SnowballStemmer(language='french').stem(token))

    def test_udpipe(self):
        """Test udpipe token lemmatization"""
        normalizer = preprocess.UDPipeLemmatizer('Slovenian')
        self.corpus.metas[0, 0] = 'sem'
        corpus = normalizer(self.corpus)
        self.assertListEqual(list(corpus.tokens[0]), ['biti'])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_udpipe_doc(self):
        """Test udpipe lemmatization with its own tokenization """
        normalizer = preprocess.UDPipeLemmatizer('Slovenian', True)
        self.corpus.metas[0, 0] = 'Gori na gori hiša gori'
        corpus = normalizer(self.corpus)
        self.assertListEqual(list(corpus.tokens[0]),
                             ['gora', 'na', 'gora', 'hiša', 'goreti'])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 1)

    def test_udpipe_pickle(self):
        normalizer = preprocess.UDPipeLemmatizer('Slovenian', True)
        loaded = pickle.loads(pickle.dumps(normalizer))
        self.assertEqual(normalizer._UDPipeLemmatizer__language,
                         loaded._UDPipeLemmatizer__language)
        self.assertEqual(normalizer._UDPipeLemmatizer__use_tokenizer,
                         loaded._UDPipeLemmatizer__use_tokenizer)
        self.corpus.metas[0, 0] = 'Gori na gori hiša gori'
        self.assertEqual(list(loaded(self.corpus).tokens[0]),
                         ['gora', 'na', 'gora', 'hiša', 'goreti'])

    def test_udpipe_deepcopy(self):
        normalizer = preprocess.UDPipeLemmatizer('Slovenian', True)
        copied = copy.deepcopy(normalizer)
        self.assertEqual(normalizer._UDPipeLemmatizer__language,
                         copied._UDPipeLemmatizer__language)
        self.assertEqual(normalizer._UDPipeLemmatizer__use_tokenizer,
                         copied._UDPipeLemmatizer__use_tokenizer)
        self.corpus.metas[0, 0] = 'Gori na gori hiša gori'
        self.assertEqual(list(copied(self.corpus).tokens[0]),
                         ['gora', 'na', 'gora', 'hiša', 'goreti'])


class UDPipeModelsTests(unittest.TestCase):
    def test_label_transform(self):
        """Test helper functions for label transformation"""
        self.assertEqual(file_to_language('slovenian-sst-ud-2.0-170801.udpipe'),
                         'Slovenian sst')
        self.assertEqual(file_to_name('slovenian-sst-ud-2.0-170801.udpipe'),
                         'sloveniansstud2.0170801.udpipe')
        self.assertEqual(language_to_name('Slovenian sst'), 'sloveniansstud')

    def test_udpipe_model(self):
        """Test udpipe models loading from server"""
        models = UDPipeModels()
        self.assertIn('Slovenian', models.supported_languages)
        self.assertEqual(68, len(models.supported_languages))

        local_file = os.path.join(models.local_data,
                                  'slovenian-ud-2.0-170801.udpipe')
        model = models['Slovenian']
        self.assertEqual(model, local_file)
        self.assertTrue(os.path.isfile(local_file))

    def test_udpipe_local_models(self):
        """Test if UDPipe works offline and uses local models"""
        models = UDPipeModels()
        [models.localfiles.remove(f[0]) for f in models.localfiles.listfiles()]
        _ = models['Slovenian']
        with mock.patch('serverfiles.ServerFiles.listfiles',
                        **{'side_effect': ConnectionError()}):
            self.assertIn('Slovenian', UDPipeModels().supported_languages)
            self.assertEqual(1, len(UDPipeModels().supported_languages))

    def test_udpipe_offline(self):
        """Test if UDPipe works offline"""
        self.assertTrue(UDPipeModels().online)
        with mock.patch('serverfiles.ServerFiles.listfiles',
                        **{'side_effect': ConnectionError()}):
            self.assertFalse(UDPipeModels().online)


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
        self.assertEqual(df._preprocess([]), [])
        self.assertEqual(df._preprocess(['a', '1']), ['a'])

    def test_stopwords(self):
        f = preprocess.StopwordsFilter('english')
        self.assertFalse(f._check('a'))
        self.assertTrue(f._check('filter'))
        self.corpus.metas[0, 0] = 'a snake is in a house'
        corpus = f(self.corpus)
        self.assertListEqual(["snake", "house"], corpus.tokens[0])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_stopwords_slovene(self):
        f = preprocess.StopwordsFilter('slovene')
        self.assertFalse(f._check('in'))
        self.assertTrue(f._check('abeceda'))
        self.corpus.metas[0, 0] = 'kača je v hiši'
        corpus = f(self.corpus)
        self.assertListEqual(["kača", "hiši"], corpus.tokens[0])
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_lexicon(self):
        f = tempfile.NamedTemporaryFile()
        f.write(b'filter\n')
        f.flush()
        lexicon = preprocess.LexiconFilter(f.name)
        self.assertFalse(lexicon._check('false'))
        self.assertTrue(lexicon._check('filter'))

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

    def assertFrequencyRange(self, corpus, min_fr, max_fr):
        dictionary = corpora.Dictionary(corpus.tokens)
        self.assertTrue(all(min_fr <= fr <= max_fr
                            for fr in dictionary.dfs.values()))

    def test_word_list(self):
        f = tempfile.NamedTemporaryFile()
        f.write(b'hello\nworld\n')
        f.flush()
        lexicon = preprocess.LexiconFilter(f.name)
        self.assertIn('hello', lexicon._lexicon)
        self.assertIn('world', lexicon._lexicon)

    def test_regex_filter(self):
        self.assertFalse(preprocess.RegexpFilter.validate_regexp('?'))
        self.assertTrue(preprocess.RegexpFilter.validate_regexp('\?'))

        reg_filter = preprocess.RegexpFilter(r'.')
        corpus = self.corpus
        filtered = reg_filter(corpus)
        self.assertFalse(filtered.tokens[0])
        self.assertEqual(len(filtered.used_preprocessor.preprocessors), 2)

        reg_filter = preprocess.RegexpFilter('foo')
        corpus = self.corpus
        corpus.metas[0, 0] = 'foo bar'
        filtered = reg_filter(corpus)
        self.assertEqual(filtered.tokens[0], ['bar'])
        self.assertEqual(len(filtered.used_preprocessor.preprocessors), 2)

        reg_filter = preprocess.RegexpFilter('^http')
        corpus = BASE_TOKENIZER(self.corpus)
        corpus._tokens[0] = ['https', 'http', ' http']
        filtered = reg_filter(corpus)
        self.assertEqual(filtered.tokens[0], [' http'])
        self.assertEqual(len(filtered.used_preprocessor.preprocessors), 2)

    def test_can_deepcopy(self):
        copied = copy.deepcopy(self.regexp)
        self.corpus.metas[0, 0] = 'foo bar'
        self.assertEqual(copied(self.corpus).tokens[0], ['bar'])

    def test_can_pickle(self):
        loaded = pickle.loads(pickle.dumps(self.regexp))
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
