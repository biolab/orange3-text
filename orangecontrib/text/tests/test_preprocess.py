import tempfile
import unittest
import os.path

import itertools
import nltk
from unittest import mock
from gensim import corpora
from requests.exceptions import ConnectionError
import numpy as np

from orangecontrib.text import preprocess
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor
from orangecontrib.text.preprocess.normalize import file_to_language, \
    file_to_name, language_to_name, UDPipeModels


def counted(f):
    def wrapped(*args, **kwargs):
        wrapped.calls += 1
        return f(*args, **kwargs)
    wrapped.calls = 0
    return wrapped


class PreprocessTests(unittest.TestCase):
    sentence = "Human machine interface for lab abc computer applications"

    def setUp(self):
        self.corpus = Corpus.from_file('deerwester')

    def test_string_processor(self):
        p = Preprocessor(transformers=preprocess.LowercaseTransformer())
        tokens = p(self.corpus).tokens
        p2 = Preprocessor(transformers=[])
        tokens2 = p2(self.corpus).tokens

        np.testing.assert_equal(tokens,
                                [[t.lower() for t in doc] for doc in tokens2])

        self.assertRaises(TypeError, Preprocessor, string_transformers=1)

    def test_tokenizer(self):
        class SpaceTokenizer(preprocess.BaseTokenizer):
            @classmethod
            def tokenize(cls, string):
                return string.split()
        p = Preprocessor(tokenizer=SpaceTokenizer())

        np.testing.assert_equal(p(self.corpus).tokens,
                         np.array([sent.split() for sent in self.corpus.documents]))

    def test_token_normalizer(self):
        class CapTokenNormalizer(preprocess.BaseNormalizer):
            @classmethod
            def normalize(cls, token):
                return token.capitalize()
        p = Preprocessor(normalizer=CapTokenNormalizer())
        tokens = p(self.corpus).tokens
        p2 = Preprocessor(normalizer=None)
        tokens2 = p2(self.corpus).tokens

        np.testing.assert_equal(
            tokens, [[t.capitalize() for t in doc] for doc in tokens2])

    def test_token_filter(self):
        class SpaceTokenizer(preprocess.BaseTokenizer):
            @classmethod
            def tokenize(cls, string):
                return string.split()

        class LengthFilter(preprocess.BaseTokenFilter):
            @classmethod
            def check(cls, token):
                return len(token) < 4

        p = Preprocessor(tokenizer=SpaceTokenizer(), filters=LengthFilter())
        np.testing.assert_equal(p(self.corpus).tokens,
                         np.array([[token for token in doc.split() if len(token) < 4]
                                   for doc in self.corpus.documents]))

    def test_inplace(self):
        p = Preprocessor(tokenizer=preprocess.RegexpTokenizer('\w'))
        corpus = p(self.corpus, inplace=True)
        self.assertIs(corpus, self.corpus)

        corpus = p(self.corpus, inplace=False)
        self.assertIsNot(corpus, self.corpus)
        self.assertEqual(corpus, self.corpus)

        p = Preprocessor(tokenizer=preprocess.RegexpTokenizer('\w+'))
        corpus = p(self.corpus, inplace=False)
        self.assertIsNot(corpus, self.corpus)
        self.assertNotEqual(corpus, self.corpus)


class TransformationTests(unittest.TestCase):
    def test_call(self):

        class ReverseStringTransformer(preprocess.BaseTransformer):
            name = "Reverse"

            def transform(self, string):
                return string[::-1]

        transformer = ReverseStringTransformer()

        self.assertEqual(transformer('abracadabra'), 'arbadacarba')
        self.assertEqual(transformer(['abra', 'cadabra']), ['arba', 'arbadac'])

        self.assertRaises(TypeError, transformer, 1)

    def test_str(self):
        class ReverseStringTransformer(preprocess.BaseTransformer):
            name = 'reverse'

            def transform(self, string):
                return string[::-1]

        transformer = ReverseStringTransformer()

        self.assertIn('reverse', str(transformer))

    def test_lowercase(self):
        transformer = preprocess.LowercaseTransformer()
        self.assertEqual(transformer.transform('Abra'), 'abra')
        self.assertEqual(transformer.transform('\u00C0bra'), '\u00E0bra')

    def test_strip_accents(self):
        transformer = preprocess.StripAccentsTransformer()
        self.assertEqual(transformer.transform('Abra'), 'Abra')
        self.assertEqual(transformer.transform('\u00C0bra'), 'Abra')

    def test_html(self):
        transformer = preprocess.HtmlTransformer()
        self.assertEqual(transformer('<p>abra<b>cadabra</b><p>'), 'abracadabra')

    def test_url_remover(self):
        url_remover = preprocess.UrlRemover()
        self.assertEqual(url_remover.transform('some link to https://google.com/'), 'some link to ')
        self.assertEqual(url_remover.transform('some link to google.com'), 'some link to google.com')


class TokenNormalizerTests(unittest.TestCase):

    def setUp(self):
        self.stemmer = nltk.PorterStemmer().stem

    def test_str(self):
        stemmer = preprocess.PorterStemmer()
        self.assertIn('porter', str(stemmer).lower())

        stemmer = preprocess.SnowballStemmer('french')
        self.assertIn('french', str(stemmer).lower())

    def test_call(self):
        word = "Testing"
        tokens = ["Testing", "tokenized", "Sentence"]
        stemmer = preprocess.PorterStemmer()
        self.assertEqual(stemmer(word), self.stemmer(word))
        self.assertEqual(stemmer(tokens),
                         [self.stemmer(token) for token in tokens])

    def test_function(self):
        stemmer = preprocess.BaseNormalizer()
        stemmer.normalizer = lambda x: x[:-1]
        self.assertEqual(stemmer.normalize('token'), 'toke')

    def test_snowball(self):
        stemmer = preprocess.SnowballStemmer()
        stemmer.language = 'french'
        token = 'voudrais'
        self.assertEqual(stemmer(token), nltk.SnowballStemmer(language='french').stem(token))

    def test_udpipe(self):
        """Test udpipe token lemmatization"""
        normalizer = preprocess.UDPipeLemmatizer()
        normalizer.language = 'Slovenian'
        self.assertEqual(normalizer('sem'), 'biti')

    def test_udpipe_doc(self):
        """Test udpipe lemmatization with its own tokenization """
        normalizer = preprocess.UDPipeLemmatizer()
        normalizer.language = 'Slovenian'
        normalizer.use_tokenizer = True
        self.assertListEqual(normalizer.normalize_doc('Gori na gori hiša gori'),
                             ['gora', 'na', 'gora', 'hiša', 'goreti'])

    def test_porter_with_bad_input(self):
        stemmer = preprocess.PorterStemmer()
        self.assertRaises(TypeError, stemmer, 10)

    def test_lookup_normalize(self):
        dln = preprocess.DictionaryLookupNormalizer(dictionary={'aka': 'also known as'})
        self.assertEqual(dln.normalize('aka'), 'also known as')


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

    def test_str(self):
        f = preprocess.StopwordsFilter('french')
        self.assertIn('french', str(f).lower())

        f = preprocess.FrequencyFilter(keep_n=None)
        self.assertNotIn('none', str(f).lower())
        f.max_df = .5
        self.assertIn('0.5', str(f))
        f.max_df = .2
        self.assertIn('0.2', str(f))

        f = preprocess.LexiconFilter()
        self.assertIn('lexicon', str(f).lower())

    def test_call(self):
        class DigitsFilter(preprocess.BaseTokenFilter):
            def check(self, token):
                return not token.isdigit()

        df = DigitsFilter()

        self.assertEqual(df([]), [])
        self.assertEqual(df(['a', '1']), ['a'])
        self.assertEqual(df([['a', '1']]), [['a']])

    def test_stopwords(self):
        filter = preprocess.StopwordsFilter('english')

        self.assertFalse(filter.check('a'))
        self.assertTrue(filter.check('filter'))

    def test_lexicon(self):
        filter = preprocess.LexiconFilter(['filter'])
        self.assertFalse(filter.check('false'))
        self.assertTrue(filter.check('filter'))

    def test_keep_n(self):
        ff = preprocess.FrequencyFilter(keep_n=5)
        p = Preprocessor(tokenizer=preprocess.RegexpTokenizer(r'\w+'),
                         filters=[ff])
        processed = p(self.corpus)
        self.assertEqual(len(set(itertools.chain(*processed.tokens))), 5)

    def test_min_df(self):
        ff = preprocess.FrequencyFilter(min_df=.5)
        p = Preprocessor(tokenizer=preprocess.RegexpTokenizer(r'\w+'),
                         filters=[ff])
        processed = p(self.corpus)
        size = len(processed.documents)
        self.assertFrequencyRange(processed, size * .5, size)

        ff.min_df = 2
        processed = p(self.corpus)
        size = len(processed.documents)
        self.assertFrequencyRange(processed, 2, size)

    def test_max_df(self):
        ff = preprocess.FrequencyFilter(max_df=.3)
        p = Preprocessor(tokenizer=preprocess.RegexpTokenizer(r'\w+'),
                         filters=[ff])
        size = len(self.corpus.documents)

        corpus = p(self.corpus)
        self.assertFrequencyRange(corpus, 1, size * .3)

        ff.max_df = 2
        corpus = p(self.corpus)
        self.assertFrequencyRange(corpus, 1, 2)

    def assertFrequencyRange(self, corpus, min_fr, max_fr):
        dictionary = corpora.Dictionary(corpus.tokens)
        self.assertTrue(all(min_fr <= fr <= max_fr
                            for fr in dictionary.dfs.values()))

    def test_word_list(self):
        lexicon = preprocess.LexiconFilter()
        f = tempfile.NamedTemporaryFile()
        f.write(b'hello\nworld\n')
        f.flush()
        lexicon.from_file(f.name)
        self.assertIn('hello', lexicon.lexicon)
        self.assertIn('world', lexicon.lexicon)

    def test_regex_filter(self):
        reg_filter = preprocess.RegexpFilter(r'.')
        filtered = reg_filter(self.corpus.tokens[0])
        self.assertFalse(filtered)

        reg_filter.pattern = 'foo'
        self.assertCountEqual(reg_filter(['foo', 'bar']), ['bar'])

        reg_filter.pattern = '^http'
        self.assertCountEqual(reg_filter(['https', 'http', ' http']), [' http'])

        self.assertFalse(preprocess.RegexpFilter.validate_regexp('?'))
        self.assertTrue(preprocess.RegexpFilter.validate_regexp('\?'))


class TokenizerTests(unittest.TestCase):
    def test_call(self):
        class DashTokenizer(preprocess.BaseTokenizer):
            @classmethod
            def tokenize(cls, string):
                return string.split('-')

        tokenizer = DashTokenizer()
        self.assertEqual(list(tokenizer('dashed-sentence')), ['dashed', 'sentence'])
        self.assertEqual(list(tokenizer(['1-2-3', '-'])), [['1', '2', '3'], ['', '']])

        self.assertRaises(TypeError, tokenizer, 1)

    def test_tokenizer_instance(self):
        class WhitespaceTokenizer(preprocess.BaseTokenizer):
            tokenizer = nltk.WhitespaceTokenizer()
            name = 'whitespace'

        tokenizer = WhitespaceTokenizer()

        sent = "Test \t tokenizer"
        self.assertEqual(tokenizer.tokenize(sent),
                         nltk.WhitespaceTokenizer().tokenize(sent))

    def test_call_with_bad_input(self):
        tokenizer = preprocess.RegexpTokenizer(pattern='\w+')
        self.assertRaises(TypeError, tokenizer.tokenize, 1)
        self.assertRaises(TypeError, tokenizer.tokenize, ['1', 2])

    def test_valid_regexp(self):
        self.assertTrue(preprocess.RegexpTokenizer.validate_regexp('\w+'))

    def test_invalid_regex(self):
        for expr in ['\\', '[', ')?']:
            self.assertFalse(preprocess.RegexpTokenizer.validate_regexp(expr))

    def test_on_change(self):
        tokenizer = preprocess.RegexpTokenizer(pattern=r'\w+')
        tokenizer.on_change = counted(tokenizer.on_change)
        tokenizer.pattern = r'\S+'
        self.assertEqual(tokenizer.on_change.calls, 1)
        self.assertEqual(tokenizer.pattern, r'\S+')

    def test_str(self):
        tokenizer = preprocess.RegexpTokenizer(pattern=r'\S+')
        self.assertIn('\S+', str(tokenizer))

    def test_skip_empty_strings(self):
        tokenizer = preprocess.RegexpTokenizer(pattern=r'[^h ]*')
        tokens = tokenizer('whatever')
        self.assertNotIn('', tokens)
