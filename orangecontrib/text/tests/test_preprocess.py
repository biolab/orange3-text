import tempfile
import unittest

import itertools
import nltk
from gensim import corpora

from orangecontrib.text import preprocess
from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import Preprocessor


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
        class StripStringTransformer(preprocess.BaseTransformer):
            @classmethod
            def transform(cls, string):
                return string[:-1]
        p = Preprocessor(transformers=StripStringTransformer())

        self.assertEqual(p(self.corpus).tokens, [[doc[:-1]] for doc in self.corpus.documents])

        p = Preprocessor(transformers=[StripStringTransformer(),
                                       preprocess.LowercaseTransformer()])

        self.assertEqual(p(self.corpus).tokens, [[doc[:-1].lower()] for doc in self.corpus.documents])

        self.assertRaises(TypeError, Preprocessor, string_transformers=1)

    def test_tokenizer(self):
        class SpaceTokenizer(preprocess.BaseTokenizer):
            @classmethod
            def tokenize(cls, string):
                return string.split()
        p = Preprocessor(tokenizer=SpaceTokenizer())

        self.assertEqual(p(self.corpus).tokens,
                         [sent.split() for sent in self.corpus.documents])

    def test_token_normalizer(self):
        class CapTokenNormalizer(preprocess.BaseNormalizer):
            @classmethod
            def normalize(cls, token):
                return token.capitalize()
        p = Preprocessor(normalizer=CapTokenNormalizer())

        self.assertEqual(p(self.corpus).tokens,
                         [[sent.capitalize()] for sent in self.corpus.documents])

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
        self.assertEqual(p(self.corpus).tokens,
                         [[token for token in doc.split() if len(token) < 4]
                          for doc in self.corpus.documents])


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

    def test_porter_with_bad_input(self):
        stemmer = preprocess.PorterStemmer()
        self.assertRaises(TypeError, stemmer, 10)

    def test_lookup_normalize(self):
        dln = preprocess.DictionaryLookupNormalizer(dictionary={'aka': 'also known as'})
        self.assertEqual(dln.normalize('aka'), 'also known as')

    def test_on_change(self):
        snowball = preprocess.SnowballStemmer()
        snowball.on_change = counted(snowball.on_change)
        snowball.language = 'french'
        self.assertEqual(snowball.on_change.calls, 1)

        dictionary = preprocess.DictionaryLookupNormalizer({})
        dictionary.on_change = counted(dictionary.on_change)
        dictionary.dictionary = {'a': 'b'}
        self.assertEqual(snowball.on_change.calls, 1)


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

    def test_on_change(self):
        df_filter = preprocess.FrequencyFilter()
        df_filter.on_change = counted(df_filter.on_change)

        df_filter.min_df = .2
        self.assertEqual(df_filter.on_change.calls, 1)

        df_filter.max_df = .5
        self.assertEqual(df_filter.on_change.calls, 2)

        df_filter.keep_n = 100
        self.assertEqual(df_filter.on_change.calls, 3)
        self.assertEqual(df_filter.keep_n, 100)

        stopwords = preprocess.StopwordsFilter()
        stopwords.on_change = counted(stopwords.on_change)
        stopwords.language = 'french'
        self.assertEqual(stopwords.on_change.calls, 1)

        lexicon = preprocess.LexiconFilter()
        lexicon.on_change = counted(stopwords.on_change)
        lexicon.lexicon = ['a', 'b', 'c']
        self.assertEqual(lexicon.on_change.calls, 1)

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
