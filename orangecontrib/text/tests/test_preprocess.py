import unittest

from nltk.tokenize import word_tokenize, RegexpTokenizer, TweetTokenizer

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.preprocess import (Preprocessor, PorterStemmer,
                                           SnowballStemmer, Lemmatizer)


class PreprocessTests(unittest.TestCase):
    TEST_STRING = 'Human machine interface for lab abc computer applications'
    TEST_LIST = None
    TEST_CORPUS = None

    def setUp(self):
        self.TEST_CORPUS = Corpus.from_file('deerwester')
        self.TEST_LIST = [entry.metas[0] for entry in self.TEST_CORPUS]

    def test_tokenize(self):
        correct = [
            ['human', 'machine', 'interface', 'for', 'lab', 'abc',
             'computer', 'applications'],
            ['a', 'survey', 'of', 'user', 'opinion', 'of', 'computer',
             'system', 'response', 'time'],
            ['the', 'eps', 'user', 'interface', 'management', 'system'],
            ['system', 'and', 'human', 'system', 'engineering', 'testing',
             'of', 'eps'],
            ['relation', 'of', 'user', 'perceived', 'response', 'time',
             'to', 'error', 'measurement'],
            ['the', 'generation', 'of', 'random', 'binary', 'unordered',
             'trees'],
            ['the', 'intersection', 'graph', 'of', 'paths', 'in', 'trees'],
            ['graph', 'minors', 'iv', 'widths', 'of', 'trees', 'and',
             'well', 'quasi', 'ordering'],
            ['graph', 'minors', 'a', 'survey']
        ]

        # String.
        p = Preprocessor(lowercase=False)
        self.assertEqual(
            p(self.TEST_STRING),
            [
                [
                    'Human',
                    'machine',
                    'interface',
                    'for',
                    'lab',
                    'abc',
                    'computer',
                    'applications',
                ]
            ]
        )

        # List.
        p = Preprocessor()
        self.assertEqual(
            p(self.TEST_LIST),
            correct
        )

        # Corpus.
        p = Preprocessor()
        self.assertEqual(
            p(self.TEST_CORPUS).tokens,
            correct
        )

    def test_preprocess_sentence_stopwords(self):
        # No stop words.
        p = Preprocessor()
        result = p(self.TEST_STRING)
        correct = [
            [
                'human',
                'machine',
                'interface',
                'for',
                'lab',
                'abc',
                'computer',
                'applications'
            ]
        ]
        self.assertEqual(result, correct)

        # English stop words.
        p = Preprocessor(stop_words='english')
        result = p(self.TEST_STRING)
        correct = [
            [
                'human',
                'machine',
                'interface',
                'lab',
                'abc',
                'computer',
                'applications'
            ]
        ]
        self.assertEqual(result, correct)

        # Custom stop words.
        custom_stop_words = [
            'abc',
            'applications',
            'computer',
            'for',
        ]
        p = Preprocessor(stop_words=custom_stop_words)
        result = p(self.TEST_STRING)
        correct = [
            [
                'human',
                'machine',
                'interface',
                'lab'
            ]
        ]
        self.assertEqual(result, correct)

    def test_preprocess_corpus_int_df(self):
        p = Preprocessor(min_df=2, max_df=4)
        result = p(self.TEST_CORPUS)
        correct = [
            ['human', 'interface', 'computer'],
            ['a', 'survey', 'user', 'computer', 'system', 'response', 'time'],
            ['the', 'eps', 'user', 'interface', 'system'],
            ['system', 'and', 'human', 'system', 'eps'],
            ['user', 'response', 'time'],
            ['the', 'trees'],
            ['the', 'graph', 'trees'],
            ['graph', 'minors', 'trees', 'and'],
            ['graph', 'minors', 'a', 'survey']
        ]
        self.assertEqual(result.tokens, correct)

    def test_preprocess_corpus_float_df(self):
        p = Preprocessor(min_df=0.2, max_df=0.5, lowercase=True)
        result = p(self.TEST_CORPUS)
        correct = [
            ['human', 'interface', 'computer'],
            ['a', 'survey', 'user', 'computer', 'system', 'response', 'time'],
            ['the', 'eps', 'user', 'interface', 'system'],
            ['system', 'and', 'human', 'system', 'eps'],
            ['user', 'response', 'time'],
            ['the', 'trees'],
            ['the', 'graph', 'trees'],
            ['graph', 'minors', 'trees', 'and'],
            ['graph', 'minors', 'a', 'survey']
        ]
        self.assertEqual(result.tokens, correct)

    def test_porter_stemmer(self):
        words = [
            'caresses',
            'flies',
            'dies',
            'mules',
            'denied',
            'died',
            'agreed',
            'owned',
            'humbled',
            'sized',
            'meeting',
            'stating',
            'siezing',
            'itemization',
            'sensational',
            'traditional',
            'reference',
            'colonizer',
            'plotted',
        ]
        stems = [
            'caress',
            'fli',
            'die',
            'mule',
            'deni',
            'die',
            'agre',
            'own',
            'humbl',
            'size',
            'meet',
            'state',
            'siez',
            'item',
            'sensat',
            'tradit',
            'refer',
            'colon',
            'plot'
        ]

        for w, s in zip(PorterStemmer(words), stems):
            self.assertEqual(w, s)

    def test_porter_sentence(self):
        corpus = [
            'Caresses flies dies mules denied died agreed owned humbled sized.'
        ]
        stemmed = [
            'caress',
            'fli',
            'die',
            'mule',
            'deni',
            'die',
            'agre',
            'own',
            'humbl',
            'size',
            '.'
        ]

        p = Preprocessor(transformation=PorterStemmer)
        result = p(corpus)[0]
        self.assertEqual(result, stemmed)

    def test_snowball_stemmer(self):
        words = [
            'caresses',
            'flies',
            'dies',
            'mules',
            'denied',
            'died',
            'agreed',
            'owned',
            'humbled',
            'sized',
            'meeting',
            'stating',
            'siezing',
            'itemization',
            'sensational',
            'traditional',
            'reference',
            'colonizer',
            'plotted'
        ]
        stems = [
            'caress',
            'fli',
            'die',
            'mule',
            'deni',
            'die',
            'agre',
            'own',
            'humbl',
            'size',
            'meet',
            'state',
            'siez',
            'item',
            'sensat',
            'tradit',
            'refer',
            'colon',
            'plot'
        ]

        for w, s in zip(SnowballStemmer(words), stems):
            self.assertEqual(w, s)

    def test_snowball_sentence(self):
        corpus = [
            'Caresses flies dies mules denied died agreed owned humbled sized.'
        ]
        stemmed = [
            'caress',
            'fli',
            'die',
            'mule',
            'deni',
            'die',
            'agre',
            'own',
            'humbl',
            'size',
            '.'
        ]

        p = Preprocessor(transformation=SnowballStemmer)
        result = p(corpus)[0]
        self.assertEqual(result, stemmed)

    def test_wordnet_lemmatizer(self):
        words = [
            'dogs',
            'churches',
            'aardwolves',
            'abaci',
            'hardrock'
        ]
        lemas = [
            'dog',
            'church',
            'aardwolf',
            'abacus',
            'hardrock'
        ]

        for w, s in zip(Lemmatizer(words), lemas):
            self.assertEqual(w, s)

    def test_wordnet_lemmatizer_sentence(self):
        corpus = [
            'Pursued brightness insightful blessed lies held timelessly minds.'
        ]
        lemmas = [
            'pursued',
            'brightness',
            'insightful',
            'blessed',
            'lie',
            'held',
            'timelessly',
            'mind',
            '.'
        ]

        p = Preprocessor(transformation=Lemmatizer,)
        result = p(corpus)[0]
        self.assertEqual(result, lemmas)

    def test_faulty_init_parameters(self):
        # Stop word source.
        with self.assertRaises(ValueError):
            Preprocessor(stop_words='faulty_value')
        # Transformation.
        with self.assertRaises(ValueError):
            Preprocessor(transformation='faulty_value')
        # Min/Max df.
        with self.assertRaises(ValueError):
            Preprocessor(min_df='faulty_value')
        with self.assertRaises(ValueError):
            Preprocessor(max_df='faulty_value')
        with self.assertRaises(ValueError):
            Preprocessor(min_df=1.5)
        with self.assertRaises(ValueError):
            Preprocessor(max_df=1.5)

    def test_tokenizer_choice(self):
        test_sentence = 'The quick #brown fox jumps over the lazy dog :-)'
        default_tokens = [
            'the', 'quick', '#', 'brown', 'fox', 'jumps', 'over', 'the',
            'lazy', 'dog', ':', '-', ')'
        ]
        no_punct_tokens = [
            'the', 'quick', 'brown', 'fox', 'jumps', 'over','the', 'lazy',
            'dog'
        ]
        tweet_tokens = [
            'the', 'quick', '#brown', 'fox', 'jumps', 'over', 'the', 'lazy',
            'dog', ':-)'
        ]

        p = Preprocessor(tokenizer='default')
        self.assertEqual(type(p.tokenizer), type(word_tokenize))
        tokens = p(test_sentence)
        self.assertEqual(tokens[0], default_tokens)

        p = Preprocessor(tokenizer='no_punct')
        self.assertEqual(type(p.tokenizer),
                         type(RegexpTokenizer(r'\w+').tokenize))
        tokens = p(test_sentence)
        self.assertEqual(tokens[0], no_punct_tokens)

        p = Preprocessor(tokenizer='twitter')
        self.assertEqual(type(p.tokenizer), type(TweetTokenizer().tokenize))
        tokens = p(test_sentence)
        self.assertEqual(tokens[0], tweet_tokens)

        with self.assertRaises(ValueError):
            Preprocessor(tokenizer='unsupported_value')
