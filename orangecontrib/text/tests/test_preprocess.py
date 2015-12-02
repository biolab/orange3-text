import os
import unittest
from orangecontrib.text.corpus import Corpus, get_sample_corpora_dir
from orangecontrib.text.preprocess import Preprocessor, PorterStemmer, SnowballStemmer, Lemmatizer


class PreprocessTests(unittest.TestCase):
    sentence = "Human machine interface for lab abc computer applications"
    corpus = [
        "Human machine interface for lab abc computer applications",
        "A survey of user opinion of computer system response time",
        "The EPS user interface management system",
        "System and human system engineering testing of EPS",
        "Relation of user perceived response time to error measurement",
        "The generation of random binary unordered trees",
        "The intersection graph of paths in trees",
        "Graph minors IV Widths of trees and well quasi ordering",
        "Graph minors A survey",
        ]

    def test_preprocess_one_sentence_lowercase_stopwords(self):
        p = Preprocessor(lowercase=True, stop_words=None)
        corpus = p(self.sentence)
        self.assertEqual(sorted(corpus),
                         sorted(['abc', 'applications', 'computer', 'for', 'human', 'interface', 'lab', 'machine']))

        p = Preprocessor(lowercase=True, stop_words='english')
        corpus = p(self.sentence)
        self.assertEqual(sorted(corpus),
                         sorted(['abc', 'applications', 'computer', 'human', 'interface', 'lab', 'machine']))

        p = Preprocessor(lowercase=True, stop_words=['abc', 'applications', 'computer', 'for'])
        corpus = p(self.sentence)
        self.assertEqual(sorted(corpus),
                         sorted(['human', 'interface', 'lab', 'machine']))

        p = Preprocessor(lowercase=False, stop_words=None)
        corpus = p(self.sentence)
        self.assertEqual(sorted(corpus),
                         sorted(['Human', 'abc', 'applications', 'computer', 'for', 'interface', 'lab', 'machine']))

        p = Preprocessor(lowercase=False, stop_words='english')
        corpus = p(self.sentence)
        self.assertEqual(sorted(corpus),
                         sorted(['abc', 'applications', 'computer', 'Human', 'interface', 'lab', 'machine']))

        p = Preprocessor(lowercase=False, stop_words=['abc', 'applications', 'computer', 'for'])
        corpus = p(self.sentence)
        self.assertEqual(sorted(corpus),
                         sorted(['Human', 'interface', 'lab', 'machine']))

    def test_preprocess_corpus_int_df(self):
        p = Preprocessor(lowercase=True, min_df=1, max_df=4)
        corpus = p(self.corpus)
        correct = [
            ['interface', 'computer'],
            ['a', 'survey', 'user', 'computer', 'system', 'response', 'time'],
            ['the', 'eps', 'user', 'interface', 'system'],
            ['system', 'and', 'system', 'eps'],
            ['relation', 'user', 'response', 'time'],
            ['the', 'trees'],
            ['the', 'trees'],
            ['minors', 'iv', 'widths', 'trees', 'and'],
            ['minors', 'a', 'survey']
        ]
        self.assertEqual(corpus, correct)

    def test_preprocess_corpus_float_df(self):
        p = Preprocessor(lowercase=True, min_df=0.1, max_df=0.8)
        corpus = p(self.corpus)
        correct = [
            ['interface', 'computer'],
            ['a', 'survey', 'of', 'user', 'of', 'computer', 'system', 'response', 'time'],
            ['the', 'eps', 'user', 'interface', 'system'],
            ['system', 'and', 'system', 'of', 'eps'],
            ['relation', 'of', 'user', 'response', 'time'],
            ['the', 'of', 'trees'],
            ['the', 'of', 'trees'],
            ['minors', 'iv', 'widths', 'of', 'trees', 'and'],
            ['minors', 'a', 'survey']
        ]
        self.assertEqual(corpus, correct)

    # Porter stemmer.
    def test_porter_stemmer(self):
        words = ['caresses', 'flies', 'dies', 'mules', 'denied',
                 'died', 'agreed', 'owned', 'humbled', 'sized',
                 'meeting', 'stating', 'siezing', 'itemization',
                 'sensational', 'traditional', 'reference', 'colonizer',
                 'plotted']
        stems = ['caress', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own',
                 'humbl', 'size', 'meet', 'state', 'siez', 'item', 'sensat',
                 'tradit', 'refer', 'colon', 'plot']

        for w, s in zip(PorterStemmer(words), stems):
            self.assertEqual(w, s)

    def test_porter_sentence(self):
        corpus = ['Caresses flies dies mules denied died agreed owned humbled sized.']
        stemmed = ['caress', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own', 'humbl', 'size', '.']

        p = Preprocessor(lowercase=True, stop_words=None,
                         transformation=PorterStemmer, use_twitter_tokenizer=False)
        corpus = p(corpus)[0]
        self.assertEqual(corpus, stemmed)

    # Snowball stemmer.
    def test_snowball_stemmer(self):
        words = ['caresses', 'flies', 'dies', 'mules', 'denied',
                 'died', 'agreed', 'owned', 'humbled', 'sized',
                 'meeting', 'stating', 'siezing', 'itemization',
                 'sensational', 'traditional', 'reference', 'colonizer',
                 'plotted']
        stems = ['caress', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own',
                 'humbl', 'size', 'meet', 'state', 'siez', 'item', 'sensat',
                 'tradit', 'refer', 'colon', 'plot']

        for w, s in zip(SnowballStemmer(words), stems):
            self.assertEqual(w, s)

    def test_snowball_sentence(self):
        corpus = ['Caresses flies dies mules denied died agreed owned humbled sized.']
        stemmed = ['caress', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own', 'humbl', 'size', '.']

        p = Preprocessor(lowercase=True, stop_words=None,
                         transformation=SnowballStemmer, use_twitter_tokenizer=False)
        corpus = p(corpus)[0]
        self.assertEqual(corpus, stemmed)

    # Lemmatizer.
    def test_wordnet_lemmatizer(self):
        words = ['dogs', 'churches', 'aardwolves', 'abaci', 'hardrock']
        lemas = ['dog', 'church', 'aardwolf', 'abacus', 'hardrock']

        for w, s in zip(Lemmatizer(words), lemas):
            self.assertEqual(w, s)

    def test_wordnet_lemmatizer_sentence(self):
        corpus = ['Pursued brightness insightful blessed lies held timelessly minds.']
        lemmas = ['pursued', 'brightness', 'insightful', 'blessed', 'lie', 'held', 'timelessly', 'mind', '.']

        p = Preprocessor(lowercase=True, stop_words=None,
                         transformation=Lemmatizer, use_twitter_tokenizer=False)
        corpus = p(corpus)[0]
        self.assertEqual(corpus, lemmas)

    def test_full_corpus_preprocess(self):
        corpus = Corpus.from_file(os.path.join(get_sample_corpora_dir(), 'deerwester.tab'))
        p = Preprocessor(lowercase=True, stop_words=None)
        preprocessed_documents = [
            ['human', 'machine', 'interface', 'for', 'lab', 'abc', 'computer', 'applications'],
            ['a', 'survey', 'of', 'user', 'opinion', 'of', 'computer', 'system', 'response', 'time'],
            ['the', 'eps', 'user', 'interface', 'management', 'system'],
            ['system', 'and', 'human', 'system', 'engineering', 'testing', 'of', 'eps'],
            ['relation', 'of', 'user', 'perceived', 'response', 'time', 'to', 'error', 'measurement'],
            ['the', 'generation', 'of', 'random', 'binary', 'unordered', 'trees'],
            ['the', 'intersection', 'graph', 'of', 'paths', 'in', 'trees'],
            ['graph', 'minors', 'iv', 'widths', 'of', 'trees', 'and', 'well', 'quasi', 'ordering'],
            ['graph', 'minors', 'a', 'survey']
        ]
        self.assertEqual(p(corpus), preprocessed_documents)
