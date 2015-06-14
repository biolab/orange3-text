import os
import unittest
from orangecontrib.text.preprocess import Preprocessor, Stemmer, Lemmatizer


class PreprocessTests(unittest.TestCase):
    sentence = "Human machine interface for lab abc computer applications"
    corpus = ["Human machine interface for lab abc computer applications",
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

    def test_preprocess_corpus_min_df(self):
        p = Preprocessor(lowercase=True, stop_words=['for', 'a', 'of', 'the', 'and', 'to', 'in'], min_df=2)
        corpus = p(self.corpus)
        correct = [['human', 'interface', 'computer'],
                   ['survey', 'user', 'computer', 'system', 'response', 'time'],
                   ['eps', 'user', 'interface', 'system'],
                   ['system', 'human', 'system', 'eps'],
                   ['user', 'response', 'time'],
                   ['trees'],
                   ['graph', 'trees'],
                   ['graph', 'minors', 'trees'],
                   ['graph', 'minors', 'survey']]
        self.assertEqual(len(corpus), len(correct))
        for i, j in zip(corpus, correct):
            self.assertEqual(sorted(i), sorted(j))

    def test_porter_stemmer(self):
        words = ['caresses', 'flies', 'dies', 'mules', 'denied',
                 'died', 'agreed', 'owned', 'humbled', 'sized',
                 'meeting', 'stating', 'siezing', 'itemization',
                 'sensational', 'traditional', 'reference', 'colonizer',
                 'plotted']
        stems = ['caress', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own',
                 'humbl', 'size', 'meet', 'state', 'siez', 'item', 'sensat',
                 'tradit', 'refer', 'colon', 'plot']

        for w, s in zip(words, stems):
            self.assertEqual(Stemmer(w), s)

    def test_porter_sentence(self):
        corpus = ['Caresses flies dies mules denied died agreed owned humbled sized.']
        stemmed = ['caress', 'fli', 'die', 'mule', 'deni', 'die', 'agre', 'own', 'humbl']

        p = Preprocessor(lowercase=True, stop_words=None, trans=Stemmer)
        corpus = p(corpus)
        print(corpus)
        self.assertEqual(sorted(corpus), sorted(stemmed))


    def test_wordnet_lematizer(self):
        words = ['dogs', 'churches', 'aardwolves', 'abaci', 'hardrock']
        lemas = ['dog', 'church', 'aardwolf', 'abacus', 'hardrock']

        for w, s in zip(words, lemas):
            self.assertEqual(Lemmatizer(w), s)

