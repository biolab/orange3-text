import tempfile
import unittest

import nltk

from orangecontrib.text import tag
from orangecontrib.text.corpus import Corpus


class POSTaggerTests(unittest.TestCase):
    def test_POSTagger(self):
        corpus = Corpus.from_file('deerwester')
        tagger = tag.default_tagger
        result = tagger.tag_corpus(corpus)
        self.assertTrue(hasattr(result, 'tags'))

        for tokens, tags in zip(result.tokens, result.tags):
            self.assertEqual(len(tokens), len(tags))

    def test_Stanford_check(self):
        model = tempfile.NamedTemporaryFile()
        resource = tempfile.NamedTemporaryFile()
        with self.assertRaises(ValueError):
            tag.StanfordPOSTagger.check(model.name, resource.name)

        with self.assertRaises(ValueError):
            tag.StanfordPOSTagger.check('model', resource.name)

    def test_str(self):
        tagger = tag.POSTagger(nltk.tag.BigramTagger, name='bigram')
        self.assertIn('bigram', str(tagger))
