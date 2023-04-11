import pickle
import copy
import tempfile
import unittest

from orangecontrib.text import tag
from orangecontrib.text.corpus import Corpus


class POSTaggerTests(unittest.TestCase):
    def setUp(self):
        self.tagger = tag.AveragedPerceptronTagger()
        self.corpus = Corpus.from_file('deerwester')

    def test_POSTagger(self):
        result = self.tagger(self.corpus)
        self.assertTrue(hasattr(result, 'pos_tags'))
        for tokens, tags in zip(result.tokens, result.pos_tags):
            self.assertEqual(len(tokens), len(tags))

    def test_str(self):
        self.assertEqual('Averaged Perceptron Tagger', str(self.tagger))

    def test_preprocess(self):
        corpus = self.tagger(self.corpus)
        self.assertIsNotNone(corpus.pos_tags)
        self.assertEqual(len(corpus.used_preprocessor.preprocessors), 2)

    def test_can_deepcopy(self):
        copied = copy.deepcopy(self.tagger)
        self.assertTrue(all(
            copied(self.corpus).pos_tags == self.tagger(self.corpus).pos_tags))

    def test_can_pickle(self):
        loaded = pickle.loads(pickle.dumps(self.tagger))
        self.assertTrue(all(
            loaded(self.corpus).pos_tags == self.tagger(self.corpus).pos_tags))

    def test_languages(self):
        taggers = [tag.AveragedPerceptronTagger(), tag.MaxEntTagger()]

        # english is supported by all
        for pp in taggers:
            self.assertIsInstance(pp(self.corpus), Corpus)

        # None supported by none
        self.corpus.attributes["language"] = None
        for pp in taggers:
            with self.assertRaises(ValueError) as cm:
                pp(self.corpus)
            self.assertEqual(
                str(cm.exception), f"{pp.name} does not support the Corpus's language."
            )

        # set to the exotic language that is not supported either
        self.corpus.attributes["language"] = "az"
        for pp in taggers:
            with self.assertRaises(ValueError) as cm:
                pp(self.corpus)
            self.assertEqual(
                str(cm.exception), f"{pp.name} does not support the Corpus's language."
            )


if __name__ == "__main__":
    unittest.main()
