import unittest

from orangecontrib.text.lda import LDA
from orangecontrib.text.corpus import Corpus


class LDATests(unittest.TestCase):
    corp = Corpus.from_file('deerwester')
    text = [d.split() for d in corp.documents]
    model = LDA(text, num_topics=5)

    def test_insert_topic_into_corpus(self):
        corp_topics = self.model.insert_topics_into_corpus(self.corp)
        self.assertEqual(len(corp_topics), len(self.corp))
        self.assertEqual(len(corp_topics.domain.attributes), 5)
        self.assertEqual(corp_topics.X.shape, (len(self.corp), 5))

    def test_get_topic_table_by_id(self):
        topic1 = self.model.get_topics_table_by_id(1)
        self.assertEqual(len(topic1), 45)
        self.assertEqual(topic1.metas.shape, (45, 2))

    def test_top_words_by_topic(self):
        words = self.model.get_top_words_by_id(1)
        self.assertEqual(len(words), 10)

    def test_too_large_id(self):
        with self.assertRaises(ValueError):
            self.model.get_topics_table_by_id(6)
