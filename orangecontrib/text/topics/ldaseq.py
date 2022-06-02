import numpy as np
from gensim import models

from .topics import GensimWrapper


class LdaSeqWrapper(GensimWrapper):
    name = 'Dynamic Topic Modeling'
    Model = models.LdaSeqModel

    def __init__(self, **kwargs):
        super().__init__(random_state=0, **kwargs, passes=5)

    def get_topic_matrix(self, corpus):
        res = []
        for doc in corpus.ngrams_corpus:
            res.append(self.model[doc])
        return np.vstack(res)

    def get_num_topic(self):
        return len(self.model.print_topics())

    def _topics_words(self, num_of_words):
        """ Returns list of list of topic words. """
        x = self.model.print_topics(top_terms=num_of_words)
        # `print_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[0] for i in topic] for topic in x]

    def _topics_weights(self, num_of_words):
        """ Returns list of list of topic weights. """
        topics = self.model.print_topics(top_terms=num_of_words)
        # `show_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[1] for i in t] for t in topics]