from gensim import matutils
import numpy as np

from Orange.data import StringVariable, ContinuousVariable, Domain
from Orange.data.table import Table
from orangecontrib.text.corpus import Corpus


def chunks(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


MAX_WORDS = 1000


class Topics(Table):
    """ Dummy wrapper for Table so signals can distinguish Topics from Data.
    """

    def __new__(cls, *args, **kwargs):
        """ Bypass Table.__new__. """
        return object.__new__(Topics)


class GensimWrapper:
    name = NotImplemented
    Model = NotImplemented
    num_topics = NotImplemented

    def fit(self, corpus, progress_callback=None):
        """ Train the model with the corpus.

        Args:
            corpus (Corpus): A corpus to lear topics from.
        """
        if not len(corpus.dictionary):
            return None
        self.reset_model(corpus)
        self.update(corpus, progress_callback)
        self.topic_names = ['Topic{} ({})'.format(i, ', '.join(words))
                            for i, words in enumerate(self._topics_words(3), 1)]

    def reset_model(self, corpus):
        self.model = self.Model(id2word=corpus.ngrams_dictionary)

    def update(self, corpus, progress_callback=None):
        gcorpus = matutils.Sparse2Corpus(corpus.ngrams_matrix.T)

        for i, chunk in enumerate(chunks(gcorpus, np.ceil(len(corpus) / 100))):
            self.model.update(chunk)
            if progress_callback:
                progress_callback(i)

    def transform(self, corpus):
        """ Create a table with topics representation. """
        topics = self.model[matutils.Sparse2Corpus(corpus.ngrams_matrix.T)]
        matrix = matutils.corpus2dense(topics,
                                       num_terms=self.num_topics).T

        # Generate the new table.
        attr = [ContinuousVariable(n) for n in self.topic_names]
        domain = Domain(attr,
                        corpus.domain.class_vars,
                        metas=corpus.domain.metas)

        return Table.from_numpy(domain,
                                matrix,
                                Y=corpus._Y,
                                metas=corpus.metas)

    def fit_transform(self, corpus, progress_callback=None):
        self.fit(corpus, progress_callback)
        return self.transform(corpus)

    def get_topics_table_by_id(self, topic_id):
        """ Transform topics from gensim model to table. """
        words = self._topics_words(MAX_WORDS)
        weights = self._topics_weights(MAX_WORDS)
        if topic_id >= len(words):
            raise ValueError("Too large topic ID.")

        num_words = len(words[topic_id])

        data = np.zeros((num_words, 2), dtype=object)
        data[:, 0] = words[topic_id]
        data[:, 1] = weights[topic_id]

        metas = [StringVariable(self.topic_names[topic_id]),
                 ContinuousVariable("Topic{}_weights".format(topic_id + 1))]
        metas[-1]._out_format = '%.2e'

        domain = Domain([], metas=metas)
        t = Topics.from_numpy(domain,
                              X=np.zeros((num_words, 0)),
                              metas=data)
        t.W = data[:, 1]
        return t

    def get_top_words_by_id(self, topic_id, num_of_words=10):
        topics = self._topics_words(num_of_words=num_of_words)
        if topic_id >= len(topics):
            raise ValueError("Too large topic ID.")
        return topics[topic_id]

    def _topics_words(self, num_of_words):
        """ Returns list of list of topic words. """
        x = self.model.show_topics(-1, num_of_words, formatted=False)
        # `show_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[0] for i in topic[1]] for topic in x]

    def _topics_weights(self, num_of_words):
        """ Returns list of list of topic weights. """
        topics = self.model.show_topics(-1, num_of_words, formatted=False)
        # `show_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[1] for i in t[1]] for t in topics]
