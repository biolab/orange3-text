import numpy as np
from gensim import corpora, models, matutils

from Orange.data.table import Table
from Orange.data.domain import Domain, ContinuousVariable, StringVariable
from orangecontrib.text.topics import Topics

def chunk_list(l, num):
    num = min(len(l), num)
    avg = len(l) / float(num)
    out = []
    last = 0.0
    while last < len(l):
        out.append(l[int(last):int(last + avg)])
        last += avg
    return out

MAX_WORDS = 1000


class LDA:
    def __init__(self, text, num_topics=5, callback=None):
        """
        Wraper for Gensim LDA model.

        :param text: Preprocessed text.
        :param num_topics: Number of topics to infer.
        :return: None
        """
        self.text = text
        self.num_topics = num_topics

        # generate dict and corpus
        dictionary = corpora.Dictionary(self.text)
        corpus = [dictionary.doc2bow(t) for t in self.text]

        self.lda = models.LdaModel(id2word=dictionary, num_topics=self.num_topics)
        done = 0
        for i, part in enumerate(chunk_list(corpus, 95)):
            self.lda.update(part)
            done += len(part)
            callback(95.0*done/len(corpus))

        corpus = self.lda[corpus]

        self.topic_names = ['Topic{} ({})'.format(i, ', '.join(words))
                            for i, words in enumerate(self.__topics_words(3), 1)]
        self.corpus = corpus

    def insert_topics_into_corpus(self, corp_in):
        """
        Insert topical representation into corpus.

        :param corp_in: Corpus into whic we want to insert topical representations
        :return: `Orange.data.table.Table`
        """
        matrix = matutils.corpus2dense(self.corpus,
                                       num_terms=self.num_topics).T

        # Generate the new table.
        attr = [ContinuousVariable(n) for n in self.topic_names]
        domain = Domain(attr,
                        corp_in.domain.class_vars,
                        metas=corp_in.domain.metas)

        return Table.from_numpy(domain,
                                matrix,
                                Y=corp_in._Y,
                                metas=corp_in.metas)

    def get_topics_table_by_id(self, topic_id):
        """
        Transform topics from gensim LDA model to table.

        :param lda: gensim LDA model.
        :return: `Orange.data.table.Table`.
        """
        words = self.__topics_words(MAX_WORDS)
        weights = self.__topics_weights(MAX_WORDS)
        if topic_id >= len(words):
            raise ValueError("Too large topic ID.")

        num_words = len(words[topic_id])

        data = np.zeros((num_words, 2), dtype=object)
        data[:, 0] = words[topic_id]
        data[:, 1] = weights[topic_id]

        metas = [StringVariable(self.topic_names[topic_id]),
                 ContinuousVariable("Topic{}_weights".format(topic_id+1))]
        metas[-1]._out_format = '%.2e'

        domain = Domain([], metas=metas)
        t = Topics.from_numpy(domain,
                              X=np.zeros((num_words, 0)),
                              metas=data)
        t.W = data[:, 1]
        return t

    def get_top_words_by_id(self, topic_id):
        topics = self.__topics_words(10)
        if topic_id >= len(topics):
            raise ValueError("Too large topic ID.")
        return topics[topic_id]

    def __topics_words(self, num_of_words):
        """ Returns list of list of topic words. """
        x = self.lda.show_topics(num_topics=-1, num_words=num_of_words, formatted=False)
        # `show_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[0] for i in topic[1]] for topic in x]

    def __topics_weights(self, num_of_words):
        """ Returns list of list of topic weights. """
        topics = self.lda.show_topics(num_topics=-1, num_words=num_of_words, formatted=False)
        # `show_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[1] for i in t[1]] for t in topics]
