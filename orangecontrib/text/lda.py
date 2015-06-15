import numpy as np
from gensim import corpora, models, matutils

from Orange.data.table import Table
from Orange.data.domain import Domain, ContinuousVariable, DiscreteVariable


class LDA:
    def __init__(self, text, num_topics=5, use_tf_idf=False):
        """
        Wraper for Gensim LDA model.

        :param text: Preprocessed text.
        :param num_topics: Number of topics to infer.
        :param use_tf_idf: Whether to use TF-IDF preprocessing.
        :return: None
        """
        self.text = text
        self.num_topics = num_topics
        self.use_tf_idf = use_tf_idf

        # generate dict and corpus
        dictionary = corpora.Dictionary(self.text)
        corpus = [dictionary.doc2bow(t) for t in self.text]

        # TODO remove tfidf when this will be separate widget
        if self.use_tf_idf:
            tf_idf = models.TfidfModel(corpus)
            corpus = tf_idf[corpus]

        lda = models.LdaModel(corpus, id2word=dictionary,
                              num_topics=self.num_topics)
        corpus = lda[corpus]

        self.corpus = corpus
        self.lda = lda

    def insert_topics_into_corpus(self, corp_in):
        """
        Insert topical representation into corpus.

        :param corp_in: Corpus into whic we want to insert topical representations
        :return: `Orange.data.table.Table`
        """
        matrix = matutils.corpus2dense(self.corpus,
                                       num_terms=self.num_topics).T

        # Generate the new table.
        topics = self.lda.show_topics(num_topics=-1, num_words=3, formatted=False)
        names = [', '.join([i[1] for i in t]) for t in topics]
        attr = [ContinuousVariable("T{} ({}, ...)".format(i, n)) for i, n in enumerate(names, 1)]
        domain = Domain(attr,
                        corp_in.domain.class_vars,
                        metas=corp_in.domain.metas)

        return Table.from_numpy(domain,
                                matrix,
                                Y=corp_in._Y,
                                metas=corp_in.metas)

    def get_topics_table(self):
        """
        Transform topics from gensim LDA model to table.

        :param lda: gensim LDA model.
        :return: `Orange.data.table.Table`.
        """

        topics = self.lda.show_topics(num_topics=-1, num_words=-1, formatted=False)
        ntopics = len(topics)
        nwords = max([len(it) for it in topics])

        data = np.zeros((nwords, 2*ntopics), dtype=object)

        for i, topic in enumerate(topics):
            data[:, 2*i] = [item[1] for item in topic]
            data[:, 2*i+1] = ['{:.3e}'.format(item[0]) for item in topic]

        print(data)
        attr = []
        for i in range(ntopics):
            attr.append(DiscreteVariable("T{}".format(i+1), values=list(data[:, 2*i])))
            attr.append(ContinuousVariable("T{}_weights".format(i+1)))
        domain = Domain(attr)

        for t in range(data.shape[1]):
            for w in range(data.shape[0]):
                if t % 2 == 0:
                    data[w, t] = attr[t].to_val(data[w, t])

        return Table.from_numpy(domain, data)
