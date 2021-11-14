import inspect
from collections import Counter
from warnings import warn

from gensim import matutils
import numpy as np
from gensim.corpora import Dictionary
from gensim.models.callbacks import Metric

from Orange.data import StringVariable, ContinuousVariable, Domain
from Orange.data.table import Table
from Orange.util import dummy_callback

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.util import chunkable

MAX_WORDS = 1000


class Topic(Table):
    """ Dummy wrapper for Table so signals can distinguish Topic from Data.
    """

    def __new__(cls, *args, **kwargs):
        """ Bypass Table.__new__. """
        return object.__new__(Topic)


class Topics(Table):
    """ Dummy wrapper for Table so signals can distinguish All Topics from Data.
    """


class GensimProgressCallback(Metric):
    """
    Callback to report the progress
    This callback is a hack since Metric class is made to measure metrics.
    Metric is used since it is the only sort of callback accepted by topic models.
    """
    def __init__(self, callback_fun):
        self.callback_fun = callback_fun
        self.epochs = 0
        # parameters required by Gensim Metric
        self.logger = "shell"
        self.title = "Progress"

    def get_value(self, model, *args, **kwargs):
        """ get_value is called on every epoch - pass """
        self.epochs += 1
        self.callback_fun(self.epochs / model.passes)
        return self.epochs / model.passes


class GensimWrapper:
    name = NotImplemented
    Model = NotImplemented
    num_topics = NotImplemented
    has_negative_weights = False    # whether words can negatively contribute
    # to a topic

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kwargs = kwargs
        self.model = None
        self.topic_names = []
        self.n_words = 0
        self.doc_topic = None
        self.tokens = None
        self.actual_topics = None

    def fit(self, corpus, on_progress=dummy_callback, **kwargs):
        """ Train the model with the corpus.

        Args:
            corpus (Corpus): A corpus to learn topics from.
        """
        if "chunk_number" in kwargs:
            warn(
                "chunk_number is deprecated and will be removed in orange3-text 1.7",
                FutureWarning
            )
        if not len(corpus.dictionary):
            return None
        model_kwars = self.kwargs
        if "callbacks" in inspect.getfullargspec(self.Model).args:
            # if method support callbacks use progress callback to report progress
            # at time of writing this code only LDA support callbacks
            model_kwars = dict(
                model_kwars, callbacks=[GensimProgressCallback(on_progress)]
            )

        id2word = Dictionary(corpus.ngrams_iterator(include_postags=True), prune_at=None)
        self.model = self.Model(
            corpus=corpus.ngrams_corpus, id2word=id2word, **model_kwars
        )
        self.n_words = len(corpus.dictionary)
        self.topic_names = ['Topic {}'.format(i+1) for i in range(self.num_topics)]

    def dummy_method(self, *args, **kwargs):
        pass

    def reset_model(self, corpus):
        warn(
            "reset_model is deprecated and will be removed in orange3-text 1.7. "
            "Model resets with calling fit.",
            FutureWarning)
        # prevent model from updating
        _update = self.Model.update
        self.Model.update = self.dummy_method
        self.id2word = Dictionary(corpus.ngrams_iterator(include_postags=True),
                                  prune_at=None)
        self.model = self.Model(corpus=corpus,
                                id2word=self.id2word, **self.kwargs)
        self.Model.update = _update

    @chunkable
    def update(self, documents):
        warn(
            "update is deprecated and will be removed in orange3-text 1.7.",
            FutureWarning)
        self.model.update(documents)

    def transform(self, corpus):
        """ Create a table with topics representation. """
        topics = self.model[corpus.ngrams_corpus]
        self.actual_topics = self.model.get_topics().shape[0]
        matrix = matutils.corpus2dense(
            topics, num_docs=len(corpus), num_terms=self.num_topics
        ).T.astype(np.float64)
        corpus = corpus.extend_attributes(
            matrix[:, :self.actual_topics],
            self.topic_names[:self.actual_topics]
        )
        self.doc_topic = matrix[:, :self.actual_topics]
        self.tokens = corpus.tokens
        corpus.store_tokens(self.tokens)
        return corpus

    def fit_transform(self, corpus, **kwargs):
        self.fit(corpus, **kwargs)
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
                 ContinuousVariable("Topic {} weights".format(topic_id + 1))]
        metas[-1]._out_format = '%.2e'

        domain = Domain([], metas=metas)
        t = Topic.from_numpy(domain,
                             X=np.zeros((num_words, 0)),
                             metas=data)
        t.W = data[:, 1]
        t.name = 'Topic {}'.format(topic_id + 1)

        # needed for coloring in word cloud
        t.attributes["topic-method-name"] = self.model.__class__.__name__
        return t

    @staticmethod
    def _marginal_probability(tokens, doc_topic):
        """
        Compute marginal probability of a topic, that is the probability of a
        topic across all documents.

        :return: np.array of marginal topic probabilities
        :return: number of tokens
        """
        doc_length = [len(i) for i in tokens]
        num_tokens = sum(doc_length)
        doc_length[:] = [x / num_tokens for x in doc_length]
        return np.reshape(np.sum(doc_topic.T * doc_length, axis=1), (-1, 1)),\
            num_tokens

    def get_all_topics_table(self):
        """ Transform all topics from gensim model to table. """
        all_words = self._topics_words(self.n_words)
        all_weights = self._topics_weights(self.n_words)
        sorted_words = sorted(all_words[0])
        n_topics = len(all_words)

        X = []
        for words, weights in zip(all_words, all_weights):
            weights = [we for wo, we in sorted(zip(words, weights))]
            X.append(weights)
        X = np.array(X)

        # take only first n_topics; e.g. when user requested 10, but gensim
        # returns only 9 — when the rank is lower than num_topics requested
        names = np.array(self.topic_names[:n_topics], dtype=object)[:, None]

        attrs = [ContinuousVariable(w) for w in sorted_words]
        metas = [StringVariable('Topics'),
                 ContinuousVariable('Marginal Topic Probability')]

        marg_proba, num_tokens = self._marginal_probability(self.tokens,
                                                            self.doc_topic)
        topic_proba = np.array(marg_proba, dtype=object)

        t = Topics.from_numpy(Domain(attrs, metas=metas), X=X,
                              metas=np.hstack((names, topic_proba)))
        t.name = 'All topics'
        # required for distinguishing between models in OWRelevantTerms
        t.attributes.update([('Model', f'{self.name}'),
                             ('Number of tokens', num_tokens)])
        return t

    def get_top_words_by_id(self, topic_id, num_of_words=10):
        topics = self._topics_words(num_of_words=num_of_words)
        weights = self._topics_weights(num_of_words=num_of_words)
        if not 0 <= topic_id < self.num_topics:
            raise ValueError("Invalid {}".format(topic_id))
        elif topic_id >= len(topics):
            return [], []
        return topics[topic_id], weights[topic_id]

    def _topics_words(self, num_of_words):
        """ Returns list of list of topic words. """
        x = self.model.show_topics(self.num_topics, num_of_words, formatted=False)
        # `show_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[0] for i in topic[1]] for topic in x]

    def _topics_weights(self, num_of_words):
        """ Returns list of list of topic weights. """
        topics = self.model.show_topics(self.num_topics, num_of_words,
                                        formatted=False)
        # `show_topics` method return a list of `(topic_number, topic)` tuples,
        # where `topic` is a list of `(word, probability)` tuples.
        return [[i[1] for i in t[1]] for t in topics]
