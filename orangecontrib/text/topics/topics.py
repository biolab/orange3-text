import numpy as np
from orangecontrib.text.preprocess.dictionary import Dictionary
from gensim.models.callbacks import Metric

from Orange.data import StringVariable, ContinuousVariable, Domain
from Orange.data.table import Table
from Orange.util import dummy_callback

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.vectorization import BowVectorizer

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
    @classmethod
    def from_file(cls, filename):
        t = super().from_file(filename)
        if not isinstance(t, cls):
            t = cls.from_numpy(t.domain, t.X, t.Y, t.metas, t.W, attributes=t.attributes)
        return t


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


def infer_ngrams_corpus(corpus, return_dict=False):

    bow_features = [
        (i, attribute.name) for i, attribute in enumerate(corpus.domain.attributes)
        if 'bow-feature' in attribute.attributes
    ]

    if len(bow_features) == 0:
        corpus = BowVectorizer().transform(corpus)
        bow_features = [
            (i, attribute.name) for i, attribute in enumerate(corpus.domain.attributes)
            if 'bow-feature' in attribute.attributes
        ]

    feature_presence = corpus.X.sum(axis=0)
    keep = [(i, a) for i, a in bow_features if feature_presence[0, i] > 0]
    # sort features by the order in the dictionary
    dictionary = Dictionary(corpus.ngrams_iterator(include_postags=False),
                            prune_at=None)
    idx_of_keep = np.argsort([dictionary.token2id[a] for _, a in keep])
    keep = [keep[i][0] for i in idx_of_keep]
    result = []
    if len(dictionary) > 0:
        result = Sparse2Corpus(corpus.X[:, keep].T)

    return (result, dictionary) if return_dict else result


class SklearnWrapper:
    name = NotImplemented
    Model = NotImplemented
    num_topics = NotImplemented

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kwargs = kwargs
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.topic_names = []
        self.n_words = 0
        self.doc_topic = None
        self.tokens = None
        self.actual_topics = None

    def fit(self, corpus, on_progress=dummy_callback):
        """ Train the model with the corpus.

        Args:
            corpus (Corpus): A corpus to learn topics from.
        """
        X, self.feature_names = self._corpus_to_dtm(corpus)

        if X.shape[1] == 0:
            return None

        # Initialize model with kwargs
        self.model = self.Model(n_components=self.num_topics, **self.kwargs)
        self.model.fit(X)

        self.n_words = X.shape[1]
        self.actual_topics = self.model.n_components
        self.topic_names = [f'Topic {i + 1}' for i in range(self.num_topics)]

    def transform(self, corpus):
        """ Create a table with topics representation. """
        X, _ = self._corpus_to_dtm(corpus)

        # Get document-topic distribution
        doc_topic = self.model.transform(X)
        self.actual_topics = self.model.n_components

        matrix = doc_topic.astype(np.float64)
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

    def _corpus_to_dtm(self, corpus):
        """Convert corpus to document-term matrix."""
        from sklearn.feature_extraction.text import CountVectorizer

        if self.vectorizer is None:
            self.vectorizer = CountVectorizer(
                tokenizer=lambda x: x,
                preprocessor=lambda x: x,
                token_pattern=None
            )
            X = self.vectorizer.fit_transform(corpus.tokens)
            feature_names = self.vectorizer.get_feature_names_out()
        else:
            X = self.vectorizer.transform(corpus.tokens)
            feature_names = self.vectorizer.get_feature_names_out()

        return X, feature_names

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
                 ContinuousVariable(f"Topic {topic_id + 1} weights")]
        metas[-1]._out_format = '%.2e'

        domain = Domain([], metas=metas)
        t = Topic.from_numpy(
            domain, X=np.zeros((num_words, 0)), metas=data, W=data[:, 1]
        )
        t.name = f"Topic {topic_id + 1}"

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
        topics = []
        for topic_idx in range(self.model.n_components):
            top_indices = self.model.components_[topic_idx].argsort()[
                          -num_of_words:][::-1]
            topics.append([self.feature_names[i] for i in top_indices])
        return topics

    def _topics_weights(self, num_of_words):
        """ Returns list of list of topic weights. """
        weights = []
        for topic_idx in range(self.model.n_components):
            top_indices = self.model.components_[topic_idx].argsort()[
                          -num_of_words:][::-1]
            weights.append(
                [self.model.components_[topic_idx][i] for i in top_indices])
        return weights
