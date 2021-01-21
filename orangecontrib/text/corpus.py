import os
from collections import Counter, defaultdict
from copy import copy
from numbers import Integral
from itertools import chain
from typing import Union, Optional, List, Tuple

import nltk
import numpy as np
import scipy.sparse as sp
from gensim import corpora
import fasttext

from Orange.data import (
    Variable,
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    RowInstance,
    Table,
    StringVariable,
)
from Orange.preprocess.transformation import Identity
# uncomment when Orange3==3.27 is available
# from Orange.data.util import get_unique_names
# remove when Orange3==3.27 is available
from orangecontrib.text.vectorization.base import get_unique_names
from orangecontrib.text.vectorization import BowVectorizer


def get_sample_corpora_dir():
    path = os.path.dirname(__file__)
    directory = os.path.join(path, 'datasets')
    return os.path.abspath(directory)


def _check_arrays(*arrays):
    for a in arrays:
        if not (a is None or isinstance(a, np.ndarray) or sp.issparse(a)):
            raise TypeError('Argument {} should be of type np.array, sparse or None.'.format(a))

    lengths = set(a.shape[0] for a in arrays if a is not None)
    if len(lengths) > 1:
        raise ValueError('Leading dimension mismatch')

    return lengths.pop() if len(lengths) else 0


class Corpus(Table):
    """Internal class for storing a corpus."""

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(cls)

    def __init__(self, domain=None, X=None, Y=None, metas=None, W=None,
                 text_features=None, ids=None):
        """
        Args:
            domain (Orange.data.Domain): the domain for this Corpus
            X (numpy.ndarray): attributes
            Y (numpy.ndarray): class variables
            metas (numpy.ndarray): meta attributes; e.g. text
            W (numpy.ndarray): instance weights
            text_features (list): meta attributes that are used for
                text mining. Infer them if None.
            ids (numpy.ndarray): Indices
        """
        n_doc = _check_arrays(X, Y, metas)

        self.X = X if X is not None else np.zeros((n_doc, 0))
        self.Y = Y if Y is not None else np.zeros((n_doc, 0))
        self.metas = metas if metas is not None else np.zeros((n_doc, 0))
        self.W = W if W is not None else np.zeros((n_doc, 0))
        self.domain = domain
        self.text_features = []    # list of text features for mining
        self._tokens = None
        self._dictionary = None
        self._ngrams_corpus = None
        self.ngram_range = (1, 1)
        self.attributes = {}
        self.pos_tags = None
        from orangecontrib.text.preprocess import PreprocessorList
        self.__used_preprocessor = PreprocessorList([])   # required for compute values
        self._titles: Optional[np.ndarray] = None
        self.languages = None
        self._pp_documents = None  # preprocessed documents

        if domain is not None and text_features is None:
            self._infer_text_features()
        elif domain is not None:
            self.set_text_features(text_features)

        if ids is not None:
            self.ids = ids
        else:
            Table._init_ids(self)
        self._set_unique_titles()

    @property
    def used_preprocessor(self):
        return self.__used_preprocessor  # type: PreprocessorList

    @used_preprocessor.setter
    def used_preprocessor(self, pp):
        from orangecontrib.text.preprocess import PreprocessorList, Preprocessor

        if isinstance(pp, PreprocessorList):
            self.__used_preprocessor = PreprocessorList(list(pp.preprocessors))
        elif isinstance(pp, Preprocessor):
            self.__used_preprocessor.preprocessors.append(pp)
        else:
            raise NotImplementedError

    def _find_identical_feature(self, feature: Variable) -> Optional[Variable]:
        """
        Find a renamed feature in the domain which is identical to a feature.

        Parameters
        ----------
        feature
            A variable to find an identical variable in the domain.

        Returns
        -------
        Variable which is identical to a feature (have different name but has
        Identity(feature) in compute value.
        """
        for var in chain(self.domain.variables, self.domain.metas):
            if (
                var == feature
                or isinstance(var.compute_value, Identity)
                and var.compute_value.variable == feature
            ):
                return var
        return None

    def set_text_features(self, feats: Optional[List[Variable]]) -> None:
        """
        Select which meta-attributes to include when mining text.

        Parameters
        ----------
        feats
            List of text features to include. If None infer them.
        """
        if feats is not None:
            feats = copy(feats)  # copy to not edit passed array inplace
            for i, f in enumerate(feats):
                if f not in chain(self.domain.variables, self.domain.metas):
                    # if not exact feature in the domain, it may be renamed
                    # find identity - renamed feature
                    id_feat = self._find_identical_feature(f)
                    if id_feat is not None:
                        feats[i] = id_feat
                    else:
                        raise ValueError('Feature "{}" not found.'.format(f))
            if len(set(feats)) != len(feats):
                raise ValueError('Text features must be unique.')
            self.text_features = feats
        else:
            self._infer_text_features()
        self._tokens = None     # invalidate tokens

    def set_title_variable(
            self, title_variable: Union[StringVariable, str, None]
    ) -> None:
        """
        Set the title attribute. Only one column can be a title attribute.

        Parameters
        ----------
        title_variable
            Variable that need to be set as a title variable. If it is None,
            do not set a variable.
        """
        for a in self.domain.variables + self.domain.metas:
            a.attributes.pop("title", None)

        if title_variable and title_variable in self.domain:
            self.domain[title_variable].attributes["title"] = True

        self._set_unique_titles()

    def _set_unique_titles(self):
        """
        Define self._titles variable as a list of titles (a title for each
        document). It is used to have an unique title for each document. In
        case when the document have the same title as the other document we
        put a number beside.
        """
        if self.domain is None:
            return
        attrs = [attr for attr in
                 chain(self.domain.variables, self.domain.metas)
                 if attr.attributes.get('title', False)]

        if attrs:
            self._titles = np.array(self._unique_titles(
                self.documents_from_features(attrs)))
        else:
            self._titles = np.array([
                'Document {}'.format(i + 1) for i in range(len(self))])

    @staticmethod
    def _unique_titles(titles: List[str]) -> List[str]:
        """
        Function adds numbers to the non-unique values fo the title.

        Parameters
        ----------
        titles
            List of titles - not necessary unique

        Returns
        -------
        List with unique titles.
        """
        counts = Counter(titles)
        cur_appearances = defaultdict(int)
        new_titles = []
        for t in titles:
            if counts[t] > 1:
                cur_appearances[t] += 1
                t += f" ({cur_appearances[t]})"
            new_titles.append(t)
        return new_titles

    def detect_languages(self):
        """
        Detects language of each document using fastText language
        identification model.
        [A. Joulin, E. Grave, P. Bojanowski, T. Mikolov,
        Bag of Tricks for Efficient Text Classification],
        [A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov,
        FastText.zip: Compressing text classification models]
        """
        path = os.path.join(os.path.dirname(__file__), 'models', 'lid.176.ftz')
        model = fasttext.load_model(path)
        texts = [' '.join(t.replace('\n', ' ').split(' ')[:2000])
                 for t in self.documents]
        self.languages = [model.predict(t)[0][0].replace('__label__', '')
                          for t in texts]

    def _infer_text_features(self):
        """
        Infer which text features to use. If nothing was provided
        in the file header, use the first text feature.
        """
        include_feats = []
        first = None
        for attr in self.domain.metas:
            if attr.is_string:
                if first is None:
                    first = attr
                if attr.attributes.get('include', 'False') == 'True':
                    include_feats.append(attr)
        if len(include_feats) == 0 and first:
            include_feats.append(first)
        self.set_text_features(include_feats)

    def extend_corpus(self, metadata, Y):
        """
        Append documents to corpus.

        Args:
            metadata (numpy.ndarray): Meta data
            Y (numpy.ndarray): Class variables
        """
        if np.prod(self.X.shape) != 0:
            raise ValueError("Extending corpus only works when X is empty"
                             "while the shape of X is {}".format(self.X.shape))

        self.metas = np.vstack((self.metas, metadata))

        cv = self.domain.class_var
        for val in set(filter(None, Y)):
            if val not in cv.values:
                cv.add_value(val)
        new_Y = np.array([cv.to_val(i) for i in Y])[:, None]
        self._Y = np.vstack((self._Y, new_Y))

        self.X = self.W = np.zeros((self.metas.shape[0], 0))
        Table._init_ids(self)

        self._tokens = None     # invalidate tokens
        self._set_unique_titles()

    def extend_attributes(
            self, X, feature_names, feature_values=None, compute_values=None,
            var_attrs=None, sparse=False, rename_existing=False
        ):
        """
        Append features to corpus. If `feature_values` argument is present,
        features will be Discrete else Continuous.

        Args:
            X (numpy.ndarray or scipy.sparse.csr_matrix): Features values to append
            feature_names (list): List of string containing feature names
            feature_values (list): A list of possible values for Discrete features.
            compute_values (list): Compute values for corresponding features.
            var_attrs (dict): Additional attributes appended to variable.attributes.
            sparse (bool): Whether the features should be marked as sparse.
            rename_existing (bool): When true and names are not unique rename
                exiting features; if false rename new features
        """
        def _rename_features(additional_names: List) -> Tuple[List, List, List]:
            cur_attr = list(self.domain.attributes)
            cur_class = self.domain.class_var
            cur_meta = list(self.domain.metas)
            if rename_existing:
                current_vars = (
                        cur_attr + (
                    [cur_class] if cur_class else []) + cur_meta
                )
                current_names = [a.name for a in current_vars]
                new_names = get_unique_names(
                    additional_names, current_names, equal_numbers=False
                )
                renamed_vars = [
                    var.renamed(n) for var, n in zip(current_vars, new_names)
                ]
                cur_attr = renamed_vars[:len(cur_attr)]
                cur_class = renamed_vars[len(cur_attr)] if cur_class else None
                cur_meta = renamed_vars[-len(cur_meta):]
            return cur_attr, cur_class, cur_meta

        if sp.issparse(self.X) or sp.issparse(X):
            X = sp.hstack((self.X, X)).tocsr()
        else:
            X = np.hstack((self.X, X))

        if compute_values is None:
            compute_values = [None] * X.shape[1]
        if feature_values is None:
            feature_values = [None] * X.shape[1]

        # rename existing variables if required
        curr_attributes, curr_class_var, curr_metas = _rename_features(
            feature_names
        )
        if not rename_existing:
            # rename new feature names if required
            feature_names = get_unique_names(
                self.domain, feature_names, equal_numbers=False
            )

        additional_attributes = []
        for f, values, cv in zip(feature_names, feature_values, compute_values):
            if values is not None:
                var = DiscreteVariable(f, values=values, compute_value=cv)
            else:
                var = ContinuousVariable(f, compute_value=cv)
            var.sparse = sparse     # don't pass this to constructor so this works with Orange < 3.8.0
            if cv is not None:      # set original variable for cv
                cv.variable = var
            if isinstance(var_attrs, dict):
                var.attributes.update(var_attrs)
            additional_attributes.append(var)

        new_domain = Domain(
                attributes=curr_attributes + additional_attributes,
                class_vars=curr_class_var,
                metas=curr_metas
        )
        c = Corpus(
            new_domain,
            X,
            self.Y.copy(),
            self.metas.copy(),
            self.W.copy(),
            copy(self.text_features)
        )
        Corpus.retain_preprocessing(self, c)
        return c

    @property
    def documents(self):
        """ Returns a list of strings representing documents — created
        by joining selected text features. """
        return self.documents_from_features(self.text_features)

    @property
    def pp_documents(self):
        """ Preprocessed documents (transformed). """
        return self._pp_documents or self.documents

    @pp_documents.setter
    def pp_documents(self, documents):
        self._pp_documents = documents

    @property
    def titles(self):
        """ Returns a list of titles. """
        assert self._titles is not None
        return self._titles

    def documents_from_features(self, feats):
        """
        Args:
            feats (list): A list fo features to join.

        Returns: a list of strings constructed by joining feats.
        """
        # create a Table where feats are in metas
        data = Table.from_table(Domain([], [], [i.name for i in feats],
                                       source=self.domain), self)

        # When we use only features coming from sparse X data.metas is sparse.
        # Transform it to dense.
        if sp.issparse(data.metas):
            data.metas = data.metas.toarray()

        return [' '.join(f.str_val(val) for f, val in zip(data.domain.metas, row))
                for row in data.metas]

    def store_tokens(self, tokens, dictionary=None):
        """
        Args:
            tokens (list): List of lists containing tokens.
        """
        self._tokens = np.array(tokens, dtype=object)
        self._dictionary = dictionary or corpora.Dictionary(self.tokens)

    @property
    def tokens(self):
        """
        np.ndarray: A list of lists containing tokens. If tokens are not yet
        present, run default preprocessor and return tokens.
        """
        if self._tokens is None:
            return self._base_tokens()[0]
        return self._tokens

    def has_tokens(self):
        """ Return whether corpus is preprocessed or not. """
        return self._tokens is not None

    def _base_tokens(self):
        from orangecontrib.text.preprocess import BASE_TRANSFORMER, \
            BASE_TOKENIZER, PreprocessorList

        # don't use anything that requires NLTK data to assure async download
        base_preprocessors = PreprocessorList([BASE_TRANSFORMER,
                                               BASE_TOKENIZER])
        corpus = base_preprocessors(self)
        return corpus.tokens, corpus.dictionary

    @property
    def dictionary(self):
        """
        corpora.Dictionary: A token to id mapper.
        """
        if self._dictionary is None:
            return self._base_tokens()[1]
        return self._dictionary

    def ngrams_iterator(self, join_with=' ', include_postags=False):
        if self.pos_tags is None:
            include_postags = False

        if include_postags:
            data = zip(self.tokens, self.pos_tags)
        else:
            data = self.tokens

        if join_with is None:
            processor = lambda doc, n: nltk.ngrams(doc, n)
        elif include_postags:
            processor = lambda doc, n: (join_with.join(token + '_' + tag for token, tag in ngram)
                                        for ngram in nltk.ngrams(zip(*doc), n))
        else:
            processor = lambda doc, n: (join_with.join(ngram) for ngram in nltk.ngrams(doc, n))

        return (list(chain(*(processor(doc, n)
                for n in range(self.ngram_range[0], self.ngram_range[1]+1))))
                for doc in data)

    @property
    def ngrams_corpus(self):
        if self._ngrams_corpus is None:
            return BowVectorizer().transform(self).ngrams_corpus
        return self._ngrams_corpus

    @ngrams_corpus.setter
    def ngrams_corpus(self, value):
        self._ngrams_corpus = value

    @property
    def ngrams(self):
        """generator: Ngram representations of documents."""
        return self.ngrams_iterator(join_with=' ')

    def copy(self):
        """Return a copy of the table."""
        c = self.__class__(self.domain, self.X.copy(), self.Y.copy(), self.metas.copy(),
                           self.W.copy(), copy(self.text_features))
        # since tokens and dictionary are considered immutable copies are not needed
        c._tokens = self._tokens
        c._dictionary = self._dictionary
        c.ngram_range = self.ngram_range
        c.pos_tags = self.pos_tags
        c.name = self.name
        c.used_preprocessor = self.used_preprocessor
        c._titles = self._titles
        c._pp_documents = self._pp_documents
        return c

    @staticmethod
    def from_documents(documents, name, attributes=None, class_vars=None, metas=None,
                       title_indices=None):
        """
        Create corpus from documents.

        Args:
            documents (list): List of documents.
            name (str): Name of the corpus
            attributes (list): List of tuples (Variable, getter) for attributes.
            class_vars (list): List of tuples (Variable, getter) for class vars.
            metas (list): List of tuples (Variable, getter) for metas.
            title_indices (list): List of indices into domain corresponding to features which will
                be used as titles.

        Returns:
            Corpus.
        """
        attributes = attributes or []
        class_vars = class_vars or []
        metas = metas or []
        title_indices = title_indices or []

        domain = Domain(attributes=[attr for attr, _ in attributes],
                        class_vars=[attr for attr, _ in class_vars],
                        metas=[attr for attr, _ in metas])

        for ind in title_indices:
            domain[ind].attributes['title'] = True

        def to_val(attr, val):
            if isinstance(attr, DiscreteVariable):
                attr.val_from_str_add(val)
            return attr.to_val(val)

        if documents:
            X = np.array([[to_val(attr, func(doc)) for attr, func in attributes]
                          for doc in documents], dtype=np.float64)
            Y = np.array([[to_val(attr, func(doc)) for attr, func in class_vars]
                          for doc in documents], dtype=np.float64)
            metas = np.array([[to_val(attr, func(doc)) for attr, func in metas]
                              for doc in documents], dtype=object)
        else:   # assure shapes match the number of columns
            X = np.empty((0, len(attributes)))
            Y = np.empty((0, len(class_vars)))
            metas = np.empty((0, len(metas)))

        corpus = Corpus(X=X, Y=Y, metas=metas, domain=domain, text_features=[])
        corpus.name = name
        return corpus

    def __getitem__(self, key):
        c = super().__getitem__(key)
        if isinstance(c, (Corpus, RowInstance)):
            Corpus.retain_preprocessing(self, c, key)
        return c

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        t = super().from_table(domain, source, row_indices)
        c = Corpus(t.domain, t.X, t.Y, t.metas, t.W, ids=t.ids)
        Corpus.retain_preprocessing(source, c, row_indices)
        return c

    @classmethod
    def from_numpy(cls, *args, **kwargs):
        c = super().from_numpy(*args, **kwargs)
        c._set_unique_titles()
        return c

    @classmethod
    def from_list(cls, domain, rows, weights=None):
        c = super().from_list(domain, rows, weights)
        c._set_unique_titles()
        return c

    @classmethod
    def from_table_rows(cls, source, row_indices):
        c = super().from_table_rows(source, row_indices)
        if hasattr(source, "_titles"):
            # covering case when from_table_rows called by from_table
            c._titles = source._titles[row_indices]
        return c

    @classmethod
    def from_file(cls, filename):
        if not os.path.exists(filename):  # check the default location
            abs_path = os.path.join(get_sample_corpora_dir(), filename)
            if not abs_path.endswith('.tab'):
                abs_path += '.tab'
            if not os.path.exists(abs_path):
                raise FileNotFoundError('File "{}" not found.'.format(filename))
            else:
                filename = abs_path

        table = Table.from_file(filename)
        corpus = cls(table.domain, table.X, table.Y, table.metas, table.W)
        return corpus

    @staticmethod
    def retain_preprocessing(orig, new, key=...):
        """ Set preprocessing of 'new' object to match the 'orig' object. """
        if isinstance(orig, Corpus):
            if isinstance(key, tuple):  # get row selection
                key = key[0]

            if orig._tokens is not None:  # retain preprocessing
                if isinstance(key, Integral):
                    new._tokens = np.array([orig._tokens[key]])
                    new.pos_tags = None if orig.pos_tags is None else np.array(
                        [orig.pos_tags[key]])
                elif isinstance(key, list) or isinstance(key, np.ndarray) \
                        or isinstance(key, slice) or isinstance(key, range):
                    new._tokens = orig._tokens[key]
                    new.pos_tags = None if orig.pos_tags is None else orig.pos_tags[key]
                elif key is Ellipsis:
                    new._tokens = orig._tokens
                    new.pos_tags = orig.pos_tags
                else:
                    raise TypeError('Indexing by type {} not supported.'.format(type(key)))
                new._dictionary = orig._dictionary

            if isinstance(new, Corpus):
                # _find_identical_feature returns non when feature not found
                # filter this Nones from list
                new.text_features = list(filter(None, [
                    new._find_identical_feature(tf)
                    for tf in orig.text_features
                ]))
            else:
                new.text_features = [
                    tf
                    for tf in orig.text_features
                    if tf in set(new.domain.metas)
                ]

            new._titles = orig._titles[key]
            new.ngram_range = orig.ngram_range
            new.attributes = orig.attributes
            new.used_preprocessor = orig.used_preprocessor

    def __eq__(self, other):
        def arrays_equal(a, b):
            if sp.issparse(a) != sp.issparse(b):
                return False
            elif sp.issparse(a) and sp.issparse(b):
                return (a != b).nnz == 0
            else:
                return np.array_equal(a, b)

        return (self.text_features == other.text_features and
                self._dictionary == other._dictionary and
                np.array_equal(self._tokens, other._tokens) and
                arrays_equal(self.X, other.X) and
                arrays_equal(self.Y, other.Y) and
                arrays_equal(self.metas, other.metas) and
                np.array_equal(self.pos_tags, other.pos_tags) and
                self.domain == other.domain and
                self.ngram_range == other.ngram_range)
