import os
from collections import Counter, defaultdict
from copy import copy
from numbers import Integral
from itertools import chain
from typing import Union, Optional, List, Tuple, Dict
from warnings import warn

import nltk
import numpy as np
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
from Orange.data.util import get_unique_names
from orangewidget.utils.signals import summarize, PartialSummary
import scipy.sparse as sp

from orangecontrib.text.language import ISO2LANG


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
    NGRAMS_SEPARATOR = " "

    def __new__(cls, *args, **kwargs):
        if args and isinstance(args[0], Domain) or "domain" in kwargs:
            warn(
                "Signature of Corpus constructor when called with numpy "
                "arrays will change in the future to be equal to Corpus.from_numpy. "
                "To avoid issues use Corpus.from_numpy instead.",
                FutureWarning,
            )
            # __init__ had a different signature than from_numpy
            params = ["domain", "X", "Y", "metas", "W", "text_features", "ids"]
            kwargs.update({param: arg for param, arg in zip(params, args)})

            # in old signature it can happen that X is missing
            n_doc = _check_arrays(
                kwargs.get("X", None), kwargs.get("Y", None), kwargs.get("metas", None)
            )
            if "X" not in kwargs:
                kwargs["X"] = np.empty((n_doc, 0))

            return cls.from_numpy(**kwargs)
        return super().__new__(cls, *args, **kwargs)

    def _setup_corpus(self, text_features: List[Variable] = None) -> None:
        """
        Parameters
        ----------
        text_features
            meta attributes that are used for text mining. Infer them if None.
        """
        self.text_features = []    # list of text features for mining
        self._tokens = None
        self.ngram_range = (1, 1)
        self._pos_tags = None
        from orangecontrib.text.preprocess import PreprocessorList
        self.__used_preprocessor = PreprocessorList([])   # required for compute values
        self._titles: Optional[np.ndarray] = None
        self._pp_documents = None  # preprocessed documents

        if text_features is None:
            self._infer_text_features()
        else:
            self.set_text_features(text_features)

        self._set_unique_titles()
        if "language" not in self.attributes:
            self.attributes["language"] = None

    @property
    def used_preprocessor(self):
        return self.__used_preprocessor

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
            if feats != self.text_features:
                # when new features are same than before it is not required
                # to invalidate tokens
                self.text_features = feats
                self._tokens = None  # invalidate tokens
            for attr in self.domain.metas:
                # update all include attributes
                if attr in feats:
                    attr.attributes['include'] = True
                else:
                    attr.attributes.pop('include', None)
        else:
            self._infer_text_features()

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
                incl = attr.attributes.get('include', False)
                # variable attributes can be boolean from Orange 3.29
                # they are string in older versions
                # incl == True, since without == string "False" would be True
                if incl == "True" or incl == True:
                    include_feats.append(attr)
        if len(include_feats) == 0 and first:
            include_feats.append(first)
        self.set_text_features(include_feats)

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
        curr_attributes, curr_class_var, curr_metas = _rename_features(feature_names)
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
            if isinstance(var_attrs, dict):
                var.attributes.update(var_attrs)
            additional_attributes.append(var)

        new_domain = Domain(
                attributes=curr_attributes + additional_attributes,
                class_vars=curr_class_var,
                metas=curr_metas
        )
        c = Corpus.from_numpy(
            new_domain,
            X,
            self.Y.copy(),
            self.metas.copy(),
            self.W.copy(),
            text_features=copy(self.text_features),
            attributes=self.attributes,
        )
        c.name = self.name  # keep corpus's name
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

    @property
    def language(self):
        return self.attributes["language"]

    def __setstate__(self, state: Dict):
        super().__setstate__(state)
        if "language" not in self.attributes:
            self.attributes["language"] = None

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

    def store_tokens(self, tokens: List):
        """
        Parameters
        ----------
        tokens
            List of lists containing tokens.
        """
        self._tokens = np.array(tokens, dtype=object)

    @property
    def tokens(self):
        """
        np.ndarray: A list of lists containing tokens. If tokens are not yet
        present, run default preprocessor and return tokens.
        """
        if self._tokens is None:
            return self._base_tokens()
        return self._tokens

    def has_tokens(self):
        """ Return whether corpus is preprocessed or not. """
        return self._tokens is not None

    def _base_tokens(self):
        from orangecontrib.text.preprocess import BASE_TRANSFORMER, \
            BASE_TOKENIZER, PreprocessorList

        # don't use anything that requires NLTK data to assure async download
        base_preprocessors = PreprocessorList([BASE_TRANSFORMER, BASE_TOKENIZER])
        corpus = base_preprocessors(self)
        return corpus.tokens

    @property
    def pos_tags(self):
        """
            np.ndarray: A list of lists containing POS tags. If there are no
            POS tags available, return None.
        """
        if self._pos_tags is None:
            return None
        return np.array(self._pos_tags, dtype=object)

    @pos_tags.setter
    def pos_tags(self, pos_tags):
        self._pos_tags = pos_tags

    def ngrams_iterator(self, join_with=NGRAMS_SEPARATOR, include_postags=False):
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

    def count_tokens(self) -> int:
        """Count number of all (non-unique) tokens in the corpus"""
        return sum(map(len, self.tokens))

    def count_unique_tokens(self) -> int:
        """Count number of all (unique) tokens in the corpus"""
        # it seems to be fast enough even datasets very large dataset, so I
        # would avoid caching to prevetnt potential problems connected to that
        return len({tk for lst in self.tokens for tk in lst})

    @property
    def ngrams(self):
        """generator: Ngram representations of documents."""
        return self.ngrams_iterator(join_with=self.NGRAMS_SEPARATOR)

    def copy(self):
        """Return a copy of the table."""
        c = super().copy()
        c._setup_corpus(text_features=copy(self.text_features))
        # since tokens are considered immutable copies are not needed
        c._tokens = self._tokens
        c.ngram_range = self.ngram_range
        c.pos_tags = self.pos_tags
        c.name = self.name
        c.used_preprocessor = self.used_preprocessor
        c._titles = self._titles
        c._pp_documents = self._pp_documents
        return c

    @staticmethod
    def from_documents(documents, name, attributes=None, class_vars=None, metas=None,
                       title_indices=None, language=None):
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
            language (str): Resulting corpus's language

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

        corpus = Corpus.from_numpy(
            domain=domain, X=X, Y=Y, metas=metas, text_features=[]
        )
        corpus.name = name
        corpus.attributes["language"] = language
        return corpus

    def __getitem__(self, key):
        c = super().__getitem__(key)
        if isinstance(c, (Corpus, RowInstance)):
            Corpus.retain_preprocessing(self, c, key)
        return c

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        c = super().from_table(domain, source, row_indices)
        c._setup_corpus()
        Corpus.retain_preprocessing(source, c, row_indices)
        return c

    @classmethod
    def from_numpy(
        cls,
        domain,
        X,
        Y=None,
        metas=None,
        W=None,
        attributes=None,
        ids=None,
        text_features=None,
        language=None
    ):
        t = super().from_numpy(
            domain, X, Y=Y, metas=metas, W=W, attributes=attributes, ids=ids
        )
        # t is corpus but corpus specific attributes were not set yet
        t._setup_corpus(text_features=text_features)
        # language can be already set in attributes if they provided
        if language is not None or "language" not in t.attributes:
            t.attributes["language"] = language
        return t

    @classmethod
    def from_list(cls, domain, rows, weights=None, language=None):
        t = super().from_list(domain, rows, weights)
        # t is corpus but corpus specific attributes were not set yet
        t._setup_corpus()
        t.attributes["language"] = language
        return t

    @classmethod
    def from_table_rows(cls, source, row_indices):
        c = super().from_table_rows(source, row_indices)
        # t is corpus but corpus specific attributes were not set yet
        c._setup_corpus()
        if hasattr(source, "_titles"):
            # covering case when from_table_rows called by from_table
            c._titles = source._titles[row_indices]
        return c

    @classmethod
    def from_file(cls, filename, sheet=None):
        if not os.path.exists(filename):  # check the default location
            abs_path = os.path.join(get_sample_corpora_dir(), filename)
            if not abs_path.endswith('.tab'):
                abs_path += '.tab'
            if os.path.exists(abs_path):
                filename = abs_path

        table = super().from_file(filename, sheet=sheet)
        if not isinstance(table, Corpus):
            # when loading regular file result of super().from_file is Table - need
            # to be transformed to Corpus, when loading pickle it is Corpus already
            name = table.name
            table = cls.from_numpy(table.domain, table.X, table.Y, table.metas, table.W, attributes=table.attributes)
            table.name = name
        return table

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
            new.used_preprocessor = orig.used_preprocessor
        else:  # orig is not Corpus
            new._set_unique_titles()
            new._infer_text_features()


@summarize.register(Corpus)
def summarize_corpus(corpus: Corpus) -> PartialSummary:
    """
    Provides automated input and output summaries for Corpus
    """
    table_summary = summarize.dispatch(Table)(corpus)
    extras = (
        (
            f"<br/><nobr>Tokens: {corpus.count_tokens()}, "
            f"Types: {corpus.count_unique_tokens()}</nobr>"
        )
        if corpus.has_tokens()
        else "<br/><nobr>Corpus is not preprocessed</nobr>"
    )
    language = ISO2LANG[corpus.language] if corpus.language else "not set"
    extras += f"<br/><nobr>Language: {language}</nobr>"
    return PartialSummary(table_summary.summary, table_summary.details + extras)
