from typing import Union, List, Callable

from Orange.util import dummy_callback, wrap_callback

from orangecontrib.text import Corpus

__all__ = ['Preprocessor', 'TokenizedPreprocessor',
           'NGrams', 'PreprocessorList']


class Preprocessor:
    name = NotImplemented

    def __call__(self, corpus: Corpus) -> Corpus:
        """
         Preprocess corpus. Should be extended when inherited and
         invoke _preprocess method on a document or token(s).

        :param corpus: Corpus
        :return: Corpus
            Preprocessed corpus.
        """
        ids = corpus.ids
        corpus = corpus.copy()
        corpus.ids = ids
        corpus.used_preprocessor = self
        return corpus

    def __str__(self):
        return self.name

    def _store_documents(self, corpus: Corpus, callback: Callable) -> Corpus:
        """
        Preprocess and set corpus.documents.

        :param corpus: Corpus
        :param corpus: progress callback function
        :return: Corpus
            Preprocessed corpus.
        """
        assert callback is not None
        docs, n = [], len(corpus.pp_documents)
        for i, doc in enumerate(corpus.pp_documents):
            callback(i / n)
            docs.append(self._preprocess(doc))
        corpus.pp_documents = docs
        return corpus

    def _store_tokens(self, corpus: Corpus, callback: Callable) -> Corpus:
        """
        Preprocess and set corpus.tokens.

        :param corpus: Corpus
        :param callback: progress callback function
        :return: Corpus
            Preprocessed corpus.
        """
        assert callback is not None
        assert corpus.has_tokens()
        tokens, n = [], len(corpus.tokens)
        for i, tokens_ in enumerate(corpus.tokens):
            callback(i / n)
            tokens.append([self._preprocess(s) for s in tokens_])
        corpus.store_tokens(tokens)
        return corpus

    def _store_tokens_from_documents(self, corpus: Corpus,
                                     callback: Callable) -> Corpus:
        """
        Create tokens from documents and set corpus.tokens.

        :param corpus: Corpus
        :param callback: progress callback function
        :return: Corpus
            Preprocessed corpus.
        """
        assert callback is not None
        tokens, n = [], len(corpus.pp_documents)
        for i, doc in enumerate(corpus.pp_documents):
            callback(i / n)
            tokens.append(self._preprocess(doc))
        corpus.store_tokens(tokens)
        return corpus

    def _preprocess(self, _: Union[str, List[str]]) -> Union[str, List[str]]:
        """ This method should be implemented when subclassed. It performs
        preprocessing operation on a document or token(s).
        """
        raise NotImplementedError


class TokenizedPreprocessor(Preprocessor):
    def __call__(self, corpus: Corpus, callback: Callable) -> Corpus:
        corpus = super().__call__(corpus)
        if not corpus.has_tokens():
            from orangecontrib.text.preprocess import BASE_TOKENIZER
            corpus = BASE_TOKENIZER(corpus, callback)
        return corpus


class NGrams(TokenizedPreprocessor):
    name = "N-grams Range"

    def __init__(self, ngrams_range=(1, 2)):
        super().__init__()
        self.__range = ngrams_range

    def __call__(self, corpus: Corpus, callback: Callable = None) -> Corpus:
        corpus = super().__call__(corpus, callback)
        assert corpus.has_tokens()
        corpus.ngram_range = self.__range
        return corpus


class PreprocessorList:
    """ Store a list of preprocessors and on call apply them to the corpus. """

    def __init__(self, preprocessors: List):
        self.preprocessors = preprocessors

    def __call__(self, corpus: Corpus, callback: Callable = None) \
            -> Corpus:
        """
        Applies a list of preprocessors to the corpus.

        :param corpus: Corpus
        :param callback: progress callback function
        :return: Corpus
            Preprocessed corpus.
        """
        if callback is None:
            callback = dummy_callback
        n_pps = len(list(self.preprocessors))
        for i, pp in enumerate(self.preprocessors):
            start = i / n_pps
            cb = wrap_callback(callback, start=start, end=start + 1 / n_pps)
            corpus = pp(corpus, cb)
        callback(1)
        return corpus
