"""This module contains classes used for embedding documents
into a vector space.
"""
import zlib
import base64
import json
import sys
import warnings
from typing import Tuple, Any, Optional
import numpy as np

from Orange.misc.server_embedder import ServerEmbedderCommunicator
from orangecontrib.text import Corpus


AGGREGATORS = ['Mean', 'Sum', 'Max', 'Min']
AGGREGATORS_L = ['mean', 'sum', 'max', 'min']
LANGS_TO_ISO = {'English': 'en', 'Slovenian': 'sl', 'German': 'de'}
LANGUAGES = list(LANGS_TO_ISO.values())


class DocumentEmbedder:
    """This class is used for obtaining dense embeddings of documents in
    corpus using fastText pretrained models from:
    E. Grave, P. Bojanowski, P. Gupta, A. Joulin, T. Mikolov,
    Learning Word Vectors for 157 Languages.
    Proceedings of the International Conference on Language Resources and Evaluation, 2018.

    Embedding is performed on server so the internet connection is a
    prerequisite for using the class. Currently supported languages are:
        - English (en)
        - Slovenian (sl)
        - German (de)

    Attributes
    ----------
    language : str
        ISO 639-1 (two-letter) code of desired language.
    aggregator : str
        Aggregator which creates document embedding (single
        vector) from word embeddings (multiple vectors).
        Allowed values are Mean, Sum, Max, Min.
    """

    def __init__(self, language: str = 'en',
                 aggregator: str = 'Mean') -> None:
        lang_error = '{} is not a valid language. Allowed values: {}'
        agg_error = '{} is not a valid aggregator. Allowed values: {}'
        if language.lower() not in LANGUAGES:
            raise ValueError(lang_error.format(language, ', '.join(LANGUAGES)))
        self.language = language.lower()
        if aggregator.lower() not in AGGREGATORS_L:
            raise ValueError(agg_error.format(aggregator, ', '.join(AGGREGATORS)))
        self.aggregator = aggregator.lower()

        self._embedder = _ServerEmbedder(self.aggregator,
                                         model_name='fasttext-'+self.language,
                                         max_parallel_requests=100,
                                         server_url='https://apiv2.garaza.io',
                                         embedder_type='text')

    def __call__(self, corpus: Corpus, copy: bool = True,
                 processed_callback=None) -> Corpus:
        """Adds matrix of document embeddings to a corpus.

        Parameters
        ----------
        corpus : Corpus
            Corpus on which transform is performed.
        copy : bool
            If set to True, a copy of corpus is made.

        Returns
        -------
        Corpus
            Corpus (original or a copy) with new features added.

        Raises
        ------
        ValueError
            If corpus is not instance of Corpus.
        RuntimeError
            If document in corpus is larger than
            50 KB after compression.
        """
        if not isinstance(corpus, Corpus):
            raise ValueError("Input should be instance of Corpus.")
        corpus = corpus.copy() if copy else corpus
        embs = self._embedder.embedd_data(
            list(corpus.ngrams),
            processed_callback=processed_callback)

        dim = None
        send_warning = False
        for emb in embs:  # find embedding dimension
            if emb is not None:
                dim = len(emb)
                break
        # Check if some documents in corpus in weren't embedded
        # for some reason. This is a very rare case.
        inds = list()
        for i, emb in enumerate(embs):
            if emb is not None:
                inds.append(i)
            else:
                embs[i] = np.zeros(dim) * np.nan
                send_warning = True

        variable_attrs = {
            'hidden': True,
            'skip-normalization': True,
            'embedding-feature': True
        }
        embs = np.array(embs)
        new_corpus = corpus[inds]

        if len(inds) > 0:
            # if at least one embedding is not None,
            # extend attributes
            new_corpus.extend_attributes(
                np.array(embs[inds]),
                ['Dim{}'.format(i + 1) for i in range(dim)],
                var_attrs=variable_attrs)

        if send_warning:
            warnings.warn(("Some documents were not embedded for " +
                           "unknown reason. Those documents " +
                           "are skipped."),
                          RuntimeWarning)

        return new_corpus

    def report(self) -> Tuple[Tuple[str, str], Tuple[str, str]]:
        """Reports on current parameters of DocumentEmbedder.

        Returns
        -------
        tuple
            Tuple of parameters.
        """
        return (('Language', self.language),
                ('Aggregator', self.aggregator))

    def set_cancelled(self):
        """Cancels current embedding process"""
        if hasattr(self, '_embedder'):
            self._embedder.set_cancelled()

    def clear_cache(self):
        """Clears embedder cache"""
        if self._embedder:
            self._embedder.clear_cache()

    def __enter__(self):
        return self

    def __exit__(self, ex_type, value, traceback):
        self.set_cancelled()

    def __del__(self):
        self.__exit__(None, None, None)


class _ServerEmbedder(ServerEmbedderCommunicator):
    def __init__(self, aggregator: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.content_type = 'application/json'
        self.aggregator = aggregator

    async def _encode_data_instance(self, data_instance: Any) -> Optional[bytes]:
        data_string = json.dumps(list(data_instance))
        data = base64.b64encode(zlib.compress(
            data_string.encode('utf-8', 'replace'),
            level=-1)).decode('utf-8', 'replace')

        if sys.getsizeof(data) > 50000:
            raise RuntimeError("Document in corpus is too large. \
                                Size limit is 50 KB (after compression).")

        data_dict = {
            "data": data,
            "aggregator": self.aggregator
        }

        json_string = json.dumps(data_dict)
        return json_string.encode('utf-8', 'replace')


if __name__ == '__main__':
    with DocumentEmbedder(language='en', aggregator='Max') as embedder:
        embedder.clear_cache()
        embedder(Corpus.from_file('deerwester'))
