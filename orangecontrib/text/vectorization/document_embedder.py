"""This module contains classes used for embedding documents
into a vector space.
"""
import base64
import json
import sys
import warnings
import zlib
from typing import Any, Optional, Tuple

import numpy as np
from Orange.misc.server_embedder import ServerEmbedderCommunicator
from Orange.util import dummy_callback

from orangecontrib.text import Corpus
from orangecontrib.text.vectorization.base import BaseVectorizer

AGGREGATORS = ['Mean', 'Sum', 'Max', 'Min']
AGGREGATORS_L = ['mean', 'sum', 'max', 'min']
LANGS_TO_ISO = {
    'English': 'en',
    'Slovenian': 'sl',
    'German': 'de',
    'Arabic': 'ar',
    'Azerbaijani': 'az',
    'Bengali': 'bn',
    'Chinese': 'zh',
    'Danish': 'da',
    'Dutch': 'nl',
    'Finnish': 'fi',
    'French': 'fr',
    'Greek': 'el',
    'Hebrew': 'he',
    'Hindi': 'hi',
    'Hungarian': 'hu',
    'Indonesian': 'id',
    'Italian': 'it',
    'Japanese': 'ja',
    'Kazakh': 'kk',
    'Korean': 'ko',
    'Nepali': 'ne',
    'Norwegian (Bokm\u00e5l)': 'no',
    'Norwegian (Nynorsk)': 'nn',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Romanian': 'ro',
    'Russian': 'ru',
    'Spanish': 'es',
    'Swedish': 'sv',
    'Tajik': 'tg',
    'Turkish': 'tr'
}
LANGUAGES = list(LANGS_TO_ISO.values())


class DocumentEmbedder(BaseVectorizer):
    """This class is used for obtaining dense embeddings of documents in
    corpus using fastText pretrained models from:
    E. Grave, P. Bojanowski, P. Gupta, A. Joulin, T. Mikolov,
    Learning Word Vectors for 157 Languages.
    Proceedings of the International Conference on Language Resources and
    Evaluation, 2018.

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
                                         server_url='https://api.garaza.io',
                                         embedder_type='text')

    def _transform(
        self, corpus: Corpus, _, callback=dummy_callback
    ) -> Tuple[Corpus, Corpus]:
        """Adds matrix of document embeddings to a corpus.

        Parameters
        ----------
        corpus : Corpus or list of lists
            Corpus on which transform is performed.

        Returns
        -------
        Embeddings
            Corpus (original or a copy) with new features added.
        Skipped documents
            Corpus of documents that were not embedded
        """
        embs = self._embedder.embedd_data(
            list(corpus.ngrams) if isinstance(corpus, Corpus) else corpus,
            callback=callback,
        )

        if isinstance(corpus, list):
            return embs

        dim = None
        for emb in embs:  # find embedding dimension
            if emb is not None:
                dim = len(emb)
                break
        # Check if some documents in corpus in weren't embedded
        # for some reason. This is a very rare case.
        skipped_documents = [emb is None for emb in embs]
        embedded_documents = np.logical_not(skipped_documents)

        new_corpus = None
        if np.any(embedded_documents):
            # if at least one embedding is not None, extend attributes
            new_corpus = corpus[embedded_documents]
            new_corpus = new_corpus.extend_attributes(
                np.array(
                    [e for e, ns in zip(embs, embedded_documents) if ns],
                    dtype=float,
                ),
                ["Dim{}".format(i + 1) for i in range(dim)],
                var_attrs={
                    "embedding-feature": True,
                    "hidden": True,
                },
            )

        skipped_corpus = None
        if np.any(skipped_documents):
            skipped_corpus = corpus[skipped_documents].copy()
            skipped_corpus.name = "Skipped documents"
            warnings.warn(
                "Some documents were not embedded for unknown reason. Those "
                "documents are skipped.",
                RuntimeWarning,
            )

        return new_corpus, skipped_corpus

    def report(self) -> Tuple[Tuple[str, str], ...]:
        """Reports on current parameters of DocumentEmbedder.

        Returns
        -------
        tuple
            Tuple of parameters.
        """
        return (
            ("Embedder", "fastText"),
            ("Language", self.language),
            ("Aggregator", self.aggregator),
        )

    def clear_cache(self):
        """Clears embedder cache"""
        if self._embedder:
            self._embedder.clear_cache()


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
        if sys.getsizeof(data) > 500000:
            # Document in corpus is too large. Size limit is 500 KB
            # (after compression). - document skipped
            return None

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
