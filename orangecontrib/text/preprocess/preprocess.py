from orangecontrib.text.preprocess import (
    FrequencyFilter, LowercaseTransformer, WordPunctTokenizer)


__all__ = ['Preprocessor', 'base_preprocessor']


# don't use anything that requires NLTK data to assure async download
BASE_TOKENIZER = WordPunctTokenizer()
BASE_TRANSFORMERS = [LowercaseTransformer()]


class Preprocessor:
    """Holds document processing objects.

    Attributes:
        transformers (List([BaseTransformer]): transforms strings
        tokenizer (BaseTokenizer): tokenizes string
        normalizer (BaseNormalizer): normalizes tokens
        filters (List[BaseTokenFilter]): filters unneeded tokens
    """

    def __init__(self, transformers=None, tokenizer=None,
                 normalizer=None, filters=None, ngrams_range=None, pos_tagger=None):

        if callable(transformers):
            transformers = [transformers]

        if callable(filters):
            filters = [filters]

        self.transformers = transformers or []
        self.tokenizer = tokenizer
        self.filters = filters or []
        self.normalizer = normalizer
        self.ngrams_range = ngrams_range
        self.pos_tagger = pos_tagger

        self.progress = 0
        self._report_frequency = 1

    def __call__(self, corpus, inplace=True, on_progress=None):
        """ Runs preprocessing over a corpus.

        Args:
            corpus(orangecontrib.text.Corpus): A corpus to preprocess.
            inplace(bool): Whether to create a new Corpus instance.
        """
        self.set_up()
        self._on_progress = on_progress
        if not inplace:
            corpus = corpus.copy()

        self.progress = 1
        self._report_frequency = len(corpus) // 80 or 1
        self._len = len(corpus) / 80
        tokens = list(map(self.process_document, corpus.documents))
        corpus.store_tokens(tokens)
        self.on_progress(80)
        if self.ngrams_range is not None:
            corpus.ngram_range = self.ngrams_range
        if self.freq_filter is not None:
            tokens, dictionary = self.freq_filter.fit_filter(corpus)
            corpus.store_tokens(tokens, dictionary)

        if self.pos_tagger:
            self.pos_tagger.tag_corpus(corpus)

        self.on_progress(100)
        corpus.used_preprocessor = self
        corpus.used_preprocessor._on_progress = None    # remove on_progress that is causing pickling problems
        self.tear_down()
        return corpus

    @property
    def filters(self):
        return self._filters

    @filters.setter
    def filters(self, filters):
        self._filters = []
        self.freq_filter = None

        for f in filters:
            if isinstance(f, FrequencyFilter):
                self.freq_filter = f
            else:
                self._filters.append(f)

    def process_document(self, document):
        for transformer in self.transformers:
            document = transformer.transform(document)

        if self.tokenizer:
            tokens = self.tokenizer.tokenize(document)
        else:
            tokens = BASE_TOKENIZER.tokenize(document)

        if self.normalizer:
            if getattr(self.normalizer, 'use_tokenizer', False):
                tokens = self.normalizer.normalize_doc(document)
            else:
                tokens = self.normalizer(tokens)

        for filter in self.filters:
            tokens = filter(tokens)

        self.progress += 1
        if self.progress % self._report_frequency == 0:
            self.on_progress(self.progress / self._len)
        return tokens

    def on_progress(self, progress):
        if self._on_progress:
            self._on_progress(progress)

    def set_up(self):
        """ Called before every __call__. Used for setting up tokenizer & filters. """
        if self.tokenizer:
            self.tokenizer.set_up()

        for f in self.filters:
            f.set_up()

    def tear_down(self):
        """ Called after every __call__. Used for cleaning up tokenizer & filters. """
        if self.tokenizer:
            self.tokenizer.tear_down()

        for f in self.filters:
            f.tear_down()

    def __str__(self):
        return '\n'.join(['{}: {}'.format(name, value) for name, value in self.report()])

    def report(self):
        if getattr(self.normalizer, 'use_tokenizer', False):
            self.tokenizer = \
                'UDPipe Tokenizer ({})'.format(self.normalizer.language)
        rep = (
            ('Transformers', ', '.join(str(tr) for tr in self.transformers)
            if self.transformers else None),
            ('Tokenizer', str(self.tokenizer) if self.tokenizer else None),
            ('Normalizer', str(self.normalizer) if self.normalizer else None),
            ('Filters', ', '.join(str(f) for f in self.filters) if
            self.filters else None),
            ('Ngrams range', str(self.ngrams_range) if self.ngrams_range else
            None),
            ('Frequency filter', str(self.freq_filter) if self.freq_filter
            else None),
            ('Pos tagger', str(self.pos_tagger) if self.pos_tagger else None),
        )
        del self.tokenizer
        return rep


base_preprocessor = Preprocessor(transformers=BASE_TRANSFORMERS,
                                 tokenizer=BASE_TOKENIZER)
