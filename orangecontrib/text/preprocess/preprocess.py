
__all__ = ['Preprocessor']


class Preprocessor:
    """Holds document processing objects.

    Attributes:
        transformers (List([BaseTransformer]): transforms strings
        tokenizer (BaseTokenizer): tokenizes string
        normalizer (BaseNormalizer): normalizes tokens
        filters (List[BaseTokenFilter]): filters unneeded tokens
    """

    def __init__(self, transformers=None, tokenizer=None,
                 normalizer=None, filters=None, on_progress=None):

        self._on_progress = on_progress
        if callable(transformers):
            transformers = [transformers]

        if callable(filters):
            filters = [filters]

        self.transformers = transformers or []
        self.tokenizer = tokenizer
        self.filters = filters or []
        self.normalizer = normalizer

    def __call__(self, corpus):
        documents = corpus.documents
        for i, transformer in enumerate(self.transformers):
            documents = list(map(transformer, documents))
            self.on_progress((i+1) * 25 / len(self.transformers))
        self.on_progress(25)

        if self.tokenizer:
            tokens = list(map(self.tokenizer, documents))
        else:
            tokens = [documents]
        self.on_progress(50)

        if self.normalizer:
            tokens = list(map(self.normalizer, tokens))
        self.on_progress(75)

        for i, filter in enumerate(self.filters):
            # some filters may need the corpus
            filter.corpus = tokens
            tokens = filter(tokens)
            self.on_progress(75 + (i+1) * 25 / len(self.filters))

        corpus.store_tokens(tokens)
        self.on_progress(100)
        return corpus

    def on_progress(self, progress):
        if self._on_progress:
            self._on_progress(progress)

    def __str__(self):
        return '\n'.join(['{}: {}'.format(name, value) for name, value in self.report()])

    def report(self):
        return (
            ('Transformers', ', '.join(str(tr) for tr in self.transformers)),
            ('Tokenizer', str(self.tokenizer)),
            ('Filters', ', '.join(str(f) for f in self.filters)),
            ('Normalizer', str(self.normalizer)),
        )
