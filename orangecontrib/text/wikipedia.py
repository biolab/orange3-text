import numpy as np
import wikipedia
from Orange import data

from orangecontrib.text.corpus import Corpus


class WikipediaAPI:
    attributes = ('pageid', 'revision_id')
    metas = ('title', 'content', 'summary', 'url')

    @classmethod
    def search(cls, lang, queries, attributes, progress_callback=None):
        wikipedia.set_lang(lang)
        attributes = [attr for attr in attributes if attr in cls.attributes]
        metas = [attr for attr in attributes if attr in cls.metas] + ['content']

        X, meta_values = [], []
        for i, query in enumerate(queries):
            articles = wikipedia.search(query)
            for j, article in enumerate(articles):
                cls._get(article, attributes, X, metas, meta_values)
                if progress_callback:
                    progress_callback(100 * (i * len(articles) + j + 1) / (len(queries) * len(articles)))
        metas = [data.StringVariable(attr) for attr in metas]
        domain = data.Domain(attributes=[], metas=metas)
        corpus = Corpus(None, metas=np.array(meta_values, dtype=object), domain=domain, text_features=metas[-1:])
        corpus.extend_attributes(np.array(X), attributes)
        return corpus

    @classmethod
    def _get(cls, article, attributes, X, metas, meta_values, recursive=True):
        try:
            article = wikipedia.page(article)

            X.append(
                [int(getattr(article, attr)) for attr in attributes]
                # [getattr(article, attr) for attr in attributes]
            )
            meta_values.append(
                [getattr(article, attr) for attr in metas]
            )

        except wikipedia.exceptions.DisambiguationError:
            if recursive:
                for article in wikipedia.search(article):
                    cls._get(article, attributes, X, metas, meta_values, recursive=False)

        except wikipedia.exceptions.PageError:
            pass
