import numpy as np
import wikipedia
import threading
from Orange import data
from orangecontrib.text.corpus import Corpus


class NetworkException(IOError, wikipedia.exceptions.HTTPTimeoutError):
    pass


class WikipediaAPI:
    """ Wraps Wikipedia API.

    Examples:
        >>> api = WikipediaAPI()
        >>> corpus = api.search('en', ['Barack Obama', 'Hillary Clinton'])
    """
    attributes = ('pageid', 'revision_id')
    metas = ('title', 'content', 'summary', 'url', 'query')

    def __init__(self, on_progress=None, on_error=None, on_finish=None):
        super().__init__()
        self.thread = None
        self.running = False

        self.on_progress = on_progress or (lambda x, y: x)
        self.on_error = on_error or (lambda x: x)
        self.on_finish = on_finish or (lambda x: x)

    def search(self, lang, queries, attributes, articles_per_query=10, async=False):
        """ Searches for articles.

        Args:
            lang(str): A language code in ISO 639-1 format.
            queries(list of str): A list of queries.
        """
        if async:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError('You cannot run several threads at the same time')
            self.thread = threading.Thread(target=self.search,
                                           args=(lang, queries, attributes, articles_per_query, False))
            self.thread.daemon = True
            self.thread.start()
            return

        self.running = True
        wikipedia.set_lang(lang)
        metas = [attr for attr in attributes if attr in self.metas] + ['content']
        attributes = [attr for attr in attributes if attr in self.attributes]

        X, meta_values = [], []
        for i, query in enumerate(queries):
            try:
                articles = wikipedia.search(query, results=articles_per_query)
                for j, article in enumerate(articles):
                    self._get(article, attributes, X, metas, meta_values, query)
                    if not self.running:
                        break
                    self.on_progress(100 * (i * len(articles) + j + 1) / (len(queries) * len(articles)),
                                     len(X))
            except (wikipedia.exceptions.HTTPTimeoutError, IOError) as e:
                self.on_error(NetworkException(e))

        metas = [data.StringVariable(attr) for attr in metas]
        domain = data.Domain(attributes=[], metas=metas)
        corpus = Corpus(None, metas=np.array(meta_values, dtype=object), domain=domain, text_features=metas[-1:])
        corpus.extend_attributes(np.array(X), attributes)
        self.on_finish(corpus)
        self.running = False
        return corpus

    def _get(self, article, attributes, X, metas, meta_values, query, recursive=True):
        try:
            if not self.running:
                return

            article = wikipedia.page(article)
            article.query = query

            X.append(
                [int(getattr(article, attr)) for attr in attributes]
                # [getattr(article, attr) for attr in attributes]
            )
            meta_values.append(
                [getattr(article, attr) for attr in metas]
            )
            return True

        except wikipedia.exceptions.DisambiguationError:
            if recursive:
                for article in wikipedia.search(article, 10):
                    if self._get(article, attributes, X, metas, meta_values, query, recursive=False):
                        break

        except wikipedia.exceptions.PageError:
            pass

    def disconnect(self):
        self.running = False
        if self.thread:
            self.thread.join()
