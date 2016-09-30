import wikipedia
import threading

from Orange import data
from orangecontrib.text import Corpus


class NetworkException(IOError, wikipedia.exceptions.HTTPTimeoutError):
    pass


class WikipediaAPI:
    """ Wraps Wikipedia API.

    Examples:
        >>> api = WikipediaAPI()
        >>> corpus = api.search('en', ['Barack Obama', 'Hillary Clinton'])
    """
    metas = [
        (data.StringVariable('Title'), lambda doc: getattr(doc, 'title')),
        (data.StringVariable('Content'), lambda doc: getattr(doc, 'content')),
        (data.StringVariable('Summary'), lambda doc: getattr(doc, 'summary')),
        (data.StringVariable('Url'), lambda doc: getattr(doc, 'url')),
        (data.ContinuousVariable('Page ID'), lambda doc: int(getattr(doc, 'pageid'))),
        (data.ContinuousVariable('Revision ID'), lambda doc: int(getattr(doc, 'revision_id'))),
        (data.DiscreteVariable('Query'), lambda doc: getattr(doc, 'query')),
    ]

    attributes = []
    class_vars = []
    text_features = [m for m, _ in metas]
    string_attributes = [m for m, _ in metas if isinstance(m, data.StringVariable)]

    def __init__(self, on_progress=None, on_error=None, on_finish=None):
        super().__init__()
        self.thread = None
        self.running = False

        self.on_progress = on_progress or (lambda x, y: x)
        self.on_error = on_error or (lambda x: x)
        self.on_finish = on_finish or (lambda x: x)

    def search(self, lang, queries, articles_per_query=10, async=False):
        """ Searches for articles.

        Args:
            lang(str): A language code in ISO 639-1 format.
            queries(list of str): A list of queries.
        """
        if async:
            if self.thread is not None and self.thread.is_alive():
                raise RuntimeError('You cannot run several threads at the same time')
            self.thread = threading.Thread(target=self.search,
                                           args=(lang, queries, articles_per_query, False))
            self.thread.daemon = True
            self.thread.start()
            return

        self.running = True
        wikipedia.set_lang(lang)

        results = []
        for i, query in enumerate(queries):
            try:
                articles = wikipedia.search(query, results=articles_per_query)
                for j, article in enumerate(articles):
                    results.extend(self._get(article, query))
                    if not self.running:
                        break
                    self.on_progress(100 * (i * len(articles) + j + 1) / (len(queries) * len(articles)),
                                     len(results))
            except (wikipedia.exceptions.HTTPTimeoutError, IOError) as e:
                self.on_error(NetworkException(e))

        corpus = Corpus.from_documents(results, 'Wikipedia', self.attributes,
                                       self.class_vars, self.metas, title_indices=[-1])
        self.on_finish(corpus)
        self.running = False
        return corpus

    def _get(self, article, query, recursive=True):
        try:
            if not self.running:
                return []

            article = wikipedia.page(article)
            article.query = query
            return [article]
        except wikipedia.exceptions.DisambiguationError:
            res = []
            if recursive:
                for article in wikipedia.search(article, 10):
                    res.extend(self._get(article, query, recursive=False))
            return res

        except wikipedia.exceptions.PageError:
            return []

    def disconnect(self):
        self.running = False
        if self.thread:
            self.thread.join()
