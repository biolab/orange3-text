import wikipedia

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

    def __init__(self, on_error=None):
        super().__init__()
        self.on_error = on_error or (lambda x: x)

    def search(self, lang, queries, articles_per_query=10, should_break=None, on_progress=None):
        """ Searches for articles.

        Args:
            lang(str): A language code in ISO 639-1 format.
            queries(list of str): A list of queries.
            should_break (callback): Callback for breaking the computation before the end.
                If it evaluates to True, downloading is stopped and document downloaded till now
                are returned in a Corpus.
            on_progress (callable): Callback for progress bar.
        """
        wikipedia.set_lang(lang)

        results = []
        for i, query in enumerate(queries):
            try:
                articles = wikipedia.search(query, results=articles_per_query)
                for j, article in enumerate(articles):
                    if callable(should_break) and should_break():
                        break

                    results.extend(self._get(article, query, should_break))

                    if callable(on_progress):
                        on_progress((i*articles_per_query + j+1) / (len(queries) * articles_per_query),
                                    len(results))
            except (wikipedia.exceptions.HTTPTimeoutError, IOError) as e:
                self.on_error(str(e))
                break

            if callable(should_break) and should_break():
                break

        return Corpus.from_documents(results, 'Wikipedia', self.attributes,
                                     self.class_vars, self.metas, title_indices=[-1])

    def _get(self, article, query, should_break, recursive=True):
        try:
            article = wikipedia.page(article)
            article.query = query
            return [article]
        except wikipedia.exceptions.DisambiguationError:
            res = []
            if recursive:
                for article in wikipedia.search(article, 10):
                    if callable(should_break) and should_break():
                        break
                    res.extend(self._get(article, query, should_break, recursive=False))
            return res

        except wikipedia.exceptions.PageError:
            return []
