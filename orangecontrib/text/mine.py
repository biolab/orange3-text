import mine

from Orange import data
from orangecontrib.text import Corpus


class NetworkException(IOError, mine.exceptions.HTTPTimeoutError):
    pass


class MineAPI:
    """ Wraps Mine API.

    Examples:
        >>> api = MineAPI()
        >>> corpus = api.search('en', ['Barack Obama', 'Hillary Clinton'])
    """
    metas = [
        (data.StringVariable('Title'), lambda doc: getattr(doc, 'titlett')),
        (data.StringVariable('Abstract'), lambda doc: getattr(doc, 'abstract')),
        #(data.StringVariable('Url'), lambda doc: getattr(doc, 'url')),
        (data.StringVariable('Authors'), lambda doc: getattr(doc, 'authors')),
        (data.StringVariable('Date'), lambda doc: getattr(doc, 'date')),
        (data.StringVariable('Format'), lambda doc: getattr(doc, 'format')),
        #(data.ContinuousVariable('Page ID', number_of_decimals=0), lambda doc: int(getattr(doc, 'pageid'))),
        #(data.ContinuousVariable('Revision ID', number_of_decimals=0), lambda doc: int(getattr(doc, 'revision_id'))),
        (data.StringVariable('Resource'), lambda doc: getattr(doc, 'resource')),
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
       
        mine.set_lang(lang)

        results = []
        for i, query in enumerate(queries):
            try:
                articles_tmp = mine.search(query, results=articles_per_query)
                articles = articles_tmp
               
                for j, article in enumerate(articles):
                    if callable(should_break) and should_break():
                        break
                    
                    results.extend(self._get(article, query, should_break))

                    if callable(on_progress):
                        on_progress((i*articles_per_query + j+1) / (len(queries) * articles_per_query),
                                    len(results))
            except (mine.exceptions.HTTPTimeoutError, IOError) as e:
                self.on_error(str(e))
                break

            if callable(should_break) and should_break():
                break
        
        
        tempc = Corpus.from_documents(results, 'Wikpedia', self.attributes,
                                     self.class_vars, self.metas, title_indices=[-1])
        
        return tempc
        
        
    def _get(self, article, query, should_break, recursive=True):
        try:
            article = mine.page(article)
           
            article.query = query
            
            return [article]
        except mine.exceptions.DisambiguationError:
            res = []
            if recursive:
                for article in mine.search(article, 20):
                    if callable(should_break) and should_break():
                        break
                    res.extend(self._get(article, query, should_break, recursive=False))
                    
            return res

        except mine.exceptions.PageError:
            
            return []
