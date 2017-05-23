import json
import math
import os
import shelve
import warnings
from datetime import date
from time import sleep
from urllib import request, parse
from http.client import HTTPException
from urllib.error import HTTPError, URLError


from Orange import data
from Orange.canvas.utils import environ
from orangecontrib.text.corpus import Corpus

SLEEP = 1
TIMEOUT = 10
MAX_DOCS = 1000
BATCH_SIZE = 10
MIN_DATE = date(1851, 1, 1)
BASE_URL = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'


class NYT:
    """ Class for fetching records from the NYT API. """

    @staticmethod
    def keywords(doc, name):
        return ', '.join([kw.get('value')
                          for kw in doc.get('keywords', [])
                          if kw['name'] == name])

    attributes = []

    class_vars = [
        (data.DiscreteVariable('Section'), lambda doc: doc.get('section_name', None)),
    ]

    tv = data.TimeVariable('Publication Date')
    metas = [
        (data.StringVariable('Headline'), lambda doc: doc.get('headline', {}).get('main') or ''),
        (data.StringVariable('Abstract'), lambda doc: doc.get('abstract') or ''),
        (data.StringVariable('Snippet'), lambda doc: doc.get('snippet') or ''),
        (data.StringVariable('Lead Paragraph'), lambda doc: doc.get('lead_paragraph') or ''),
        (data.StringVariable('Subject Keywords'), lambda doc: NYT.keywords(doc, 'subject')),
        (data.StringVariable('URL'), lambda doc: doc.get('web_url') or ''),
        (data.StringVariable('Locations'), lambda doc: NYT.keywords(doc, 'glocations')),
        (data.StringVariable('Persons'), lambda doc: NYT.keywords(doc, 'persons')),
        (data.StringVariable('Organizations'), lambda doc: NYT.keywords(doc, 'organizations')),
        (data.StringVariable('Creative Works'), lambda doc: NYT.keywords(doc, 'creative_works')),
        (tv, lambda doc: NYT.tv.parse(doc.get('pub_date'))),
        (data.DiscreteVariable('Article Type'), lambda doc: doc.get('type_of_material', None)),
        (data.ContinuousVariable('Word Count', number_of_decimals=0), lambda doc: doc.get('word_count', None)),
    ]

    text_features = [metas[0][0], metas[1][0]]  # headline + abstract

    def __init__(self, api_key):
        """
        Args:
            api_key (str): NY Time API key.
        """
        self.api_key = api_key
        self.on_error = None
        self.on_rate_limit = None
        self.on_no_connection = None
        self.cache_path = None
        self._cache_init()

    def api_key_valid(self):
        """ Checks whether api key given at initialization is valid. """
        url = self._encode_url('test')
        try:
            with request.urlopen(url) as connection:
                if connection.getcode() == 200:
                    return True
        except (HTTPError, URLError, HTTPException):
            return False

    def search(self, query, date_from=None, date_to=None, max_docs=None,
               on_progress=None, should_break=None):
        """
        Args:
            query (str): Search query.
            date_from (date): Start date limit.
            date_to (date): End date limit.
            max_docs (int): Maximal number of documents returned.
            on_progress (callback): Called after every iteration of downloading.
            should_break (callback): Callback for breaking the computation before the end.
                If it evaluates to True, downloading is stopped and document downloaded till now
                are returned in a Corpus.

        Returns:
            Corpus: Search results.
        """
        if max_docs is None or max_docs > MAX_DOCS:
            max_docs = MAX_DOCS

        # TODO create corpus on the fly and extend, so it stops faster.
        records = []
        data, go_sleep = self._fetch_page(query, date_from, date_to, 0)
        if data is None:
            return None
        records.extend(data['response']['docs'])
        max_docs = min(data['response']['meta']['hits'], max_docs)
        if callable(on_progress):
            on_progress(len(records), max_docs)

        for page in range(1, math.ceil(max_docs/BATCH_SIZE)):
            if callable(should_break) and should_break():
                break

            if go_sleep:
                sleep(SLEEP)

            data, go_sleep = self._fetch_page(query, date_from, date_to, page)

            if data is None:
                break

            records.extend(data['response']['docs'])
            if callable(on_progress):
                on_progress(len(records), max_docs)

        if len(records) > max_docs:
            records = records[:max_docs]

        return Corpus.from_documents(records, 'NY Times', self.attributes,
                                     self.class_vars, self.metas, title_indices=[-1])

    def _cache_init(self):
        """ Initialize cache in Orange environment buffer dir. """
        path = os.path.join(environ.buffer_dir, "nytcache")
        try:
            if not os.path.exists(path):
                os.makedirs(path)
            self.cache_path = os.path.join(path, "query_cache")
        except OSError as e:
            warnings.warn('Could not initialize NYT cache: {}'.format(str(e)), RuntimeWarning)

    def _cache_fetch(self, url):
        """ Fetch URL from cache if present. """
        with shelve.open(self.cache_path) as cache:
            if url in cache.keys():
                return cache[url]
            else:
                return None

    def _cache_store(self, url, data):
        """ Store data for URL in cache. """
        with shelve.open(self.cache_path) as cache:
            cache[url] = data

    def _fetch_page(self, query, date_from, date_to, page):
        """ Fetch one page either from cache or web. """
        cache_url = self._encode_url(query, date_from, date_to, page, for_caching=True)
        data = self._cache_fetch(cache_url)
        if data:
            return data, False
        else:
            url = self._encode_url(query, date_from, date_to, page, for_caching=False)
            try:
                with request.urlopen(url, timeout=TIMEOUT) as conn:
                    data = conn.read().decode('utf-8')
            except HTTPError as e:
                if e.code == 403 and page > 0:
                    # occasionally some pages return error 403 (Forbidden)
                    # while all other page numbers seem to work just fine.
                    # Skip such pages and don't break loading!
                    warnings.warn('NYT api returned HTTPError with code 403 '
                                  '(Forbidden)! Skipping this page ...')
                    return {'response': {'docs': []}}, True
                if e.code == 429 and callable(self.on_rate_limit):
                    self.on_rate_limit()
                elif callable(self.on_error):
                    self.on_error(str(e))
                return None, False
            except URLError:
                if callable(self.on_no_connection):
                    self.on_no_connection()
                    return None, False
                raise
            data = json.loads(data)
            self._cache_store(cache_url, data)
            return data, True

    def _encode_url(self, query, date_from=None, date_to=None, page=0, for_caching=False):
        """
        Encode url for given query, date restrictions and page number.

        Args:
            query (str): Search query.
            date_from (date): Date restriction.
            date_to (date): Date restriction.
            page (int): Page number.
            for_caching (bool): Whether URL would be used for caching. If set, exclude BASE_URL
                and API key.

        Returns:
            str: An encoded URL.
        """
        params = [   # list required to preserve order - important for caching
            ('fq', 'The New York Times'),
            ('api-key', self.api_key),
            ('q', query),
            ('page', page),
        ]
        if date_from:
            params.append(('begin_date', date_from.strftime('%Y%m%d')))
        if date_to:
            params.append(('end_date', date_to.strftime('%Y%m%d')))

        if for_caching:     # remove api key, return only params
            del params[0]
            return parse.urlencode(params)
        else:
            return '{}?{}'.format(BASE_URL, parse.urlencode(params))
