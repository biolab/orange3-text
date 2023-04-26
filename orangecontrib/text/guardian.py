""" This module fetches data from The Guardian API.

To use first create :class:`TheGuardianCredentials`:

    >>> from orangecontrib.text.guardian import TheGuardianCredentials
    >>> credentials = TheGuardianCredentials('<your-api-key>')

Then create :class:`TheGuardianAPI` object and use it for searching:

    >>> from orangecontrib.text.guardian import TheGuardianAPI
    >>> api = TheGuardianAPI(credentials)
    >>> corpus = api.search('Slovenia', max_documents=10)
    >>> len(corpus)
    10

"""
import math
import json
import os
from functools import partial

import requests
from Orange.data import (
    StringVariable,
    DiscreteVariable,
    ContinuousVariable,
    TimeVariable,
)
from dateutil.parser import isoparse

from orangecontrib.text.util import create_corpus

BASE_URL = 'http://content.guardianapis.com/search'
ARTICLES_PER_PAGE = 10


class APILimitError(Exception):
    pass


class TheGuardianCredentials:
    """ The Guardian API credentials. """
    def __init__(self, key):
        """
        Args:
            key (str): The Guardian API key. Use `test` for testing purposes.
        """
        self.key = key

    @property
    def valid(self):
        """ Check if given API key is valid. """
        response = requests.get(BASE_URL, {'api-key': self.key})
        return response.status_code == 200

    def __eq__(self, other):
        return self.key == other.key


class TheGuardianAPI:
    class_vars = [
        (partial(DiscreteVariable, "Section"), lambda doc: doc["sectionName"]),
    ]

    metas = [
        (partial(StringVariable, "Headline"), lambda doc: doc["fields"]["headline"]),
        (partial(StringVariable, "Content"), lambda doc: doc["fields"]["bodyText"]),
        (partial(StringVariable, "Trail Text"), lambda doc: doc["fields"]["trailText"]),
        (partial(StringVariable, "HTML"), lambda doc: doc["fields"]["body"]),
        (
            partial(TimeVariable, "Publication Date", have_time=1, have_date=1),
            lambda doc: isoparse(doc["webPublicationDate"]).timestamp(),
        ),
        (partial(DiscreteVariable, "Type"), lambda doc: doc["type"]),
        (partial(DiscreteVariable, "Language"), lambda doc: doc["fields"]["lang"]),
        (
            partial(StringVariable, "Tags"),
            lambda doc: ", ".join(tag["webTitle"] for tag in doc["tags"]),
        ),
        (partial(StringVariable, "URL"), lambda doc: doc["webUrl"]),
        (
            partial(ContinuousVariable, "Word Count", number_of_decimals=0),
            lambda doc: doc["fields"]["wordcount"],
        ),
    ]

    text_features = ["Headline", "Content"]  #
    title_indices = [-1]    # Headline

    def __init__(self, credentials, on_progress=None, should_break=None):
        """
        Args:
            credentials (:class:`TheGuardianCredentials`): The Guardian Creentials.
            on_progress (callable): Function for progress reporting.
            should_break (callable): Function for early stopping.
        """
        self.per_page = ARTICLES_PER_PAGE
        self.pages = 0
        self.credentials = credentials
        self.on_progress = on_progress or (lambda x, y: None)
        self.should_break = should_break or (lambda: False)

        self.results = []

    def _search(self, query, from_date, to_date, page=1):
        data = self._build_query(query, from_date, to_date, page)

        response = requests.get(BASE_URL, data)
        response.encoding = "UTF-8"
        if response.status_code == 429:
            raise APILimitError("API limit exceeded")

        parsed = json.loads(response.text)

        if page == 1:   # store number of pages
            self.pages = parsed['response']['pages']

        self.results.extend(parsed['response']['results'])

    def _build_query(self, query, from_date=None, to_date=None, page=1):
        data = {
            'q': query,
            'api-key': self.credentials.key,
            'page': str(page),
            'show-fields': 'headline,trailText,body,bodyText,lang,wordcount',
            'show-tags': 'all',
        }
        if from_date is not None:
            data['from-date'] = from_date
        if to_date is not None:
            data['to-date'] = to_date

        return data

    def search(self, query, from_date=None, to_date=None, max_documents=None,
               accumulate=False):
        """
        Search The Guardian API for articles.

        Args:
            query (str): A query for searching the articles by
            from_date (str): Search only articles newer than the date provided.
                Date should be in ISO format; e.g. '2016-12-31'.
            to_date (str): Search only articles older than the date provided.
                Date should be in ISO format; e.g. '2016-12-31'.
            max_documents (int): Maximum number of documents to retrieve.
                When not given, retrieve all documents.
            accumulate (bool): A flag indicating whether to accumulate results
                of multiple consequent search calls.

        Returns:
            :ref:`Corpus`
        """
        if not accumulate:
            self.results = []

        self._search(query, from_date, to_date)

        pages = math.ceil(max_documents/self.per_page) if max_documents else self.pages
        self.on_progress(self.per_page, pages * self.per_page)

        for p in range(2, pages+1):     # to one based
            if self.should_break():
                break
            self._search(query, from_date, to_date, p)
            self.on_progress(p*self.per_page, pages * self.per_page)

        return create_corpus(
            self.results,
            [],
            self.class_vars,
            self.metas,
            self.title_indices,
            self.text_features,
            "The Guardian",
            "Language",
        )


if __name__ == '__main__':
    key = os.getenv('THE_GUARDIAN_API_KEY', 'test')
    credentials = TheGuardianCredentials(key)
    print(credentials.valid)
    api = TheGuardianAPI(credentials=credentials)
    c = api.search('refugees', max_documents=10)
    print(c)
