import os
import json
import shelve
import numpy as np
from datetime import date
from urllib import request, parse
from urllib.error import HTTPError, URLError

from Orange.canvas.utils import environ
from Orange.data import StringVariable


def validate_year(year):
    if int(year) <= 1851:
        return "1851"
    if int(year) >= date.today().year:
        return str(date.today().year)
    return year


def parse_record_json(record, includes_metadata):
    """
    Parses the JSON representation of the record returned by the New York Times Article API.
    :param record: The JSON representation of the query's results.
    :param includes_metadata: The flags that determine which fields to include.
    :return: A list of articles parsed into documents and a list of the
        corresponding metadata, joined in a tuple.
    """
    text_fields = ["headline", "lead_paragraph", "snippet", "abstract", "keywords"]

    documents = []
    meta_vars = [StringVariable.make(field) for field, flag in zip(text_fields, includes_metadata) if flag]
    # Also add pub_date and glocation.
    meta_vars += [StringVariable.make("pub_date"), StringVariable.make("country")]
    metadata = np.empty((0, len(meta_vars)), dtype=object)
    for doc in record["response"]["docs"]:
        string_document = ""
        metas_row = []
        for field, flag in zip(text_fields, includes_metadata):
            if flag and field in doc:
                field_value = ""
                if isinstance(doc[field], dict):
                    field_value = " ".join([val for val in doc[field].values() if val])
                elif isinstance(doc[field], list):
                    field_value = " ".join([kw["value"] for kw in doc[field] if kw])
                else:
                    if doc[field]:
                        field_value = doc[field]
                string_document += field_value
                metas_row.append(field_value)
        # Add the pub_date.
        field_value = ""
        if "pub_date" in doc and doc["pub_date"]:
            field_value = doc["pub_date"]
        metas_row.append(field_value)
        # Add the glocation.
        metas_row.append(" ".join([kw["value"] for kw in doc["keywords"] if kw["name"] == "glocations"]))

        documents.append(string_document)
        metadata = np.vstack((metadata, np.array(metas_row)))
    return documents, metadata, meta_vars


class NYT:
    base_url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'

    def __init__(self, api_key):
        self._api_key = api_key
        self._query_url = ""
        self.query_key = ""

        self.cache_pth = os.path.join(environ.buffer_dir, "nytcache")
        try:
            os.makedirs(self.cache_pth)
        except:
            pass
        self.query_cache = None
        self.includes_fields = [False]*5

    def set_query_url(self,
                      query,
                      year_from=None, year_to=None,
                      text_includes=None):
        """
        Builds a NYT article API query url with the input parameters.
        The page parameter is added later when the query is executed.

        :param query: The query keywords in a string, but separated with whitespaces.
        :param year_from: A digit input that signifies to return articles
            from this year forth only.
        :param year_to: A digit input that signifies to return articles
            up to this year only.
        :param text_includes: A list of boolean flags that determines what text fields,
            the API should return. Ordered as follows: [headline, lead_paragraph, snippet,
            abstract, keywords]
        :return: Returns the query URL in the form of a string.
        """
        if text_includes:
            self.includes_fields = text_includes

        # Query keywords, base url and API key.
        query_url = parse.urlencode({
            "q": query,
            "fq": "The New York Times",
            "api-key": self._api_key
        })

        # Years.
        if year_from and year_from.isdigit():
            query_url += "&begin_date=" + validate_year(year_from) + "0101"
        if year_to and year_to.isdigit():
            query_url += "&end_date=" + validate_year(year_to) + "1231"

        # Text fields.
        if text_includes and True in text_includes:
            fl_fields = ["headline", "lead_paragraph", "snippet", "abstract", "keywords"]
            fl = ",".join([f1 for f1, f2 in zip(fl_fields, text_includes) if f2])
            query_url += "&fl=" + fl
        # Add pub_date.
        query_url += ",pub_date"
        # Add keywords in every case, since we need them for geolocating.
        if not text_includes[-1]:
            query_url += ",keywords"

        self._query_url = "{0}?{1}".format(self.base_url, query_url)
        # This query's key, to store with shelve.
        # Queries differ in query words, included text fields and date range.
        query_key = query.split(" ") + [f1 for f1, f2 in zip(fl_fields, text_includes) if f2]
        if year_from:
            query_key += [validate_year(year_from)]
        if year_to:
            query_key += [validate_year(year_to)]
        self.query_key = "_".join(query_key)
        return self._query_url

    def check_api_key(self):
        query_url = parse.urlencode({
            "q": "test",
            "fq": "The New York Times",
            "api-key": self._api_key
        })
        try:
            connection = request.urlopen("{0}?{1}".format(self.base_url, query_url+"&page=0"))
            if connection.getcode() == 200:  # The API key works.
                return True
        except:
            return False

    def execute_query(self, page):
        """
        Execute a query and get the data from the New York Times Article API.
        Will not execute, if the class object's query_url has not been set.

        :param page: Determine what page of the query to return.
        :return: A JSON representation of the query's results and a boolean flag,
            that serves as feedback, whether the request was cached or not.
        """
        if self._query_url:
            current_query = self._query_url+"&page={}".format(page)
            curreny_query_key = self.query_key + "_" + str(page)
            self.query_cache = shelve.open(os.path.join(self.cache_pth, "query_cache"))
            if curreny_query_key in self.query_cache.keys():
                response = self.query_cache[curreny_query_key]
                self.query_cache.close()
                return json.loads(response), True, None
            else:
                try:
                    connection = request.urlopen(current_query)
                    response = connection.readall().decode("utf-8")
                    self.query_cache[curreny_query_key] = response
                    self.query_cache.close()
                    return json.loads(response), False, None
                except HTTPError as error:
                    return "", False, error
                except URLError as error:
                    return "", False, error
