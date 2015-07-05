import os
import json
import shelve
import warnings
import datetime
import numpy as np
from urllib import request, parse
from urllib.error import HTTPError, URLError

from Orange.canvas.utils import environ


def parse_record_json(record, includes_metadata):
    """
    Parses the JSON representation of the record returned by the New York Times Article API.
    :param record: The JSON representation of the query's results.
    :param includes_metadata: The flags that determine which fields to include.
    :return: A list of the corresponding metadata and class values for the instances.
    """
    class_values = []
    metadata = np.empty((0, len(includes_metadata)+2), dtype=object)    # columns: Text fields + (pub_date + country)
    for doc in record["response"]["docs"]:
        metas_row = []
        for field in includes_metadata:
            if field in doc:
                field_value = ""
                if isinstance(doc[field], dict):
                    field_value = " ".join([val for val in doc[field].values() if val])
                elif isinstance(doc[field], list):
                    field_value = " ".join([kw["value"] for kw in doc[field] if kw])
                else:
                    if doc[field]:
                        field_value = doc[field]
                metas_row.append(field_value)
        # Add the pub_date.
        metas_row.append(doc.get("pub_date", ""))
        # Add the glocation.
        metas_row.append(",".join([kw["value"] for kw in doc["keywords"] if kw["name"] == "glocations"]))

        # Add the section_name.
        class_values.append(doc.get("section_name", None))

        metas_row = ["" if v is None else v for v in metas_row]
        metadata = np.vstack((metadata, np.array(metas_row)))

    return metadata, class_values


def _date_to_str(input_date):
    iso = input_date.isoformat()
    date_part = iso.strip().split("T")[0].split("-")
    return "%s%s%s" % (date_part[0], date_part[1], date_part[2])


class NYT:
    base_url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json'

    def __init__(self, api_key):
        self._api_key = api_key.strip()
        self._query_url = None
        self.query_key = None
        self.includes_fields = None

        self.cache_path = None
        cache_folder = os.path.join(environ.buffer_dir, "nytcache")
        try:
            if not os.path.exists(cache_folder):
                os.makedirs(cache_folder)
            self.cache_path = os.path.join(cache_folder, "query_cache")
        except:
            warnings.warn('Could not assemble NYT query cache path', RuntimeWarning)

    def set_query_url(self, query, year_from=None, year_to=None, text_includes=None):
        """
        Builds a NYT article API query url with the input parameters.
        The page parameter is added later when the query is executed.

        :param query: The query keywords in a string, but separated with whitespaces.
        :param year_from: A digit input that signifies to return articles
            from this year forth only.
        :param year_to: A digit input that signifies to return articles
            up to this year only.
        :param text_includes: A list of text fields that are to be requested.
        :return: Returns the query URL in the form of a string.
        """
        # Query keywords, base url and API key.
        query_url = self.encode_base_url(query)
        query_key = []  # This query's key, to store with shelve.

        # Use these as parts of cache keys to prevent re-use of old cache.
        query_key_sdate = _date_to_str(datetime.date(1851, 1, 1))
        query_key_edate = _date_to_str(datetime.datetime.now().date())
        # Check user date range input.
        if year_from and year_from.isdigit():
            query_key_sdate = year_from + "0101"
            query_url += "&begin_date=" + query_key_sdate
        if year_to and year_to.isdigit():
            query_key_edate = year_to + "1231"
            query_url += "&end_date=" + query_key_edate
        # Update cache keys.
        query_key.append(query_key_sdate)
        query_key.append(query_key_edate)

        # Text fields.
        if text_includes:
            self.includes_fields = text_includes
            fl = ",".join(text_includes)
            query_url += "&fl=" + fl
        # Add pub_date.
        query_url += ",pub_date"
        # Add section_name.
        query_url += ",section_name"
        # Add keywords in every case, since we need them for geolocating.
        if "keywords" not in text_includes:
            query_url += ",keywords"

        self._query_url = "{0}?{1}".format(self.base_url, query_url)

        # Queries differ in query words, included text fields and date range.
        query_key.extend(query.split(" "))
        query_key.extend(text_includes)

        self.query_key = "_".join(query_key)
        return self._query_url

    def encode_base_url(self, query):
        """
        Builds the foundation url string for this query.

        :param query: The keywords for this query.
        :return: The foundation url for this query.
        """
        return parse.urlencode({"q": query, "fq": "The New York Times", "api-key": self._api_key})

    def check_api_key(self):
        """
        Checks whether the api key provided to this class instance, is valid.

        :return: True or False depending on the validation outcome.
        """
        query_url = self.encode_base_url("test")
        try:
            connection = request.urlopen("{0}?{1}&page=0".format(self.base_url, query_url))
            if connection.getcode() == 200:  # The API key works.
                return True
        except:
            return False

    def execute_query(self, page):
        """
        Execute a query and get the data from the New York Times Article API.
        Will not execute, if the class object's query_url has not been set.

        :param page: Determine what page of the query to return.
        :return: A JSON representation of the query's results, a boolean flag,
            that serves as feedback, whether the request was cached or not and
            the error if one occurred during execution.
        """
        if self._query_url:
            # Method return values.
            response_data = ""
            is_cached = False

            current_query = self._query_url+"&page={}".format(page)
            current_query_key = self.query_key + "_" + str(page)

            with shelve.open(self.cache_path) as query_cache:
                if current_query_key in query_cache.keys():
                    response = query_cache[current_query_key]
                    response_data = json.loads(response)
                    is_cached = True
                    error = None
                else:
                    try:
                        connection = request.urlopen(current_query)
                        response = connection.readall().decode("utf-8")
                        query_cache[current_query_key] = response

                        response_data = json.loads(response)
                        is_cached = False
                        error = None
                    except (HTTPError, URLError) as err:
                        error = err

                query_cache.close()    # Release resources.
            return response_data, is_cached, error
