import json
from datetime import date
from urllib import request


def validate_year(year):
    if int(year) <= 1851:
        return "1851"
    if int(year) >= date.today().year:
        return str(date.today().year)
    return year


class NYT:
    def __init__(self, api_key):
        self._api_key = api_key

    @staticmethod
    def execute_query(query_url):
        """
        Execute a query and get the data from the New York Times Article API.

        :param query: The URL that we wish to query.
        :return: A JSON representation of the query's results.
        """
        connection = request.urlopen(query_url)
        response = connection.readall().decode("utf-8")
        return json.loads(response)

    def parse_record_json(self, record):
        """
        Parses the JSON representation of the record returned by the New York Times Article API.

        :param record: The JSON representation of the query's results.
        :return: A list of articles parsed into documents and a list of the
            corresponding metadata, joined in a tuple.
        """
        documents = []
        for doc in record["response"]["docs"]:
            documents.append(self._walk_json(doc))
        return documents

    def _walk_json(self, json_document):
        """
        A recursive method called to collect and concatenate all values from a JSON
        of unknown hierarchical structure. Not meant to be called by the user.

        :param json_document: The JSON representation of the document.
        :return: All values of the input JSON concatenated into a single string.
        """
        string_document = ""
        for item in json_document:
            item = json_document[item]
            if isinstance(item, list) or isinstance(item, dict):
                string_document += self._walk_json(item)
            else:
                string_document += " " + item
        return string_document
