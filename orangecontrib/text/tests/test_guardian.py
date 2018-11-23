import unittest
import os

from datetime import date, datetime
from unittest import mock

from orangecontrib.text import guardian


API_KEY = os.getenv('THE_GUARDIAN_API_KEY', 'test')


class TestCredentials(unittest.TestCase):
    @mock.patch('requests.get')
    def test_valid(self, mock_get):
        mock_get().status_code = 200
        credentials = guardian.TheGuardianCredentials(API_KEY)
        self.assertTrue(credentials.valid)

    def test_equal(self):
        credentials = guardian.TheGuardianCredentials(API_KEY)
        self.assertEquals(credentials, credentials)


def skip_limit_exceeded(fun):
    def wrapper(*args, **kwargs):
        try:
            fun(*args, **kwargs)
        except guardian.APILimitError as err:
            raise unittest.SkipTest(str(err))
    return wrapper


class TestGuardian(unittest.TestCase):
    def setUp(self):
        c = guardian.TheGuardianCredentials(API_KEY)
        self.api = guardian.TheGuardianAPI(c)

    @skip_limit_exceeded
    def test_search(self):
        corp = self.api.search('Slovenia', max_documents=20)
        self.assertEqual(len(corp), 20)

    @skip_limit_exceeded
    def test_search_from_to_date(self):
        from_date = date(2016, 12, 1)
        to_date = date(2016, 12, 31)
        corp = self.api.search('Slovenia', max_documents=10,
                               from_date=from_date.isoformat(),
                               to_date=to_date.isoformat())
        time_ind = 4
        data = corp.metas[:, time_ind]
        for val in data:
            dt = datetime.utcfromtimestamp(val)
            self.assertGreaterEqual(dt.date(), from_date)
            self.assertLessEqual(dt.date(), to_date)

    @skip_limit_exceeded
    def test_breaking(self):
        self.api.should_break = lambda: True
        corp = self.api.search('Slovenia', max_documents=20)
        self.assertEqual(len(corp), 10)
        self.api.should_break = lambda: False

    @skip_limit_exceeded
    def test_accumulate(self):
        self.api.search('Slovenia', max_documents=10, accumulate=True)
        corp = self.api.search('Ljubljana', max_documents=10, accumulate=True)
        self.assertEqual(len(corp), 20)

    @mock.patch('requests.get')
    def test_api_limit_error(self, mock_get):
        mock_get().status_code = 429
        self.assertRaises(guardian.APILimitError, self.api.search, 'Slovenia')

    @mock.patch('requests.get')
    def test_search_mock_data(self, mock_get):
        mock_get().text = """
        {
          "response": {
            "pages": 2,
            "results": [
              {
                "type": "article",
                "sectionName": "World news",
                "webPublicationDate": "2018-07-05T23:27:25Z",
                "webUrl": "https://www.theguardian.com/world/2018/jul/06",
                "fields": {
                  "headline": "Rohingya refugees reject UN-Myanmar repatriati",
                  "trailText": "Leaders say agreement does not address concer",
                  "body": "<p><strong><strong><strong></strong></strong></str",
                  "wordcount": "512",
                  "lang": "en",
                  "bodyText": "Rohingya community leaders have rejected an."
                },
                "tags": [
                  {
                    "webTitle": "Myanmar"
                  }
                ]
              }
            ]
          }
        }
        """
        corp = self.api.search('Slovenia')
        self.assertEqual(len(corp), 2)
