import unittest
import os

from datetime import date, datetime

from orangecontrib.text import guardian


API_KEY = os.getenv('THE_GUARDIAN_API_KEY', 'test')


class TestCredentials(unittest.TestCase):
    def test_valid(self):
        credentials = guardian.TheGuardianCredentials(API_KEY)
        self.assertTrue(credentials.valid)

    def test_equal(self):
        credentials = guardian.TheGuardianCredentials(API_KEY)
        self.assertEquals(credentials, credentials)


class TestGuardian(unittest.TestCase):
    def setUp(self):
        c = guardian.TheGuardianCredentials(API_KEY)
        self.api = guardian.TheGuardianAPI(c)

    def test_search(self):
        corp = self.api.search('Slovenia', max_documents=20)
        self.assertEqual(len(corp), 20)

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

    def test_breaking(self):
        self.api.should_break = lambda: True
        corp = self.api.search('Slovenia', max_documents=20)
        self.assertEqual(len(corp), 10)
        self.api.should_break = lambda: False

    def test_accumulate(self):
        self.api.search('Slovenia', max_documents=10, accumulate=True)
        corp = self.api.search('Ljubljana', max_documents=10, accumulate=True)
        self.assertEqual(len(corp), 20)
