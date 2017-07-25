import json
import os
import unittest
from unittest.mock import patch

import numpy as np

from orangecontrib.text.pubmed import (
    Pubmed, PUBMED_TEXT_FIELDS, PUBMED_FIELD_DATE,
    _mesh_headings_to_class,
    _date_to_iso, _corpus_from_records,
    _records_to_corpus_entries
)


class MockEntrezHandle:
    @staticmethod
    def close():
        return


class MockEntrez:
    """Used to mock Entrez/Medline reading/parsing methods.

    Mocks read (after esearch and epost) and parse (after efetch).
    """

    def __init__(self, cache):
        self.mock_responses = {}
        with open(cache, 'r') as f:
            self.mock_responses = json.loads(f.read())

    def esearch(self, db, term, **keywds):
        return MockEntrezHandle()

    def read(self, handle):
        return self.mock_responses.get('read')

    def efetch(self, db, **keywords):
        return MockEntrezHandle()

    def epost(self, db, **keywds):
        return MockEntrezHandle()

    def parse(self, handle):
        return self.mock_responses.get('parse')

    # Exception mocking.
    def esearch_exception(self, db, term, **keywds):
        raise IOError

    def efetch_exception(self, db, **keywds):
        raise Exception

    def epost_exception(self, db, **keywds):
        raise IOError


CACHE = os.path.join(os.path.dirname(__file__), 'pubmed-cache.txt')
mock_entrez = MockEntrez(CACHE)


def error_callback(exception):
    return


def progress_callback(progress=None):
    return


class PubmedTests(unittest.TestCase):
    EMAIL = 'mockemail@mockdomain.com'

    def setUp(self):
        self.pubmed = Pubmed(
                self.EMAIL,
                progress_callback=progress_callback,
                error_callback=error_callback
        )

    def test_pubmed_object_creation(self):
        self.assertRaises(
            ValueError,
            Pubmed,
            'faulty_email'
        )

    def test_mesh_headings_to_class(self):
        input_headings = [
            'heading1 & heading2/heading3,heading4/*heading5',
            'heading1/heading2/*heading3',
        ]
        self.assertEqual(_mesh_headings_to_class(input_headings), 'heading1')

    def test_date_to_iso(self):
        # Correct inputs.
        input_dates = [
            '2015 Nov',
            '2015',
            '2015 Sep-Oct',
            '2015 Fall',
        ]

        correct_results = [
            '2015-11-01',
            '2015-01-01',
            '2015-09-01',
            '2015-09-01',
        ]

        for date, result in zip(input_dates, correct_results):
            self.assertEqual(_date_to_iso(date), result)

        # Unexpected inputs.
        unexpected_input = '2015 Unexpected'
        self.assertWarns(
            RuntimeWarning,
            _date_to_iso,
            unexpected_input,
        )
        self.assertEqual(type(_date_to_iso(unexpected_input)), type(np.nan))


    def test_record_to_corpus(self):
        mock_records = [
            {
                'FAU': ['Mock Author 1', 'Mock Author 2'],
                'TI': 'Mock title',
                'MH': ['heading1/heading2'],
                'AB': 'Mock abstract',
                'DP': '2015 Sep',
                'PMID': 1,
            },
        ]

        correct_metas = np.array([
            [
                'Mock Author 1 Mock Author 2',
                'Mock title',
                'heading1/heading2',
                'Mock abstract',
                'http://www.ncbi.nlm.nih.gov/pubmed/?term=1',
                1441065600.0
            ]
        ], dtype=object)
        correct_classes = np.array([
            'heading1'
        ])

        # Perform asserting.
        corpus = _corpus_from_records(mock_records, PUBMED_TEXT_FIELDS)
        meta_values, class_values = _records_to_corpus_entries(
                mock_records,
                PUBMED_TEXT_FIELDS,
                corpus.domain[PUBMED_FIELD_DATE]
        )
        self.assertCountEqual(meta_values[0], correct_metas[0])
        self.assertCountEqual(class_values, correct_classes)
        self.assertIsNotNone(corpus)

    @patch('Bio.Entrez.esearch', mock_entrez.esearch)
    @patch('Bio.Entrez.read', mock_entrez.read)
    def test_pubmed_search_records(self):
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'

        self.pubmed._search_for_records(
                terms=test_terms,
                authors=authors,
                pub_date_start=pub_date_start,
                pub_date_end=pub_date_end
        )

        # The only certain check is to make sure we got all the parameters.
        self.assertIsNotNone(self.pubmed.record_id_list)
        self.assertIsNotNone(self.pubmed.search_record_count)
        self.assertIsNotNone(self.pubmed.search_record_web_env)
        self.assertIsNotNone(self.pubmed.search_record_query_key)

        # Faulty input check.
        self.assertRaises(
                ValueError,
                self.pubmed._search_for_records,
                terms=test_terms,
                authors=None,
                pub_date_start=pub_date_start,
                pub_date_end=pub_date_end
        )

    @patch('Bio.Entrez.esearch', mock_entrez.esearch)
    @patch('Bio.Entrez.read', mock_entrez.read)
    @patch('Bio.Entrez.efetch', mock_entrez.efetch)
    @patch('Bio.Medline.parse', mock_entrez.parse)
    def test_pubmed_retrieve_record_batch(self):
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'
        offset = 0
        num_requested_records = 5

        # Attempt to retrieve without searching first.
        self.assertRaises(
                ValueError,
                self.pubmed._retrieve_record_batch,
                offset,
                num_requested_records
        )

        # Must search for records first.
        self.pubmed._search_for_records(
                test_terms,
                authors,
                pub_date_start,
                pub_date_end
        )

        # Retrieve the records.
        data = self.pubmed._retrieve_record_batch(
                offset,
                num_requested_records
        )

        self.assertEqual(len(data), num_requested_records)

    @patch('Bio.Entrez.esearch', mock_entrez.esearch)
    @patch('Bio.Entrez.read', mock_entrez.read)
    @patch('Bio.Entrez.efetch', mock_entrez.efetch)
    @patch('Bio.Medline.parse', mock_entrez.parse)
    @patch('Bio.Entrez.epost', mock_entrez.epost)
    def test_pubmed_retrieve_records(self):
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'

        num_records = 5

        # Must search for records first.
        self.pubmed._search_for_records(
                test_terms,
                authors,
                pub_date_start,
                pub_date_end
        )

        # Retrieve the records and build a corpus.
        corpus = self.pubmed._retrieve_records(num_records)
        self.assertEqual(len(corpus), num_records)

        meta_fields = sorted([field_name
                              for field_name, field_tag
                              in PUBMED_TEXT_FIELDS])
        test_meta_fields = sorted([m.name
                                   for m
                                   in corpus.domain.metas])
        self.assertEqual(meta_fields, test_meta_fields)

    @patch('Bio.Entrez.esearch', mock_entrez.esearch)
    @patch('Bio.Entrez.read', mock_entrez.read)
    @patch('Bio.Entrez.efetch', mock_entrez.efetch)
    @patch('Bio.Medline.parse', mock_entrez.parse)
    @patch('Bio.Entrez.epost', mock_entrez.epost)
    def test_pubmed_retrieve_records_no_cache(self):
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'

        num_records = 5

        # Must search for records first.
        self.pubmed._search_for_records(
                test_terms,
                authors,
                pub_date_start,
                pub_date_end
        )

        # Retrieve the records and build a corpus.
        corpus = self.pubmed._retrieve_records(
                num_records,
                use_cache=False
        )
        self.assertEqual(len(corpus), num_records)

        meta_fields = sorted([field_name
                              for field_name, field_tag
                              in PUBMED_TEXT_FIELDS])
        test_meta_fields = sorted([m.name
                                   for m
                                   in corpus.domain.metas])
        self.assertEqual(meta_fields, test_meta_fields)

    @patch('Bio.Entrez.esearch', mock_entrez.esearch)
    @patch('Bio.Entrez.read', mock_entrez.read)
    @patch('Bio.Entrez.efetch', mock_entrez.efetch)
    @patch('Bio.Medline.parse', mock_entrez.parse)
    @patch('Bio.Entrez.epost', mock_entrez.epost)
    def test_download_records(self):
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'

        num_records = 5

        # Retrieve the records and build a corpus.
        corpus = self.pubmed.download_records(
            test_terms,
            authors,
            pub_date_start,
            pub_date_end,
            num_records
        )
        self.assertEqual(len(corpus), num_records)

        meta_fields = sorted([field_name
                              for field_name, field_tag
                              in PUBMED_TEXT_FIELDS])
        test_meta_fields = sorted([m.name
                                   for m
                                   in corpus.domain.metas])
        self.assertEqual(meta_fields, test_meta_fields)

    @patch('Bio.Entrez.esearch', mock_entrez.esearch_exception)
    def test_entrez_search_exceptions(self):
        # Search exception.
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'

        self.assertWarns(
            RuntimeWarning,
            self.pubmed._search_for_records,
            terms=test_terms,
            authors=authors,
            pub_date_start=pub_date_start,
            pub_date_end=pub_date_end
        )

    @patch('Bio.Entrez.esearch', mock_entrez.esearch)
    @patch('Bio.Entrez.read', mock_entrez.read)
    @patch('Bio.Entrez.efetch', mock_entrez.efetch_exception)
    @patch('Bio.Medline.parse', mock_entrez.parse)
    @patch('Bio.Entrez.epost', mock_entrez.epost)
    def test_pubmed_retrieve_record_batch_exception(self):
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'

        num_records = 5

        # Must search for records first.
        self.pubmed._search_for_records(
                test_terms,
                authors,
                pub_date_start,
                pub_date_end
        )

        self.assertWarns(
            RuntimeWarning,
            self.pubmed._retrieve_records,
            num_records,
            use_cache=False
        )

    @patch('Bio.Entrez.esearch', mock_entrez.esearch)
    @patch('Bio.Entrez.read', mock_entrez.read)
    @patch('Bio.Entrez.efetch', mock_entrez.efetch)
    @patch('Bio.Medline.parse', mock_entrez.parse)
    @patch('Bio.Entrez.epost', mock_entrez.epost_exception)
    def test_pubmed_epost_exception(self):
        test_terms = ['orchid']
        authors = []
        pub_date_start = '2011/07/07'
        pub_date_end = '2014/07/07'

        num_records = 5

        # Must search for records first.
        self.pubmed._search_for_records(
                test_terms,
                authors,
                pub_date_start,
                pub_date_end
        )

        self.assertWarns(
            RuntimeWarning,
            self.pubmed._retrieve_records,
            num_records,
            use_cache=False
        )
