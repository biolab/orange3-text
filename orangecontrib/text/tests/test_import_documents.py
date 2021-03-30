import unittest
from unittest.mock import patch

import pandas as pd

from orangecontrib.text.import_documents import ImportDocuments, UrlReader, \
    TxtReader, TextData


class TestUrlReader(unittest.TestCase):
    def test_init(self):
        path = "http://dummy.server.com/data/foo.txt"
        reader = UrlReader(path)
        self.assertEqual(reader.filename, path)
        self.assertEqual(reader.path, path)

    def test_get_reader(self):
        path = "http://dummy.server.com/data/foo.txt"
        reader = UrlReader.get_reader(path)
        self.assertIsInstance(reader, TxtReader)

    def test_read(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/C-1.txt"
        reader = UrlReader(path)
        textdata, error = reader.read()
        self.assertIsInstance(textdata, TextData)
        self.assertEqual(textdata.name, "C-1")
        self.assertEqual(textdata.path, path)
        self.assertEqual(textdata.ext, [".txt"])
        self.assertEqual(textdata.category, "semeval")
        self.assertTrue(textdata.content.startswith("On The Complexity of Co"))
        self.assertEqual(error, "")

    def test_read_file(self):
        path = "http://file.biolab.si/text-semantics/data/elektrotehniski-" \
               "vestnik-clanki/detektiranje-utrdb-v-šahu-.txt"
        reader = UrlReader(path)
        reader.read_file()
        self.assertIsInstance(reader.content, str)

    def test_name_text_data(self):
        path = "http://dummy.server.com/data/foo.txt"
        reader = UrlReader(path)
        reader.content = "text"
        text_data = reader.make_text_data()
        self.assertIsInstance(text_data, TextData)
        self.assertEqual(text_data.name, "foo")
        self.assertEqual(text_data.path, path)
        self.assertEqual(text_data.ext, [".txt"])
        self.assertEqual(text_data.category, "data")
        self.assertEqual(text_data.content, "text")


class TestImportDocuments(unittest.TestCase):
    def test_scan_url(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path)
        self.assertEqual(len(paths), 101)

    def test_scan_url_txt(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path, include_patterns=["*.txt"])
        self.assertEqual(len(paths), 100)

    def test_scan_url_csv(self):
        path = "http://file.biolab.si/text-semantics/data/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path, include_patterns=["*.csv"])
        self.assertEqual(len(paths), 6)

    def test_read_meta_data_url(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        data1, err = importer._read_meta_data()
        self.assertIsInstance(data1, pd.DataFrame)
        self.assertEqual(len(err), 0)

    @patch("orangecontrib.text.import_documents.ImportDocuments."
           "META_DATA_FILE_KEY", "File")
    def test_merge_metadata_url(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        text_data, _ = importer._read_text_data()
        meta_data, _ = importer._read_meta_data()

        importer._text_data = text_data[:4]  # 'C-1', 'C-14', 'C-17', 'C-18'
        importer._meta_data = meta_data[:50]
        corpus = importer._create_corpus()
        corpus = importer._add_metadata(corpus)
        self.assertEqual(len(corpus), 4)
        columns = ["name", "path", "content", "Content", "File", "Keywords"]
        self.assertEqual([v.name for v in corpus.domain.metas], columns)

        importer._text_data = text_data[:4]  # 'C-1', 'C-14', 'C-17', 'C-18'
        importer._meta_data = None
        corpus = importer._create_corpus()
        corpus = importer._add_metadata(corpus)
        self.assertEqual(len(corpus), 4)
        columns = ["name", "path", "content"]
        self.assertEqual([v.name for v in corpus.domain.metas], columns)

    def test_run_url(self):
        path = "http://file.biolab.si/text-semantics/data/" \
               "elektrotehniski-vestnik-clanki/"
        importer = ImportDocuments(path, True)
        corpus, errors = importer.run()
        self.assertEqual(len(corpus), 382)


if __name__ == "__main__":
    unittest.main()
