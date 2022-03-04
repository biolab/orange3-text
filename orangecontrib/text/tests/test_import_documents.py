import asyncio
import os
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
from httpx import URL, ConnectTimeout

from orangecontrib.text.import_documents import (
    ImportDocuments,
    UrlProxyReader,
    TxtReader,
    TextData,
)


PATCH_METHOD = "httpx.AsyncClient.get"


async def dummy_post(_, url):
    await asyncio.sleep(0)
    return DummyResponse(url)


class DummyResponse:
    content = b"lorem ipsum"

    def __init__(self, url):
        super().__init__()
        self.url = URL(url)

    raise_for_status = MagicMock()


class TestUrlProxyReader(unittest.TestCase):
    @patch(PATCH_METHOD, dummy_post)
    def test_init(self):
        path = "http://dummy.server.com/data/foo.txt"
        result = UrlProxyReader.read_files([path])
        reader, text_data, error = result[0]
        self.assertEqual(text_data.path, path)
        self.assertEqual("", error)
        self.assertIsInstance(reader, TxtReader)

    def test_read(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/C-1.txt"
        result = UrlProxyReader.read_files([path])
        reader, textdata, error = result[0]
        self.assertIsInstance(textdata, TextData)
        self.assertEqual(textdata.name, "C-1")
        self.assertEqual(textdata.path, path)
        self.assertEqual(textdata.ext, [".txt"])
        self.assertEqual(textdata.category, "semeval")
        self.assertTrue(textdata.content.startswith("On The Complexity of Co"))
        self.assertEqual(error, "")

    @patch(PATCH_METHOD, dummy_post)
    def test_name_text_data(self):
        path = "http://dummy.server.com/data/foo.txt"
        result = UrlProxyReader.read_files([path])
        reader, text_data, error = result[0]
        self.assertIsInstance(text_data, TextData)
        self.assertEqual(text_data.name, "foo")
        self.assertEqual(text_data.path, path)
        self.assertEqual(text_data.ext, [".txt"])
        self.assertEqual(text_data.category, "data")
        self.assertEqual(text_data.content, "lorem ipsum")

    @patch(PATCH_METHOD, side_effect=ConnectTimeout("test message", request=""))
    def test_error(self, _):
        path = "http://dummy.server.com/data/foo.txt"
        result = UrlProxyReader.read_files([path])
        reader, text_data, error = result[0]
        self.assertIsNone(reader)
        self.assertIsNone(text_data)
        self.assertEqual(path, error)


class TestImportDocuments(unittest.TestCase):
    def test_scan_url(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path)
        self.assertGreater(len(paths), 0)

    def test_scan_url_txt(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path, include_patterns=["*.txt"])
        self.assertGreater(len(paths), 0)

    def test_scan_url_csv(self):
        path = "http://file.biolab.si/text-semantics/data/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path, include_patterns=["*.csv"])
        self.assertGreater(len(paths), 0)

    def test_read_meta_data_url(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        _, meta_paths = importer._retrieve_paths()
        callback = importer._shared_callback(len(meta_paths))
        data1, err = importer._read_meta_data(meta_paths, callback)
        self.assertIsInstance(data1, pd.DataFrame)
        self.assertEqual(len(err), 0)

    # @patch("orangecontrib.text.import_documents.ImportDocuments."
    #        "META_DATA_FILE_KEY", "File")
    def test_merge_metadata_url(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        file_paths, meta_paths = importer._retrieve_paths()
        callback = importer._shared_callback(len(file_paths) + len(meta_paths))
        text_data, _, _, _, _, _ = importer._read_text_data(file_paths, callback)
        meta_data, _ = importer._read_meta_data(meta_paths, callback)

        importer._text_data = text_data[:4]  # 'C-1', 'C-14', 'C-17', 'C-18'
        importer._meta_data = meta_data[:50]
        corpus = importer._create_corpus()
        corpus = importer._add_metadata(corpus)
        self.assertGreater(len(corpus), 0)
        columns = ["name", "path", "content", "Content",
                   "Text file", "Keywords"]
        self.assertEqual([v.name for v in corpus.domain.metas], columns)

        importer._text_data = text_data[:4]  # 'C-1', 'C-14', 'C-17', 'C-18'
        importer._meta_data = None
        corpus = importer._create_corpus()
        corpus = importer._add_metadata(corpus)
        self.assertGreater(len(corpus), 0)
        columns = ["name", "path", "content"]
        self.assertEqual([v.name for v in corpus.domain.metas], columns)

    def test_run_url(self):
        path = "http://file.biolab.si/text-semantics/data/predlogi-vladi-20/"
        importer = ImportDocuments(path, True)
        corpus1, _, _, _, _, _ = importer.run()
        self.assertGreater(len(corpus1), 0)

        mask = np.ones_like(corpus1.metas, dtype=bool)
        mask[:, 1] = False

        path = "http://file.biolab.si/text-semantics/data/predlogi-vladi-20////"
        importer = ImportDocuments(path, True)
        corpus2, _, _, _, _, _ = importer.run()
        self.assertGreater(len(corpus1), 0)
        np.testing.assert_array_equal(corpus1.metas[mask].tolist(),
                                      corpus2.metas[mask].tolist())

        path = "http://file.biolab.si/text-semantics/data/predlogi-vladi-20"
        importer = ImportDocuments(path, True)
        corpus3, _, _, _, _, _ = importer.run()
        self.assertGreater(len(corpus2), 0)
        np.testing.assert_array_equal(corpus1.metas[mask].tolist(),
                                      corpus3.metas[mask].tolist())

    def test_run_url_special_characters(self):
        path = "http://file.biolab.si/text-semantics/data/" \
               "elektrotehniski-vestnik-clanki/"
        importer = ImportDocuments(path, True)
        corpus, errors, _, _, _, _ = importer.run()
        self.assertGreater(len(corpus), 0)
        self.assertEqual(0, len(errors))

    def test_conllu_reader(self):
        path = os.path.join(os.path.dirname(__file__),
                            "../widgets/tests/data/conllu")
        importer = ImportDocuments(path)
        corpus, errors, lemma, pos, ner, _ = importer.run()
        self.assertEqual(len(corpus), 5)
        self.assertEqual(len(corpus), len(lemma))
        self.assertEqual(len(corpus), len(pos))
        self.assertEqual(len(corpus), len(ner))

    @patch(PATCH_METHOD, side_effect=ConnectTimeout("test message", request=""))
    def test_url_errors(self, _):
        path = "http://file.biolab.si/text-semantics/data/elektrotehniski-vestnik-clanki/"
        importer = ImportDocuments(path, True)
        corpus, errors, _, _, _, _ = importer.run()
        self.assertIsNone(corpus)
        self.assertGreater(len(errors), 0)


if __name__ == "__main__":
    unittest.main()
