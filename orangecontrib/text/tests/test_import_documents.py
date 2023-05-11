import asyncio
import os
import unittest
from os.path import join, splitext
from tempfile import NamedTemporaryFile
from unittest.mock import patch, MagicMock, call

import numpy as np
import pandas as pd
import yaml
from httpx import URL, ConnectTimeout

from orangecontrib.text.import_documents import (
    ImportDocuments,
    UrlProxyReader,
    TxtReader,
    TextData,
    XmlReader,
)


PATCH_METHOD = "httpx.AsyncClient.get"
SF_LIST = "orangecontrib.text.import_documents.serverfiles.ServerFiles.listfiles"


async def dummy_post(_, url):
    await asyncio.sleep(0)
    return DummyResponse(url)


class DummyResponse:
    def __init__(self, url, content=b"lorem ipsum"):
        super().__init__()
        self.url = URL(url)
        self.content = content

    raise_for_status = MagicMock()


def dummy_yaml(file_name):
    nn = splitext(file_name)[0] + ".txt"
    return yaml.dump({"Title": "Test Doc", ImportDocuments.META_DATA_FILE_KEY: nn})


TEXTS_URL = "http://dummyserver.com/special/"
TEXTS_FILES = [
    [f] for f in ("a.txt", "b.txt", "c.txt", "a.yaml", "b.yaml", "c.test", "meta.csv")
]
TEXTS_RESPONSES = [
    DummyResponse(join(TEXTS_URL, f[0]), b"test")
    for f in TEXTS_FILES
    if f[0].endswith(".txt")
]
EXAMPLE_YAMLS = [
    (f[0], dummy_yaml(f[0])) for f in TEXTS_FILES if f[0].endswith(".yaml")
]
TEXTS_YAML_RESPONSES = [
    DummyResponse(join(TEXTS_URL, f), r.encode("utf-8")) for f, r in EXAMPLE_YAMLS
]

SPECIAL_CHAR_URL = "http://dummyserver.com/special/"
SPECIAL_CHAR_FILES = [[f] for f in ("č.txt", "š.txt", "ž.txt")]
SPECIAL_CHAR_RESPONSES = [
    DummyResponse(join(SPECIAL_CHAR_URL, f[0]), b"test") for f in SPECIAL_CHAR_FILES
]


class TestUrlProxyReader(unittest.TestCase):
    @patch(PATCH_METHOD, dummy_post)
    def test_init(self):
        path = "http://dummy.server.com/data/foo.txt"
        result = UrlProxyReader.read_files([path])
        reader, text_data, error = result[0]
        self.assertEqual(text_data.path, path)
        self.assertEqual("", error)
        self.assertIsInstance(reader, TxtReader)

    resp = DummyResponse("http://dummyserver.com/semval/C-1.txt", b"Test 1")

    @patch(PATCH_METHOD, return_value=resp)
    def test_read(self, _):
        path = "http://dummyserver.com/semval/C-1.txt"
        result = UrlProxyReader.read_files([path])
        reader, textdata, error = result[0]
        self.assertIsInstance(textdata, TextData)
        self.assertEqual(textdata.name, "C-1")
        self.assertEqual(textdata.path, path)
        self.assertEqual(textdata.ext, [".txt"])
        self.assertEqual(textdata.category, "semval")
        self.assertEqual("Test 1", textdata.content)
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
    @patch(SF_LIST, return_value=TEXTS_FILES)
    def test_scan_url(self, _):
        importer = ImportDocuments(TEXTS_URL, True)
        paths = importer.scan_url(TEXTS_URL)
        self.assertEqual(7, len(paths))
        self.assertListEqual([join(TEXTS_URL, f[0]) for f in TEXTS_FILES], paths)

    @patch(SF_LIST, return_value=TEXTS_FILES)
    def test_scan_url_txt(self, _):
        importer = ImportDocuments(TEXTS_URL, True)
        paths = importer.scan_url(TEXTS_URL, include_patterns=["*.txt"])
        self.assertEqual(3, len(paths))
        self.assertListEqual(
            [join(TEXTS_URL, f[0]) for f in TEXTS_FILES if f[0].endswith(".txt")], paths
        )

    @patch(SF_LIST, return_value=TEXTS_FILES)
    def test_scan_url_csv(self, _):
        importer = ImportDocuments(TEXTS_URL, True)
        paths = importer.scan_url(TEXTS_URL, include_patterns=["*.csv"])
        self.assertEqual(1, len(paths))
        self.assertListEqual(
            [join(TEXTS_URL, f[0]) for f in TEXTS_FILES if f[0].endswith(".csv")], paths
        )

    @patch(SF_LIST, return_value=TEXTS_FILES)
    @patch(PATCH_METHOD, side_effect=TEXTS_YAML_RESPONSES)
    def test_read_meta_data_url(self, _, __):
        importer = ImportDocuments(TEXTS_URL, True)
        _, meta_paths = importer._retrieve_paths()
        self.assertListEqual(
            [
                join(TEXTS_URL, f[0])
                for f in TEXTS_FILES
                if f[0].endswith((".yaml", ".csv"))
            ],
            meta_paths,
        )
        meta_paths.remove(join(TEXTS_URL, "meta.csv"))
        callback = importer._shared_callback(len(meta_paths))
        data1, err = importer._read_meta_data(meta_paths, callback)
        self.assertIsInstance(data1, pd.DataFrame)
        expected = pd.DataFrame(
            {
                "Text file": ["a.txt", "b.txt"],
                "Title": ["Test Doc", "Test Doc"],
            }
        )
        pd.testing.assert_frame_equal(data1.reset_index(drop=True), expected)
        self.assertEqual(len(err), 0)

    @patch(SF_LIST, return_value=TEXTS_FILES)
    @patch(PATCH_METHOD, side_effect=TEXTS_RESPONSES + TEXTS_YAML_RESPONSES)
    def test_merge_metadata_url(self, _, __):
        importer = ImportDocuments(TEXTS_URL, True)
        file_paths, meta_paths = importer._retrieve_paths()
        meta_paths.remove(join(TEXTS_URL, "meta.csv"))
        callback = importer._shared_callback(len(file_paths) + len(meta_paths))
        text_data, _, _, _, _, _ = importer._read_text_data(file_paths, callback)
        meta_data, _ = importer._read_meta_data(meta_paths, callback)

        importer._text_data = text_data
        importer._meta_data = meta_data
        corpus = importer._create_corpus()
        corpus = importer._add_metadata(corpus)
        self.assertGreater(len(corpus), 0)
        columns = ["name", "path", "content", "Text file", "Title"]
        self.assertEqual([v.name for v in corpus.domain.metas], columns)

        importer._text_data = text_data
        importer._meta_data = None
        corpus = importer._create_corpus()
        corpus = importer._add_metadata(corpus)
        self.assertGreater(len(corpus), 0)
        columns = ["name", "path", "content"]
        self.assertEqual([v.name for v in corpus.domain.metas], columns)

    @patch(SF_LIST, side_effect=[TEXTS_FILES[:-1]] * 6)
    @patch(PATCH_METHOD, side_effect=TEXTS_RESPONSES + TEXTS_YAML_RESPONSES)
    def test_run_url(self, post_mock, sf_mock):
        importer = ImportDocuments(TEXTS_URL, True)
        corpus1, _, _, _, _, _ = importer.run()
        self.assertEqual(3, len(corpus1))
        post_mock.assert_has_calls(
            [call(join(TEXTS_URL, f[0])) for f in TEXTS_FILES[:-2]]
        )

        mask = np.ones_like(corpus1.metas, dtype=bool)
        mask[:, 1] = False

        new_text_reponses = [
            DummyResponse(join(TEXTS_URL + "///", f[0]), b"test")
            for f in TEXTS_FILES
            if f[0].endswith(".txt")
        ]
        post_mock.reset_mock()

        post_mock.side_effect = new_text_reponses + TEXTS_YAML_RESPONSES
        importer = ImportDocuments(TEXTS_URL + "///", True)
        corpus2, _, _, _, _, _ = importer.run()
        post_mock.assert_has_calls(
            [call(join(TEXTS_URL + "///", f[0])) for f in TEXTS_FILES[:-2]]
        )
        self.assertEqual(3, len(corpus1))
        np.testing.assert_array_equal(corpus1.metas[mask].tolist(),
                                      corpus2.metas[mask].tolist())

        post_mock.side_effect = TEXTS_RESPONSES + TEXTS_YAML_RESPONSES
        importer = ImportDocuments(TEXTS_URL[:-1], True)
        corpus3, _, _, _, _, _ = importer.run()
        post_mock.assert_has_calls(
            [call(join(TEXTS_URL, f[0])) for f in TEXTS_FILES[:-2]]
        )
        self.assertEqual(3, len(corpus1))
        np.testing.assert_array_equal(corpus1.metas.tolist(), corpus3.metas.tolist())

    @patch(SF_LIST, return_value=SPECIAL_CHAR_FILES)
    @patch(PATCH_METHOD, side_effect=SPECIAL_CHAR_RESPONSES)
    def test_run_url_special_characters(self, _, __):
        importer = ImportDocuments(SPECIAL_CHAR_URL, True)
        corpus, errors, _, _, _, _ = importer.run()
        self.assertEqual(3, len(corpus))
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

    @patch(SF_LIST, return_value=SPECIAL_CHAR_FILES)
    @patch(PATCH_METHOD, side_effect=ConnectTimeout("test message", request=""))
    def test_url_errors(self, _, __):
        importer = ImportDocuments(TEXTS_URL, True)
        corpus, errors, _, _, _, _ = importer.run()
        self.assertIsNone(corpus)
        self.assertGreater(len(errors), 0)


XML_EXAMPLE = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<root testAttr="testValue">
    The Tree
    <children>
        <child name="Jack">First</child>
        <child name="Rose">Second</child>
        <child name="Blue Ivy">
            Third
                <grandchildren>
                    <data>One</data>
                    <data>Two</data>
                    <unique>Twins</unique>
                </grandchildren>
            </child>
        <child name="Jane">Fourth</child>
    </children>
    After
</root>"""


class TestXMLReader(unittest.TestCase):
    def test_file(self):
        exp = "The Tree\nFirst\nSecond\nThird\nOne\nTwo\nTwins\nFourth\nAfter"
        with NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write(XML_EXAMPLE)
        reader = XmlReader(fp.name)
        res = reader.read()[0]
        self.assertEqual(exp, res.content)
        os.remove(fp.name)

    def test_error(self):
        with NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write("Test")
        reader = XmlReader(fp.name)
        res = reader.read()
        self.assertIsNone(res[0])
        self.assertEqual(fp.name.split(os.sep)[-1], res[1])
        os.remove(fp.name)


if __name__ == "__main__":
    unittest.main()
