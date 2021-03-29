import unittest

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
        print(paths)

    def test_scan_url_txt(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path, include_patterns=["*.txt"])
        print(paths)

    def test_scan_url_csv(self):
        path = "http://file.biolab.si/text-semantics/data/"
        importer = ImportDocuments(path, True)
        paths = importer.scan_url(path, include_patterns=["*.csv"])
        print(paths)

    def test_run_url(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True)
        res, err = importer.run()
        print(res)

    def test_run_url_metadata(self):
        path = "http://file.biolab.si/text-semantics/data/semeval/"
        importer = ImportDocuments(path, True, formats=["csv"])
        res, err = importer.run()
        print(res)
        print(err)


if __name__ == "__main__":
    unittest.main()
