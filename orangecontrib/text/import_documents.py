import fnmatch
import logging
import os
import pathlib
import re

from collections import namedtuple
from types import SimpleNamespace as namespace

import numpy as np

import docx2txt
from odf.opendocument import load
from odf import text, teletype

from pdfminer.pdfparser import PDFParser, PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from bs4 import BeautifulSoup

from Orange.data import DiscreteVariable, Domain, StringVariable
from Orange.data.io import detect_encoding
from Orange.util import Registry

from orangecontrib.text.corpus import Corpus


DefaultFormats = ("docx", "odt", "txt", "pdf", "xml")

TextData = namedtuple(
    "Text",
    ["name", "path", "ext", "category", "content"]
)
TextData.isvalid = property(lambda self: True)

TextDataError = namedtuple(
    "TextDataError",
    ["path", "error", "error_str"]
)
TextDataError.isvalid = property(lambda self: False)

log = logging.getLogger(__name__)


class Reader(metaclass=Registry):
    def __init__(self, path, replace_white_space=False):
        self.path = path
        self.replace_white_space = replace_white_space
        self.content = None

    @classmethod
    def get_reader(cls, path):
        ext = pathlib.Path(path).suffix
        for _reader in cls.registry:
            reader = eval(_reader)
            if ext in reader.ext:
                return reader(path)
        return Reader(path)

    def read(self, ):
        error = ""
        try:
            self.read_file()
        except Exception as ex:
            textdata = None
            error = "{}".format(pathlib.Path(self.path).name)
            log.exception('Error reading failed', exc_info=ex)
        else:
            textdata = self.make_text_data()
        return textdata, error

    def read_file(self):
        raise NotImplementedError("No reader for {}".format(pathlib.Path(self.path).suffix))

    def make_text_data(self):
        name = pathlib.Path(self.path).stem
        directory = pathlib.PurePath(self.path).parent
        category = directory.parts[-1] or "None"
        if self.replace_white_space:
            self.content = re.sub('\s+', ' ', self.content)
        return TextData(name, self.path, self.ext, category, self.content)


class TxtReader(Reader):
    ext = [".txt"]

    def read_file(self):
        encoding = detect_encoding(self.path)
        with open(self.path, 'r', encoding=encoding) as f:
            self.content = f.read()


class DocxReader(Reader):
    ext = [".docx"]

    def read_file(self):
        self.content = docx2txt.process(self.path)


class OdtReader(Reader):
    ext = [".odt"]

    def read_file(self):
        odtfile = load(self.path)
        texts = odtfile.getElementsByType(text.P)
        self.content = " ".join(teletype.extractText(t) for t in texts)


class PdfReader(Reader):
    """
    char_margin — two text chunks whose distance is closer than this value are considered
    contiguous and get grouped into one.
    word_margin — it may be required to insert blank characters (spaces) as necessary if
    the distance between two words is greater than this value, as a blank between words might
    not be represented as a space, but indicated by the positioning of each word.
    """
    ext = [".pdf"]

    def read_file(self):
        with open(self.path, 'rb') as f:
            parser = PDFParser(f)
        doc = PDFDocument()
        parser.set_document(doc)
        doc.set_parser(parser)
        doc.initialize('')
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        laparams.char_margin = 0.1
        laparams.word_margin = 1.0
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        extracted_text = []

        for page in doc.get_pages():
            interpreter.process_page(page)
            layout = device.get_result()
            for lt_obj in layout:
                if isinstance(lt_obj, LTTextBox) or isinstance(lt_obj, LTTextLine):
                    extracted_text.append(lt_obj.get_text())
        self.content = ' '.join(extracted_text)


class XmlReader(Reader):
    ext = [".xml"]

    def read_file(self):
        encoding = detect_encoding(self.path)
        with open(self.path, encoding=encoding, errors='ignore') as markup:
            soup = BeautifulSoup(markup.read(), "lxml")
        self.content = soup.get_text()


class ImportDocuments:
    def __init__(self, startdir, formats=DefaultFormats, report_progress=None):
        self.startdir = startdir
        self.formats = formats
        self._report_progress = report_progress
        self.cancelled = False
        self._text_data = []

    def run(self):
        text_data = []
        errors = []
        patterns = ["*.{}".format(fmt.lower()) for fmt in self.formats]
        paths = self.scan(self.startdir, include_patterns=patterns)
        n_paths = len(paths)
        batch = []

        for path in paths:
            if len(batch) == 1 and self._report_progress is not None:
                self._report_progress(
                    namespace(progress=len(text_data) / n_paths,
                              lastpath=path,
                              batch=batch))
                batch = []

            reader = Reader.get_reader(path)
            text, error = reader.read()
            if text is not None:
                text_data.append(text)
                batch.append(text_data)
            else:
                errors.append(error)

            if self.cancelled:
                return

        self._text_data = text_data
        return self._create_corpus(), errors

    def _create_corpus(self):
        corpus = None
        names = ["name", "path", "content"]
        data = []
        category_data = []
        text_categories = list(set(t.category for t in self._text_data))
        values = list(set(text_categories))
        category_var = DiscreteVariable.make("category", values=values)
        for textdata in self._text_data:
            data.append(
                [textdata.name,
                 textdata.path,
                 textdata.content]
            )
            category_data.append(category_var.to_val(textdata.category))
        if len(text_categories) > 1:
            category_data = np.array(category_data)
        else:
            category_var = []
            category_data = np.empty((len(data), 0))
        domain = Domain(
            [], category_var, [StringVariable.make(name) for name in names]
        )
        domain["name"].attributes["title"] = True
        data = np.array(data, dtype=object)
        if len(data):
            corpus = Corpus(domain,
                            Y=category_data,
                            metas=data,
                            text_features=[domain.metas[2]])

        return corpus

    @staticmethod
    def scan(topdir, include_patterns=("*",), exclude_patterns=(".*",)):
        """
        Yield file system paths under `topdir` that match include/exclude patterns

        Parameters
        ----------
        topdir: str
            Top level directory path for the search.
        include_patterns: List[str]
            `fnmatch.fnmatch` include patterns.
        exclude_patterns: List[str]
            `fnmatch.fnmatch` exclude patterns.

        Returns
        -------
        list of paths
        """
        if include_patterns is None:
            include_patterns = ["*"]

        def matches_any(fname, patterns):
            return any(fnmatch.fnmatch(fname.lower(), pattern)
                       for pattern in patterns)

        paths = []

        for dirpath, dirnames, filenames in os.walk(topdir):
            for dirname in list(dirnames):
                # do not recurse into hidden dirs
                if fnmatch.fnmatch(dirname, ".*"):
                    dirnames.remove(dirname)

            filenames = [fname for fname in filenames
                         if matches_any(fname, include_patterns)
                            and not matches_any(fname, exclude_patterns)]
            paths = paths + [os.path.join(dirpath, fname) for fname in filenames]
        return paths
