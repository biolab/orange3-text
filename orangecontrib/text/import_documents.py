import asyncio
import fnmatch
import logging
import os
import pathlib
import re
import xml.etree.ElementTree as ET
from collections import namedtuple
from tempfile import NamedTemporaryFile
from types import SimpleNamespace as namespace
from typing import Callable, List, Optional, Tuple
from unicodedata import normalize

import docx2txt
import httpx
import numpy as np
import pandas as pd
import serverfiles
import yaml
from conllu import parse_incr
from odf import teletype, text
from odf.opendocument import load
from Orange.data import DiscreteVariable, Domain, StringVariable, guess_data_type
from Orange.data.io import detect_encoding, sanitize_variable
from Orange.data.util import get_unique_names
from Orange.misc.utils.embedder_utils import get_proxies
from Orange.util import Registry, dummy_callback
from pypdf import PdfReader as PyPDFReader
from requests.exceptions import ConnectionError

from orangecontrib.text.corpus import Corpus

DefaultFormats = ("docx", "odt", "txt", "pdf", "xml", "conllu")

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


class NoDocumentsException(Exception):
    pass


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
            error = pathlib.Path(self.path).name
            log.exception('Error reading failed', exc_info=ex)
        else:
            textdata = self.make_text_data()
        return textdata, error

    def read_file(self):
        raise NotImplementedError(
            "No reader for {}".format(pathlib.Path(self.path).suffix))

    def make_text_data(self):
        name = pathlib.Path(self.path).stem
        directory = pathlib.PurePath(self.path).parent
        category = directory.parts[-1] or "None"
        if self.replace_white_space:
            self.content = re.sub(r'\s+', ' ', self.content)
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
        reader = PyPDFReader(self.path)
        texts = [page.extract_text() for page in reader.pages]
        self.content = " ".join(texts)


class XmlReader(Reader):
    ext = [".xml"]

    def read_file(self):
        root = ET.parse(self.path).getroot()
        self.content = "\n".join(t.strip() for t in root.itertext() if t.strip())


class CsvMetaReader(Reader):
    ext = [".csv"]

    def read_file(self):
        self.content = pd.read_csv(self.path)


class YamlMetaReader(Reader):
    ext = [".yaml"]

    def read_file(self):
        with open(self.path, "r") as f:
            self.content = yaml.safe_load(f)
            for k in self.content:
                if self.content[k] is None:
                    self.content[k] = ""


class TsvMetaReader(Reader):
    ext = [".tsv"]

    def read_file(self):
        self.content = pd.read_csv(self.path, delimiter="\t")


def quote_url(u):
    u = u.strip()
    # Support URL with query or fragment like http://filename.txt?a=1&b=2#c=3

    def quote_byte(b):
        return chr(b) if b < 0x80 else "%{:02X}".format(b)

    return "".join(map(quote_byte, u.encode("utf-8")))


ResponseType = Tuple[Optional[Reader], Optional[TextData], Optional[str]]


def _rewrite_proxies_to_mounts(proxies):
    if proxies is None:
        return None
    return {c: httpx.AsyncHTTPTransport(proxy=url) for c, url in get_proxies().items()}


class UrlProxyReader:
    """
    A collection of functions to handle async downloading of a list of documents
    and opening the with the appropriate reader
    """

    @staticmethod
    def read_files(
        urls: List[str], callback: Callable = dummy_callback
    ) -> List[ResponseType]:
        """
        Download a list of document asynchronously

        Parameters
        ----------
        urls
            The list of documents' URLs
        callback
            Callback that is called on every successul donwload

        Returns
        -------
        List of tuples for each URL. Tuple contain the reader instance; TextData
        instance with text and other information; a document name if downloading
        was not successful
        """
        return asyncio.run(UrlProxyReader._read_files(urls, callback))

    @staticmethod
    async def _read_files(urls: List[str], callback: Callable) -> List[ResponseType]:
        proxy_mounts = _rewrite_proxies_to_mounts(get_proxies())
        async with httpx.AsyncClient(timeout=10.0, mounts=proxy_mounts) as client:
            req = [UrlProxyReader._read_file(url, client, callback) for url in urls]
            return await asyncio.gather(*req)

    @staticmethod
    async def _read_file(
        url: str, client: httpx.AsyncClient, callback: Callable
    ) -> ResponseType:
        # repeat if unsuccessful (can be due to network error)
        for _ in range(3):
            try:
                response = await client.get(quote_url(url))
                response.raise_for_status()
                callback(response.url.path)
                return UrlProxyReader._parse_response(response)
            except httpx.HTTPError:
                pass
        return None, None, url

    @staticmethod
    def _parse_response(response) -> ResponseType:
        path = pathlib.Path(response.url.path)
        extension = "".join(path.suffixes)

        with NamedTemporaryFile(suffix=extension, delete=False) as f:
            f.write(response.content)
        reader = Reader.get_reader(f.name)
        text_data, error = reader.read()
        text_data = TextData(
            path.stem,
            str(response.url),
            text_data.ext,
            path.parent.parts[-1],
            text_data.content,
        )
        os.remove(f.name)
        return reader, text_data, error


class ConlluReader(Reader):
    TextData = namedtuple(
        "Text",
        ["name", "path", "ext", "category", "doc_id", "content"]
    )

    ext = [".conllu"]

    def __init__(self, path):
        super().__init__(path)
        self.tokens = None
        self.pos = None
        self.ner = None

    @staticmethod
    def parse_ner(tokens):
        entities = []
        temp_ner = []
        for token in tokens:
            if token["misc"] is None or "NER" not in token["misc"]:
                continue
            # "0" means the token is not named entity
            if token["misc"]["NER"] != "O":
                # lemma?
                temp_ner.append(token["lemma"])
            elif temp_ner:
                entities.append(" ".join(temp_ner))
                temp_ner = []
        if temp_ner:
            entities.append(" ".join(temp_ner))
        return entities

    def read_file(self):
        content = []
        file = open(self.path, "r", encoding="utf-8")
        utterance_id = ""
        utterance = []
        tokens = []
        pos = []
        ner = []
        temp_tokens = []
        temp_pos = []
        temp_ner = []
        for sentence in parse_incr(file):
            if "newdoc id" in sentence.metadata.keys():
                if utterance_id:
                    content.append([utterance_id, " ".join(utterance)])
                    tokens.append(temp_tokens)
                    pos.append(temp_pos)
                    ner.append(temp_ner)
                    utterance = []
                    temp_tokens = []
                    temp_pos = []
                    temp_ner = []
                utterance_id = sentence.metadata["newdoc id"]
            utterance.append(sentence.metadata["text"])
            temp_tokens.extend([token["lemma"] for token in sentence])
            temp_pos.extend([token["upos"] for token in sentence])
            temp_ner.extend(self.parse_ner(sentence))
        if temp_tokens or utterance:
            content.append([utterance_id, " ".join(utterance)])
            tokens.append(temp_tokens)
            pos.append(temp_pos)
            ner.append(temp_ner)
        file.close()
        self.tokens = tokens
        self.pos = pos
        self.ner = np.array([", ".join(tokens) for tokens in ner], dtype=object)
        self.content = pd.DataFrame(content, columns=["newdoc id", "text"])

    def make_text_data(self):
        text_objects = []
        name = pathlib.Path(self.path).stem
        directory = pathlib.PurePath(self.path).parent
        category = directory.parts[-1] or "None"
        for _, row in self.content.iterrows():
            if self.replace_white_space:
                row["text"] = re.sub(r'\s+', ' ', row["text"])
            text_objects.append(self.TextData(name, self.path, self.ext,
                                              category,
                                              row["newdoc id"],
                                              row["text"]))
        return text_objects


class ImportDocuments:
    META_DATA_FILE_KEY = "Text file"
    # this is what we will merge meta data on, change to user-set variable
    CONLLU_META_DATA = ["ID", "Text_ID"]

    def __init__(
        self,
        startdir: str,
        is_url: bool = False,
        formats: Tuple[str] = DefaultFormats,
        report_progress: Callable = dummy_callback,
    ):
        if is_url and not startdir.endswith("/"):
            startdir += "/"
        elif not is_url:
            startdir = os.path.join(startdir, "")
        self.startdir = startdir
        self.formats = formats
        self._report_progress = report_progress
        self.cancelled = False
        self._is_url = is_url
        self._text_data = []
        self._meta_data: pd.DataFrame = None
        self.is_conllu = False
        self.tokens = None
        self.pos = None
        self.ner = None

    def run(self) -> Tuple[Corpus, List, List, List, List, bool]:
        file_paths, meta_paths = self._retrieve_paths()
        callback = self._shared_callback(len(file_paths) + len(meta_paths))
        self._text_data, errors_text, tokens, pos, ner, conllu = self._read_text_data(
            file_paths, callback
        )
        self._meta_data, errors_meta = self._read_meta_data(meta_paths, callback)
        self.is_conllu = conllu
        corpus = self._create_corpus()
        corpus = self._add_metadata(corpus)
        return corpus, errors_text + errors_meta, tokens, pos, ner, conllu

    def _shared_callback(self, num_all_files):
        items = iter(np.linspace(0, 1, num_all_files))

        def callback(path):
            if self.cancelled:
                raise Exception
            self._report_progress(namespace(progress=next(items), lastpath=path))

        return callback

    def _retrieve_paths(self):
        # retrieve file paths
        patterns = ["*.{}".format(fmt.lower()) for fmt in self.formats]
        scan = self.scan_url if self._is_url else self.scan
        file_paths = scan(self.startdir, include_patterns=patterns)

        # retrieve meta paths
        scan = self.scan_url if self._is_url else self.scan
        patterns = ["*.csv", "*.yaml", "*.yml", "*.tsv"]
        meta_paths = scan(self.startdir, include_patterns=patterns)

        return file_paths, meta_paths

    def _read_text_data(self, paths, callback):
        text_data = []
        errors = []
        tokens = []
        pos = []
        ner = []
        conllu = False

        if len(paths) == 0:
            raise NoDocumentsException()

        if self._is_url:
            results = UrlProxyReader().read_files(paths, callback)
        else:
            results = []
            for path in paths:
                reader = Reader.get_reader(path)
                text, error = reader.read()
                results.append((reader, text, error))
                callback(path)

        for reader, text, error in results:
            if text is not None:
                if type(reader) == ConlluReader:
                    conllu = True
                    for t in text:
                        text_data.append(t)
                    tokens.extend(reader.tokens)
                    pos.extend(reader.pos)
                    ner.extend(reader.ner)
                else:
                    conllu = False
                    text_data.append(text)
            else:
                errors.append(error)

            if self.cancelled:
                return

        return text_data, errors, tokens, pos, ner, conllu

    def _read_meta_data(self, paths, callback):
        meta_dfs, errors = [], []
        if self._is_url:
            results = UrlProxyReader().read_files(paths, callback)
        else:
            results = []
            for path in paths:
                reader = Reader.get_reader(path)
                data, error = reader.read()
                results.append((reader, data, error))
                callback(path)

        for reader, data, error in results:
            if data is not None:
                content = data.content
                if isinstance(content, dict):
                    content = pd.DataFrame(content, index=[0])
                meta_dfs.append(content)
            else:
                errors.append(error)

            if self.cancelled:
                return

        return pd.concat(meta_dfs) if meta_dfs else None, errors

    def _create_corpus(self) -> Corpus:
        corpus = None
        names = ["name", "path", "content"] if not self.is_conllu else [
            "name", "path", "utterance", "content"]
        data = []
        category_data = []
        text_categories = list(set(t.category for t in self._text_data))
        values = list(set(text_categories))
        category_var = DiscreteVariable.make("category", values=values)
        for textdata in self._text_data:
            datum = [
                # some characters are written as decomposed (č is char c
                # and separate char for caron), with NFC normalization we
                # normalize them to be written as precomposed (č is one
                # unicode char - 0x10D)
                # https://docs.python.org/3/library/unicodedata.html#unicodedata.normalize
                normalize('NFC', textdata.name),
                normalize('NFC', textdata.path),
                normalize('NFC', textdata.content)
            ]
            if self.is_conllu:
                datum.insert(2, normalize('NFC', textdata.doc_id))
            data.append(datum)
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
            corpus = Corpus.from_numpy(
                domain,
                X=np.empty((len(category_data), 0)),
                Y=category_data,
                metas=data,
                text_features=[domain.metas[-1]]
            )
        return corpus

    def _add_metadata(self, corpus: Corpus) -> Corpus:
        if (
            corpus is None
            or "path" not in corpus.domain
            or self._meta_data is None
            or (
                self.META_DATA_FILE_KEY not in self._meta_data.columns
                and not any(i in self._meta_data.columns for i in
                         self.CONLLU_META_DATA)
            )
        ):
            return corpus

        if self.is_conllu:
            # find the first matching column
            match_id = next((idx for idx in self.CONLLU_META_DATA if idx in
                             self._meta_data.columns))
            df = self._meta_data.set_index(match_id)
            path_column = corpus.get_column("utterance")
        else:
            df = self._meta_data.set_index(
                self.startdir + self._meta_data[self.META_DATA_FILE_KEY].apply(quote_url)
            )
            path_column = corpus.get_column("path")

        if len(df.index.drop_duplicates()) != len(df.index):
            df = df[~df.index.duplicated(keep='first')]
        filtered = df.reindex(path_column)
        for name, column in filtered.items():
            data = column.astype(str).values
            val_map, vals, var_type = guess_data_type(data)
            values, variable = sanitize_variable(val_map, vals, data,
                                                 var_type, {},
                                                 name=get_unique_names(
                                                     corpus.domain, name))
            corpus = corpus.add_column(
                variable,
                values,
                to_metas=True
            )
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

        paths = []

        for dirpath, dirnames, filenames in os.walk(topdir):
            for dirname in list(dirnames):
                # do not recurse into hidden dirs
                if fnmatch.fnmatch(dirname, ".*"):
                    dirnames.remove(dirname)

            filenames = [fname for fname in filenames
                         if matches_any(fname, include_patterns)
                         and not matches_any(fname, exclude_patterns)]
            paths = paths + [os.path.join(dirpath, fname) for fname in
                             filenames]
        return sorted(paths)

    @staticmethod
    def scan_url(topdir: str, include_patterns: Tuple[str] = ("*",),
                 exclude_patterns: Tuple[str] = (".*",)) -> List[str]:
        try:
            files = serverfiles.ServerFiles(topdir).listfiles()
        except ConnectionError:
            return []

        include_patterns = include_patterns or ("*",)
        paths = []
        for filename in files:
            path = topdir + "/".join(filename)
            if matches_any(path, include_patterns) and \
                    not matches_any(path, exclude_patterns):
                paths.append(path)
        return paths


def matches_any(fname: str, patterns: Tuple[str]) -> bool:
    return any(fnmatch.fnmatch(fname.lower(), pattern)
               for pattern in patterns)
