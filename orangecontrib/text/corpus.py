import os
import codecs

import numpy as np
from Orange.data import Domain, StringVariable, Table, DiscreteVariable, ContinuousVariable


def get_sample_corpora_dir():
    path = os.path.dirname(__file__)
    dir = os.path.join(path, 'datasets')
    return os.path.realpath(dir)


def documents_to_numpy(documents, metadata):
    metadata2 = dict()
    for md in metadata:
        for key in md.keys():
            metadata2.setdefault(key, set()).add(md[key])

    meta_vars = [StringVariable('text')]
    for key in sorted(metadata2.keys()):
        if len(metadata2[key]) < 21:
            meta_vars.append(DiscreteVariable(key, values=list(metadata2[key])))
        else:
            meta_vars.append(StringVariable(key))

    metas = [[None] * len(meta_vars) for x in range(len(documents))]
    for i, md in enumerate(metadata):
        for j, var in enumerate(meta_vars):
            if var.name in md:
                metas[i][j] = var.to_val(md[var.name])

    text = np.array(documents).reshape(len(documents), 1)
    metas = np.array(metas, dtype=object)

    return np.hstack((text, metas)), meta_vars


class Corpus(Table):
    """
        Internal class for storing a corpus of orangecontrib.text.corpus.Corpus.
    """

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(Corpus)

    def __init__(self, documents, metadata=[]):
        self.documents = documents

        metas, meta_vars = documents_to_numpy(documents, metadata)

        self.metas = metas
        self.X = self._Y = self.W = np.zeros((len(documents), 0))
        self.domain = Domain([], metas=meta_vars)
        Table._init_ids(self)

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        c = super().from_table(domain, source, row_indices)
        c.documents = source.documents
        c._Y = c._Y.astype(np.float64)
        return c

    def get_number_of_categories(self):
        return len(self.domain['category'].values)

    @classmethod
    def from_file(cls, filename):
        documents = []
        metadata = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            header = f.readline().strip().split('\t')
            if header.count('text') != 1:
                raise RuntimeError("File should contain exactly one column labeled 'text'.")
            text_ind = header.index('text')
            f.readline()

            for line in f:
                fields = line.strip().split("\t")
                if len(fields) != len(header):
                    raise RuntimeError("All lines should contain the same number of fields as a header.")
                documents.append(fields[text_ind])
                metadata.append({header[i]: fields[i] for i in range(len(fields)) if i != text_ind})
        return cls(documents, metadata)

    def extend_corpus(self, documents, metadata=[]):
        metas, meta_vars = documents_to_numpy(documents, metadata)

        # TODO check if Domains match!

        self.metas = np.vstack((self.metas, metas))
        self.documents += documents

        self.X = self._Y = self.W = np.zeros((len(self.documents), 0))
        Table._init_ids(self)

    def __len__(self):
        return len(self.documents)
