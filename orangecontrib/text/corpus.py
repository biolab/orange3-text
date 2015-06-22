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

    meta_vars = []
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
    meta_vars.insert(0, StringVariable('text'))
    return np.hstack((text, metas)), meta_vars


class Corpus(Table):
    """
        Internal class for storing a corpus of orangecontrib.text.corpus.Corpus.
    """

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(Corpus)

    def __init__(self, documents, X, Y, metas, domain):
        self.documents = documents
        self.X = X
        self.Y = Y
        self.metas = metas
        self.W = np.zeros((len(documents), 0))
        self.domain = domain
        Table._init_ids(self)

    @classmethod
    def from_table(cls, domain, source, row_indices=...):
        c = super().from_table(domain, source, row_indices)
        c.documents = source.documents
        c._Y = c._Y.astype(np.float64)
        return c

    @classmethod
    def from_file(cls, filename):
        table = Table.from_file(filename)
        include_ids = []
        first_id = None
        for i, attr in enumerate(table.domain.metas):
            if isinstance(attr, StringVariable):
                if first_id is None:
                    first_id = i
                if attr.attributes.get('include', 'False') == 'True':
                    include_ids.append(i)
        documents = []
        if len(include_ids) > 0:
            for line in range(table.metas.shape[0]):
                documents.append(' '.join(table.metas[line, include_ids]))
        else:
            documents = table.metas[:, first_id].tolist()
        return cls(documents, table.X, table.Y, table.metas, table.domain)

    def extend_corpus(self, documents, metadata=[]):
        metas, meta_vars = documents_to_numpy(documents, metadata)

        # TODO check if Domains match!

        self.metas = np.vstack((self.metas, metas))
        self.documents += documents

        self.X = self._Y = self.W = np.zeros((len(self.documents), 0))
        Table._init_ids(self)

    def __len__(self):
        return len(self.documents)
