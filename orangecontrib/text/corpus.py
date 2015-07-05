import os

import numpy as np
from Orange.data import StringVariable, Table, DiscreteVariable, Domain


def get_sample_corpora_dir():
    path = os.path.dirname(__file__)
    dir = os.path.join(path, 'datasets')
    return os.path.realpath(dir)


class Corpus(Table):
    """
        Internal class for storing a corpus of orangecontrib.text.corpus.Corpus.
    """

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(Corpus)

    def __init__(self, documents, X, Y, metas, domain):
        self.documents = documents
        if X is not None:
            self.X = X
        else:
            self.X = np.zeros((len(documents), 0))
        if Y is not None:
            self.Y = Y
        else:
            self.Y = np.zeros((len(documents), 0))
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
        if len(include_ids) == 0:
            include_ids.append(first_id)

        documents = []
        for line in range(table.metas.shape[0]):
            documents.append(' '.join(table.metas[line, include_ids]))

        corp = cls(documents, table.X, table.Y, table.metas, table.domain)
        corp.used_features = [f for i, f in enumerate(table.domain.metas) if i in include_ids]
        return corp

    def regenerate_documents(self, selected_features):
        documents = []
        indices = [self.domain.metas.index(f) for f in selected_features]
        for line in range(self.metas.shape[0]):
            documents.append(' '.join(self.metas[line, indices]))
        self.documents = documents

    def extend_corpus(self, documents, metadata, class_values):
        # TODO check if Domains match!
        self.metas = np.vstack((self.metas, metadata))
        self.documents += documents

        for val in set(class_values):
            if val not in self.domain.class_var.values:
                self.domain.class_var.add_value(val)
        new_Y = np.array([self.domain.class_var.to_val(cv) for cv in class_values])[:, None]
        self._Y = np.vstack((self._Y, new_Y))

        self.X = self.W = np.zeros((len(self.documents), 0))
        Table._init_ids(self)

    def __len__(self):
        return len(self.documents)
