import os
import codecs

import numpy as np
from Orange.data import Domain, StringVariable, Table, DiscreteVariable, ContinuousVariable


def get_sample_corpora_dir():
    path = os.path.dirname(__file__)
    dir = os.path.join(path, 'datasets')
    return os.path.realpath(dir)


class Corpus(Table):
    """
        Internal class for storing a corpus of orangecontrib.text.corpus.Document.
    """

    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(Corpus)

    def __init__(self, documents, metadata=None):
        self.documents = documents

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

        self.metas = np.hstack((text, metas))
        self.X = self._Y = self.W = np.zeros((len(documents), 0))
        self.domain = Domain([], metas=[StringVariable('Text')] + meta_vars)
        Table._init_ids(self)

    def get_number_of_categories(self):
        return len(self.domain['category'].values)

    @classmethod
    def from_file(cls, filename):
        documents = []
        metadata = []
        with codecs.open(filename, 'r', 'utf-8') as f:
            for line in f:
                category, text = line.strip().split("\t")
                documents.append(text)
                metadata.append(dict(category=category))
        return cls(documents, metadata)

    def __len__(self):
        return len(self.documents)


class Document:
    """
        A class holding the data of a single document.
    """
    def __init__(self, text, category):
        """
        :param text: The text of the document.
        :type text: string
        :param category: The type of the document.
        :type category: string
        :return: :class: `orangecontrib.text.corpus.Document`
        """
        self.text = text
        self.category = category
        self.tokens = None
