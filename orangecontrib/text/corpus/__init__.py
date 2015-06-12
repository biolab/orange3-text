import os

def get_sample_corpora_dir():
    path = os.path.dirname(__file__)
    dir = os.path.join(path, '..', 'datasets')
    return os.path.realpath(dir)

class Corpus:
    """
        Internal class for storing a corpus of orangecontrib.text.corpus.Document.
    """
    def __init__(self, path):
        file = open(path, 'r')
        self.documents = []
        for line in file.readlines():
            category, text = line.strip().split("\t")
            self.documents.append(Document(text, category))

    def get_number_of_categories(self):
        return len(set([d.category for d in self.documents]))

    def get_number_of_documents(self):
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
