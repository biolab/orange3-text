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
