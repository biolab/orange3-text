class Corpus:
    """
        Internal class for storing a corpus of orangecontrib.text.corpus.Document.
    """
    def __init__(self, path):
        file = open(path, 'r')
        self.documents = []
        for line in file.readlines():
            type, text = line.strip().split("\t")
            self.documents.append(Document(text, type))


class Document:
    """
        TODO
    """
    def __init__(self, text, type):
        """
        :param text: The text of the document.
        :type text: string
        :param type: The type of the document.
        :type type: string
        :return: :class: `orangecontrib.text.corpus.Document`
        """
        self.text = text
        self.type = type
        self.tokens = None
