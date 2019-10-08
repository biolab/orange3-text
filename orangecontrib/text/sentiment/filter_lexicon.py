import os


class FilterSentiment:

    @staticmethod
    def read_file(file):
        with open(file, 'r') as f:
            return f.read().split('\n')


class SloSentiment(FilterSentiment):

    resources_folder = os.path.dirname(__file__)

    @classmethod
    def positive(cls):
        f = os.path.join(cls.resources_folder,
                         'resources/positive_words_Slolex.txt')
        return cls.read_file(f)

    @classmethod
    def negative(cls):
        f = os.path.join(cls.resources_folder,
                         'resources/negative_words_Slolex.txt')
        return cls.read_file(f)
