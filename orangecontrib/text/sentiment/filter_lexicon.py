import fnmatch
import os
import pickle


class FilterSentiment:

    @staticmethod
    def read_file(file):
        if fnmatch.fnmatch(file, '*.pickle'):
            with open(file, 'rb') as f:
                return pickle.loads(f.read())
        else:
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
