import os


class opinion_lexicon_slo:

    resources_folder = os.path.dirname(__file__)

    @classmethod
    def positive(cls):
        with open(os.path.join(cls.resources_folder,
                               'resources/positive_words_Slolex.txt'),
                  'r') as f:
            return f.read().split('\n')

    @classmethod
    def negative(cls):
        with open(os.path.join(cls.resources_folder,
                               'resources/negative_words_Slolex.txt'),
                  'r') as f:
            return f.read().split('\n')
