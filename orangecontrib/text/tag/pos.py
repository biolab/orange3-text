import nltk
import numpy as np

from orangecontrib.text.topics.topics import chunks

nltk.download(['averaged_perceptron_tagger', 'maxent_treebank_pos_tagger'], quiet=True)


class POSTagger:
    def __init__(self, tagger, name='POS Tagger'):
        self.tag_sents = tagger.tag_sents
        self.name = name

    def tag_corpus(self, corpus, chunk_count=None, on_progress=None):
        if chunk_count:
            tags = []
            size = np.ceil(len(corpus) / chunk_count)
            for i, chunk in enumerate(chunks(corpus.tokens, size)):
                tags.extend(self.tag_sents(chunk))
                if on_progress:
                    on_progress(100 * (i + 1) * size / len(corpus))
        else:
            tags = self.tag_sents(corpus.tokens)

        corpus.pos_tags = list(map(lambda sent: list(map(lambda x: x[1], sent)), tags))
        # corpus.store_tokens(list(map(lambda sent: list(map(lambda x: '{0[0]}_{0[1]}'.format(x), sent)), tags)))
        return corpus

    def __str__(self):
        return self.name


class StanfordPOSTagger(nltk.StanfordPOSTagger, POSTagger):
    name = 'Stanford POS Tagger'

    @classmethod
    def check(cls, path_to_model, path_to_jar):
        try:
            cls(path_to_model, path_to_jar).tag(())
        except OSError as e:
            raise ValueError('Either Java SDK not installed or some of the files are invalid.\n' + str(e))
        except LookupError as e:
            raise ValueError(str(e).strip(' =\n'))


taggers = [
    POSTagger(nltk.PerceptronTagger(), 'Averaged Perceptron Tagger'),
    POSTagger(nltk.data.load('taggers/maxent_treebank_pos_tagger/english.pickle'),
              'Treebank Part of Speech Tagger (Maximum entropy)'),
]
