from typing import List, Tuple

from nltk.corpus import stopwords

from orangecontrib.text.keywords.rake import Rake
from orangecontrib.text.preprocess import StopwordsFilter


# all available languages for RAKE
RAKE_LANGUAGES = StopwordsFilter.supported_languages()


def rake(
    texts: List[str], language: str, max_len: int
) -> List[List[Tuple[str, float]]]:
    """
    Extract keywords from text with RAKE method.

    Parameters
    ----------
    texts
        List of texts from which keywords are extracted
    language
        The language of texts
    max_len
        Maximal length of keywords/keyphrases extracted

    Returns
    -------
    List which contains list of keywords for each of the documents in texts.
    For each keyword function returns a tuple with keyword and it's score.
    """
    if language.lower() not in [l.lower() for l in RAKE_LANGUAGES]:
        raise ValueError(f"Language must be one of: {RAKE_LANGUAGES}")

    stop_words_ = [x.strip() for x in stopwords.words(language.lower())]
    rake_object = Rake(stop_words_, max_words_length=max_len)
    kws = [rake_object.run(text) for text in texts]
    return kws


if __name__ == "__main__":
    print(rake(["sample text"], "english", max_len=3))