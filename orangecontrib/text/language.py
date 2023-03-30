from collections import Counter
from typing import Optional

from AnyQt.QtCore import Qt
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from Orange.data import DiscreteVariable
from orangewidget.utils.itemmodels import PyListModel

# languages supported by at least one method in Orange3-text
# language dependent methods: YAKE!, nltk - stopwords, sentiment methods,
# normalizers, embedding
ISO2LANG = {
    "af": "Afrikaans",
    "am": "Amharic",
    "ar": "Arabic",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bo": "Tibetan",
    "br": "Breton",
    "bs": "Bosnian",
    "ca": "Catalan",
    "ckb": "Central Kurdish",
    "cop": "Coptic",
    "cs": "Czech",
    "cu": "Old Church Slavonic",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "dv": "Divehi",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "eu": "Basque",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "gl": "Galician",
    "got": "Gothic",
    "grc": "Ancient greek",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hi-Latn": "Hindi (latin)",
    "hr": "Croatian",
    "ht": "Haitian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "ka": "Georgian",
    "kk": "Kazakh",
    "km": "Central Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lo": "Lao",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "my": "Burmese",
    "nb": "Norwegian Bokmål",
    "ne": "Nepali",
    "nl": "Dutch",
    "nn": "Norwegian Nynorsk",
    "no": "Norwegian",
    "or": "Oriya",
    "pa": "Punjabi",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sa": "Sanskrit",
    "sd": "Sindhi",
    "si": "Sinhala",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sr": "Serbian",
    "sv": "Swedish",
    "ta": "Tamil",
    "te": "Telugu",
    "tg": "Tajik",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "ug": "Uyghur",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zh_char": "Chinese - Chinese Characters",
    None: None,
}
LANG2ISO = {lang: code for code, lang in ISO2LANG.items()}
DEFAULT_LANGUAGE = "English"


class LanguageModel(PyListModel):
    """Model for language selection dropdowns in the widgets"""

    def __init__(self):
        languages = sorted(filter(None, ISO2LANG.values()))
        super().__init__(iterable=[None] + languages)

    def data(self, index, role=Qt.DisplayRole):
        if index.row() == 0 and role == Qt.DisplayRole:
            return "(no language)"
        else:
            return super().data(index, role)


DetectorFactory.seed = 0
MAX_DOCS = 50  # max number of documents considered for language detection
MAX_WORDS = 2000  # max number of words in document considered for lang detection


def detect_language(corpus: "Corpus") -> Optional[str]:
    """
    Detect the language in the corpus

    Parameters
    ----------
    corpus
        Corpus to detect the language

    Returns
    -------
    Detected language ISO code or None if language not detected
    """
    texts = corpus.documents[:MAX_DOCS]
    texts = [" ".join(t.replace("\n", " ").split(" ")[:MAX_WORDS]) for t in texts]
    languages = list()
    for text in texts:
        try:
            languages.append(detect(text))
        except LangDetectException:
            languages.append(None)
    # filter out languages not in supported by Orange
    candidates = [l for l, _ in Counter(languages).most_common() if l in ISO2LANG]
    return candidates[0] if candidates else None


def infer_language_from_variable(variable: DiscreteVariable) -> Optional[str]:
    """
    Infer language from DiscreteVariable that holds documents' language information.
    If documents have different language return None

    Parameters
    ----------
    variable
        The DiscreteVariable to infer language from it's values

    Returns
    -------
    Language ISO code if all documents have the same language, None otherwise
    """
    return variable.values[0] if len(variable.values) == 1 else None
