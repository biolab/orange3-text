from functools import wraps
from math import ceil
from typing import Union, List

import numpy as np
import scipy.sparse as sp
from gensim.matutils import Sparse2Corpus
from scipy.sparse import csc_matrix


def chunks(iterable, chunk_size):
    """ Splits iterable objects into chunk of fixed size.
    The last chunk may be truncated.
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def chunkable(method):
    """ This decorators wraps methods that can be executed by passing data by chunks.

    It allows you to pass additional arguments like `chunk_number` and `on_progress` callback
    to monitor the execution's progress.

    Note:
        If no callback is provided data won't be splitted.
    """

    @wraps(method)
    def wrapper(self, data, chunk_number=100, on_progress=None, *args, **kwargs):
        if on_progress:
            chunk_size = ceil(len(data) / chunk_number)
            progress = 0
            res = []
            for i, chunk in enumerate(chunks(data, chunk_size=chunk_size)):
                chunk_res = method(self, chunk, *args, **kwargs)
                if chunk_res:
                    res.extend(chunk_res)

                progress += len(chunk)
                on_progress(progress/len(data))
        else:
            res = method(self, data, *args, **kwargs)

        return res

    return wrapper


def np_sp_sum(x, axis=None):
    """ Wrapper for summing either sparse or dense matrices.
    Required since with scipy==0.17.1 np.sum() crashes."""
    if sp.issparse(x):
        r = x.sum(axis=axis)
        if axis is not None:
            r = np.array(r).ravel()
        return r
    else:
        return np.sum(x, axis=axis)


class Sparse2CorpusSliceable(Sparse2Corpus):
    """
    Sparse2Corpus support only retrieving a vector for single document.
    This class implements slice operation on the Sparse2Corpus object.

    Todo: this implementation is temporary, remove it when/if implemented in gensim
    """

    def __getitem__(
        self, key: Union[int, List[int], np.ndarray, type(...), slice]
    ) -> Sparse2Corpus:
        """Retrieve a document vector from the corpus by its index.

        Parameters
        ----------
        key
            Index of document or slice for documents

        Returns
        -------
        Selected subset of sparse data from self.
        """
        sparse = self.sparse.__getitem__((slice(None, None, None), key))
        return Sparse2CorpusSliceable(sparse)


# ISO language codes from https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes
LANGUAGES = [
    'Abkhazian', 'Afar', 'Afrikaans', 'Akan', 'Albanian', 'Amharic', 'Arabic',
    'Aragonese', 'Armenian', 'Assamese', 'Avaric', 'Avestan', 'Aymara',
    'Azerbaijani', 'Bambara', 'Bashkir', 'Basque', 'Belarusian', 'Bengali',
    'Bislama', 'Bosnian', 'Breton', 'Bulgarian', 'Burmese', 'Catalan', 'Chamorro',
    'Chechen', 'Chichewa', 'Chinese', 'Church Slavic', 'Chuvash', 'Cornish', 'Corsican',
    'Cree', 'Croatian', 'Czech', 'Danish', 'Divehi', 'Dutch', 'Dzongkha',
    'English', 'Esperanto', 'Estonian', 'Ewe', 'Faroese', 'Fijian', 'Finnish',
    'French', 'Western', 'Fulah', 'Gaelic', 'Galician', 'Ganda', 'Georgian', 'German',
    'Greek', 'Kalaallisut', 'Guarani', 'Gujarati', 'Haitian', 'Hausa', 'Hebrew', 'Herero',
    'Hindi', 'Hiri', 'Hungarian', 'Icelandic', 'Ido', 'Igbo', 'Indonesian', 'Interlingua',
    'Interlingue', 'Inuktitut', 'Inupiaq', 'Irish', 'Italian', 'Japanese', 'Javanese',
    'Kannada', 'Kanuri', 'Kashmiri', 'Kazakh', 'Central', 'Kikuyu', 'Kinyarwanda',
    'Kirghiz', 'Komi', 'Kongo', 'Korean', 'Kuanyama', 'Kurdish', 'Lao', 'Latin',
    'Latvian', 'Limburgan', 'Lingala', 'Lithuanian', 'Luba-Katanga', 'Luxembourgish',
    'Macedonian', 'Malagasy', 'Malay', 'Malayalam', 'Maltese', 'Manx', 'Maori',
    'Marathi', 'Marshallese', 'Mongolian', 'Nauru', 'Navajo', 'North', 'South', 'Ndonga',
    'Nepali', 'Norwegian', 'Norwegian', 'Norwegian', 'Sichuan Yi', 'Occitan', 'Ojibwa', 'Oriya',
    'Oromo', 'Ossetian', 'Pali', 'Pashto', 'Persian', 'Polish', 'Portuguese', 'Punjabi',
    'Quechua', 'Romanian', 'Romansh', 'Rundi', 'Russian', 'Northern Sami', 'Samoan', 'Sango',
    'Sanskrit', 'Sardinian', 'Serbian', 'Shona', 'Sindhi', 'Sinhala', 'Slovak', 'Slovenian',
    'Somali', 'Southern Sotho', 'Spanish', 'Sundanese', 'Swahili', 'Swati', 'Swedish', 'Tagalog',
    'Tahitian', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Tigrinya', 'Tonga',
    'Tsonga', 'Tswana', 'Turkish', 'Turkmen', 'Twi', 'Uighur', 'Ukrainian', 'Urdu', 'Uzbek',
    'Venda', 'Vietnamese', 'Volapük', 'Walloon', 'Welsh', 'Wolof', 'Xhosa', 'Yiddish', 'Yoruba',
    'Zhuang', 'Zulu'
]
ISO = [
    'ab', 'aa', 'af', 'ak', 'sq', 'am', 'ar', 'an', 'hy', 'as', 'av', 'ae', 'ay',
    'az', 'bm', 'ba', 'eu', 'be', 'bn', 'bi', 'bs', 'br', 'bg', 'my', 'ca', 'ch',
    'ce', 'ny', 'zh', 'cu', 'cv', 'kw', 'co', 'cr', 'hr', 'cs', 'da', 'dv', 'nl',
    'dz', 'en', 'eo', 'et', 'ee', 'fo', 'fj', 'fi', 'fr', 'fy', 'ff', 'gd', 'gl',
    'lg', 'ka', 'de', 'el', 'kl', 'gn', 'gu', 'ht', 'ha', 'he', 'hz', 'hi', 'ho',
    'hu', 'is', 'io', 'ig', 'id', 'ia', 'ie', 'iu', 'ik', 'ga', 'it', 'ja', 'jv',
    'kn', 'kr', 'ks', 'kk', 'km', 'ki', 'rw', 'ky', 'kv', 'kg', 'ko', 'kj', 'ku',
    'lo', 'la', 'lv', 'li', 'ln', 'lt', 'lu', 'lb', 'mk', 'mg', 'ms', 'ml', 'mt',
    'gv', 'mi', 'mr', 'mh', 'mn', 'na', 'nv', 'nd', 'nr', 'ng', 'ne', 'no', 'nb',
    'nn', 'ii', 'oc', 'oj', 'or', 'om', 'os', 'pi', 'ps', 'fa', 'pl', 'pt', 'pa',
    'qu', 'ro', 'rm', 'rn', 'ru', 'se', 'sm', 'sg', 'sa', 'sc', 'sr', 'sn', 'sd',
    'si', 'sk', 'sl', 'so', 'st', 'es', 'su', 'sw', 'ss', 'sv', 'tl', 'ty', 'tg',
    'ta', 'tt', 'te', 'th', 'bo', 'ti', 'to', 'ts', 'tn', 'tr', 'tk', 'tw', 'ug',
    'uk', 'ur', 'uz', 've', 'vi', 'vo', 'wa', 'cy', 'wo', 'xh', 'yi', 'yo', 'za',
    'zu '
]
ISO2LANG = dict(zip(ISO, LANGUAGES))
