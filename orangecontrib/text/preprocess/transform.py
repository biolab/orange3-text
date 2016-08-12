import re
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import strip_accents_unicode


__all__ = ['BaseTransformer', 'HtmlTransformer', 'LowercaseTransformer',
           'StripAccentsTransformer', 'UrlRemover']


class BaseTransformer:
    name = NotImplemented

    def __call__(self, data):
        """ Transforms strings in `data`.

        Arguments:
            data (str or iterable): Items to transform

        Returns:
            str or list: Transformed items

        """
        if isinstance(data, str):
            return self.transform(data)
        return [self.transform(string) for string in data]

    @classmethod
    def transform(cls, string):
        """ Transforms `string`. """
        raise NotImplementedError("Method 'transform' isn't implemented "
                                  "in '{cls}' class".format(cls=cls.__name__))

    def __str__(self):
        return self.name


class LowercaseTransformer(BaseTransformer):
    """ Converts all characters to lowercase. """
    name = 'Lowercase'

    @classmethod
    def transform(cls, string):
        return string.lower()


class StripAccentsTransformer(BaseTransformer):
    """ Removes accents. """
    name = "Remove accents"

    @classmethod
    def transform(cls, string):
        return strip_accents_unicode(string)


class HtmlTransformer(BaseTransformer):
    """ Removes all html tags from string. """
    name = "Parse html"

    @classmethod
    def transform(cls, string):
        return BeautifulSoup(string, 'html.parser').getText()


class UrlRemover(BaseTransformer):
    """ Removes hyperlinks. """
    name = "Remove urls"
    urlfinder = re.compile(r"((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)")

    @classmethod
    def transform(cls, string):
        return cls.urlfinder.sub('', string)
