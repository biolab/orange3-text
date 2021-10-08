from collections import OrderedDict, Iterable

import tweepy

from Orange.data import (
    StringVariable,
    ContinuousVariable,
    DiscreteVariable,
    TimeVariable,
)
from orangecontrib.text import Corpus
from orangecontrib.text.language_codes import code2lang

__all__ = ["Credentials", "TwitterAPI"]


def coordinates_geoJSON(json):
    if json:
        return json.get("coordinates", [None, None])
    return [None, None]


class Credentials:
    """ Twitter API credentials. """

    def __init__(self, consumer_key, consumer_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self._valid = None

    @property
    def valid(self):
        if self._valid is None:
            self.check()
        return self._valid

    def check(self):
        try:
            self.auth.get_authorization_url()
            self._valid = True
        except tweepy.TweepyException:
            self._valid = False
        return self._valid

    def __getstate__(self):
        odict = self.__dict__.copy()
        odict["_valid"] = None
        odict.pop("auth")
        return odict

    def __setstate__(self, odict):
        self.__dict__.update(odict)
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)

    def __eq__(self, other):
        return (
            isinstance(other, Credentials)
            and self.consumer_key == other.consumer_key
            and self.consumer_secret == other.consumer_secret
        )


class TwitterAPI:
    """ Fetch tweets from the Tweeter API.

    Notes:
        Results across multiple searches are aggregated. To remove tweets form
        previous searches and only return results from the last search either
        call `reset` method before searching or provide `collecting=False`
        argument to search method.
    """

    attributes = []
    class_vars = []

    tv = TimeVariable("Date")
    authors = [
        (DiscreteVariable("Author"), lambda doc: "@" + doc.author.screen_name,),
    ]
    metas = [
        (
            StringVariable("Content"),
            lambda doc: doc.full_text if not doc.retweeted else doc.text,
        ),
        # temporary fix until Orange>3.30.1 then change back to
        # (tv, lambda doc: TwitterAPI.tv.parse(doc.created_at.isoformat())),
        (tv, lambda doc: TwitterAPI.tv.parse(
                    TwitterAPI.tv._tzre_sub(doc.created_at.isoformat()))),
        (DiscreteVariable("Language"), lambda doc: doc.lang),
        (
            DiscreteVariable("Location"),
            lambda doc: getattr(doc.place, "country_code", None),
        ),
        (
            ContinuousVariable("Number of Likes", number_of_decimals=0),
            lambda doc: doc.favorite_count,
        ),
        (
            ContinuousVariable("Number of Retweets", number_of_decimals=0),
            lambda doc: doc.retweet_count,
        ),
        (
            DiscreteVariable("In Reply To"),
            lambda doc: "@" + doc.in_reply_to_screen_name
            if doc.in_reply_to_screen_name
            else "",
        ),
        (DiscreteVariable("Author Name"), lambda doc: doc.author.name),
        (
            StringVariable("Author Description"),
            lambda doc: doc.author.description,
        ),
        (
            ContinuousVariable("Author Statuses Count", number_of_decimals=0),
            lambda doc: doc.author.statuses_count,
        ),
        (
            ContinuousVariable("Author Favourites Count", number_of_decimals=0),
            lambda doc: doc.author.favourites_count,
        ),
        (
            ContinuousVariable("Author Friends Count", number_of_decimals=0),
            lambda doc: doc.author.friends_count,
        ),
        (
            ContinuousVariable("Author Followers Count", number_of_decimals=0),
            lambda doc: doc.author.followers_count,
        ),
        (
            ContinuousVariable("Author Listed Count", number_of_decimals=0),
            lambda doc: doc.author.listed_count,
        ),
        (
            DiscreteVariable("Author Verified"),
            lambda doc: str(doc.author.verified),
        ),
        (
            ContinuousVariable("Longitude"),
            lambda doc: coordinates_geoJSON(doc.coordinates)[0],
        ),
        (
            ContinuousVariable("Latitude"),
            lambda doc: coordinates_geoJSON(doc.coordinates)[1],
        ),
    ]

    text_features = [metas[0][0]]  # Content
    string_attributes = [m for m, _ in metas if isinstance(m, StringVariable)]

    def __init__(self, credentials):
        self.key = credentials
        self.api = tweepy.API(credentials.auth)
        self.container = OrderedDict()
        self.search_history = []

    @property
    def tweets(self):
        return self.container.values()

    def search_content(
        self,
        content,
        *,
        max_tweets=0,
        lang=None,
        allow_retweets=True,
        collecting=False,
        callback=None
    ):
        """ Search by content.

        Args:
            content (list of str): A list of key words to search for.
            max_tweets (int): If greater than zero limits the number of
                downloaded tweets.
            lang (str): A language's code (either ISO 639-1 or ISO 639-3
                formats).
            allow_retweets(bool): Whether to download retweets.
            collecting (bool): Whether to collect results across multiple
                search calls.

        Returns:
            Corpus
        """
        if not collecting:
            self.reset()

        if max_tweets == 0:
            max_tweets = float("Inf")

        def build_query():
            nonlocal content
            if not content:
                q = "from: "
            else:
                if not isinstance(content, list):
                    content = [content]
                q = " OR ".join(['"{}"'.format(q) for q in content])
            if not allow_retweets:
                q += " -filter:retweets"
            return q

        query = build_query()
        cursor = tweepy.Cursor(
            self.api.search_tweets, q=query, lang=lang, tweet_mode="extended"
        )

        corpus, count = self.fetch(
            cursor, max_tweets, search_author=False, callback=callback
        )
        self.append_history(
            "Content",
            content,
            lang if lang else "Any",
            str(allow_retweets),
            count,
        )
        return corpus

    def search_authors(
        self, authors, *, max_tweets=0, collecting=False, callback=None
    ):
        """ Search by authors.

        Args:
            authors (list of str): A list of authors to search for.
            max_tweets (int): If greater than zero limits the number of
                downloaded tweets.
            collecting (bool): Whether to collect results across multiple
                search calls.

        Returns:
            Corpus
        """
        if not collecting:
            self.reset()

        if max_tweets == 0:  # set to max allowed for progress
            max_tweets = 3200

        if not isinstance(authors, list):
            authors = [authors]

        cursors = [
            tweepy.Cursor(
                self.api.user_timeline, screen_name=a, tweet_mode="extended"
            )
            for a in authors
        ]
        corpus, count = self.fetch(
            cursors, max_tweets, search_author=True, callback=callback
        )
        self.append_history("Author", authors, None, None, count)
        return corpus

    def fetch(self, cursors, max_tweets, search_author, callback):
        if not isinstance(cursors, list):
            cursors = [cursors]

        count = 0
        for i, cursor in enumerate(cursors):
            for j, tweet in enumerate(cursor.items(max_tweets), start=1):
                if tweet.id not in self.container:
                    count += 1
                self.container[tweet.id] = tweet
                if j % 20 == 0:
                    if callback is not None:
                        callback(
                            (i * max_tweets + j) / (len(cursors) * max_tweets)
                        )

        return self.create_corpus(search_author), count

    def create_corpus(self, search_author):
        if search_author:
            class_vars = self.authors
            metas = self.metas
        else:
            class_vars = []
            metas = self.metas + self.authors
        return Corpus.from_documents(
            self.tweets,
            "Twitter",
            self.attributes,
            class_vars,
            metas,
            title_indices=[-1],
        )

    def reset(self):
        """ Removes all downloaded tweets. """
        self.search_history = []
        self.container = OrderedDict()

    def append_history(self, mode, query, lang, allow_retweets, n_tweets):
        query = ", ".join(query) if isinstance(query, Iterable) else query
        if lang in code2lang.keys():
            lang = code2lang[lang]
        self.search_history.append(
            (
                ("Query", query),
                ("Search by", mode),
                ("Language", lang),
                ("Allow retweets", allow_retweets),
                ("Tweets count", n_tweets),
            )
        )

    def report(self):
        return self.search_history
