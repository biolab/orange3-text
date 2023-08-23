import logging
from functools import partial
from typing import List, Optional, Callable

import numpy as np
import tweepy
from Orange.data import (
    ContinuousVariable,
    DiscreteVariable,
    Domain,
    StringVariable,
    TimeVariable,
)
from Orange.util import dummy_callback, wrap_callback
from tweepy import TooManyRequests

from orangecontrib.text import Corpus
from orangecontrib.text.language import ISO2LANG, infer_language_from_variable

log = logging.getLogger(__name__)

# fmt: off
SUPPORTED_LANGUAGES = [
    "am", "ar", "bg", "bn", "bo", "ca", "ckb", "cs", "cy", "da", "de", "dv",
    "el", "en", "es", "et", "eu", "fa", "fi", "fr", "gu", "he", "hi", "hi-Latn",
    "ht", "hu", "hy", "id", "is", "it", "ja", "ka", "km", "kn", "ko", "lo",
    "lt", "lv", "ml", "mr", "my", "ne", "nl", "no", "or", "pa", "pl", "ps",
    "pt", "ro", "ru", "sd", "si", "sl", "sr", "sv", "ta", "te", "th", "tl",
    "tr", "ug", "uk", "ur", "vi", "zh"
]
# fmt: on


class NoAuthorError(ValueError):
    pass


def coordinates(tweet, _, __, dim):
    coord = tweet.geo.get("coordinates", None) if tweet.geo else None
    return coord["coordinates"][dim] if coord else None


def country_code(tweet, _, places):
    place_id = tweet.geo.get("place_id", None) if tweet.geo else None
    return places[place_id].country_code if place_id else ""


METAS = [
    (partial(StringVariable, "Content"), lambda doc, _, __: doc.text),
    (
        partial(DiscreteVariable, "Author"),
        lambda doc, users, _: "@" + users[doc.author_id].username,
    ),
    # Twitter API return values in UTC, since Date variable is created later we
    # don't use TimeVariable.parse but transform to UNIX timestamp manually
    (partial(TimeVariable, "Date"), lambda doc, _, __: doc.created_at.timestamp()),
    (partial(DiscreteVariable, "Language"), lambda doc, _, __: doc.lang),
    (partial(DiscreteVariable, "Location"), country_code),
    (
        partial(ContinuousVariable, "Number of Likes", number_of_decimals=0),
        lambda doc, _, __: doc.public_metrics["like_count"],
    ),
    (
        partial(ContinuousVariable, "Number of Retweets", number_of_decimals=0),
        lambda doc, _, __: doc.public_metrics["retweet_count"],
    ),
    (
        partial(DiscreteVariable, "In Reply To"),
        lambda doc, users, _: "@" + users[doc.in_reply_to_user_id].username
        if doc.in_reply_to_user_id and doc.in_reply_to_user_id in users
        else "",
    ),
    (
        partial(DiscreteVariable, "Author Name"),
        lambda doc, users, __: users[doc.author_id].name,
    ),
    (
        partial(StringVariable, "Author Description"),
        lambda doc, users, _: users[doc.author_id].description,
    ),
    (
        partial(ContinuousVariable, "Author Tweets Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["tweet_count"],
    ),
    (
        partial(ContinuousVariable, "Author Following Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["following_count"],
    ),
    (
        partial(ContinuousVariable, "Author Followers Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["followers_count"],
    ),
    (
        partial(ContinuousVariable, "Author Listed Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["listed_count"],
    ),
    (
        partial(DiscreteVariable, "Author Verified"),
        lambda doc, users, _: str(users[doc.author_id].verified),
    ),
    (partial(ContinuousVariable, "Longitude"), partial(coordinates, dim=0)),
    (partial(ContinuousVariable, "Latitude"), partial(coordinates, dim=1)),
]
# maximum number of tweets that can be downloaded in one set of requests
# max 450requests/15min, request can contain max 100 tweets
MAX_TWEETS = 450 * 100


request_settings = {
    "tweet_fields": [
        "lang",
        "public_metrics",
        "in_reply_to_user_id",
        "author_id",
        "geo",
        "created_at",
    ],
    "user_fields": ["description", "public_metrics", "verified"],
    "place_fields": ["country_code"],
    "expansions": ["author_id", "in_reply_to_user_id", "geo.place_id"],
    "max_results": 100,
}


class TwitterAPI:
    """Fetch tweets from the Tweeter API.

    Notes:
        Results across multiple searches are aggregated. To remove tweets form
        previous searches and only return results from the last search either
        call `reset` method before searching or provide `collecting=False`
        argument to search method.
    """
    def __init__(self, bearer_token):
        self.api = tweepy.Client(bearer_token)
        self.tweets = {}
        self.search_history = []

    def search_content(
        self,
        content: List[str],
        *,
        max_tweets: Optional[int] = MAX_TWEETS,
        lang: Optional[str] = None,
        allow_retweets: bool = True,
        collecting: bool = False,
        callback: Callable = dummy_callback,
    ) -> Optional[Corpus]:
        """
        Search recent tweets by content (keywords).

        Parameters
        ----------
        content
            A list of key-words to search for.
        max_tweets
            Limits the number of downloaded tweets. If none use APIs maximum.
        lang
            A language's code (either ISO 639-1 or ISO 639-3 formats).
        allow_retweets
            Whether to download retweets.
        collecting
            Whether to collect results across multiple search calls.
        callback
            Function to report the progress

        Returns
        -------
        Corpus with tweets
        """
        if not collecting:
            self.reset()
        max_tweets = max_tweets or MAX_TWEETS

        def build_query():
            assert len(content) > 0, "At leas one keyword required"
            q = " OR ".join(['"{}"'.format(q) for q in content])
            if not allow_retweets:
                q += " -is:retweet"
            if lang:
                q += f" lang:{lang}"
            return q

        paginator = tweepy.Paginator(
            self.api.search_recent_tweets, build_query(), **request_settings
        )
        count = self._fetch(paginator, max_tweets, callback=callback)
        self.append_history("Content", content, lang or "Any", allow_retweets, count)
        return self._create_corpus(lang)

    def search_authors(
        self,
        authors: List[str],
        *,
        max_tweets: Optional[int] = MAX_TWEETS,
        collecting: bool = False,
        callback: Callable = dummy_callback,
    ) -> Optional[Corpus]:
        """
        Search recent tweets by authors.

        Parameters
        ----------
        authors
            A list of authors to search for.
        max_tweets
            Limits the number of downloaded tweets. If none use APIs maximum.
        collecting
            Whether to collect results across multiple search calls.
        callback
            Function to report the progress

        Returns
        -------
        Corpus with tweets
        """
        if not collecting:
            self.reset()

        count_sum = 0
        n = len(authors)
        for i, author in enumerate(authors):
            author_ = self.api.get_user(username=author)
            if author_.data is None:
                raise NoAuthorError(author)
            paginator = tweepy.Paginator(
                self.api.get_users_tweets, author_.data.id, **request_settings
            )
            count_sum += self._fetch(
                paginator,
                max_tweets,
                callback=wrap_callback(callback, i / n, (i + 1) / n),
            )
        self.append_history("Author", authors, None, None, count_sum)
        return self._create_corpus()

    def _fetch(
        self, paginator: tweepy.Paginator, max_tweets: int, callback: Callable
    ) -> int:
        count = 0
        try:
            done = False
            for response in paginator:
                users = {u.id: u for u in response.includes.get("users", [])}
                places = {p.id: p for p in response.includes.get("places", [])}
                for tweet in response.data or []:
                    if tweet.id not in self.tweets:
                        count += 1
                    self.tweets[tweet.id] = [f(tweet, users, places) for _, f in METAS]
                    callback(count / max_tweets)
                    if count >= max_tweets:
                        done = True
                        break
                if done:
                    break
        except TooManyRequests:
            log.debug("TooManyRequests raised")
        return count

    def _create_corpus(self, language: Optional[str] = None) -> Optional[Corpus]:
        if len(self.tweets) == 0:
            return None

        def to_val(attr, val):
            if isinstance(attr, DiscreteVariable):
                attr.val_from_str_add(val)
            return attr.to_val(val)

        m = [attr() for attr, _ in METAS]
        domain = Domain(attributes=[], class_vars=[], metas=m)

        metas = np.array(
            [
                [to_val(attr, t) for attr, t in zip(m, ts)]
                for ts in self.tweets.values()
            ],
            dtype=object,
        )
        x = np.empty((len(metas), 0))

        language_var = domain["Language"]
        assert isinstance(language_var, DiscreteVariable)
        language = language or infer_language_from_variable(language_var)
        return Corpus.from_numpy(
            domain, x, metas=metas, text_features=[domain["Content"]], language=language
        )

    def append_history(
        self,
        mode: str,
        query: List[str],
        lang: Optional[str],
        allow_retweets: Optional[bool],
        n_tweets: int,
    ):
        lang = ISO2LANG.get(lang, lang)
        self.search_history.append(
            (
                ("Query", query),
                ("Search by", mode),
                ("Language", lang),
                ("Allow retweets", str(allow_retweets)),
                ("Tweets count", n_tweets),
            )
        )

    def reset(self):
        """Removes all downloaded tweets."""
        self.tweets = {}
        self.search_history = []
