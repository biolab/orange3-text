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
from orangecontrib.text.language_codes import code2lang


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


tv = TimeVariable("Date")
METAS = [
    (StringVariable("Content"), lambda doc, _, __: doc.text),
    (
        DiscreteVariable("Author"),
        lambda doc, users, _: "@" + users[doc.author_id].username,
    ),
    (tv, lambda doc, _, __: tv.parse(doc.created_at.isoformat())),
    (DiscreteVariable("Language"), lambda doc, _, __: doc.lang),
    (DiscreteVariable("Location"), country_code),
    (
        ContinuousVariable("Number of Likes", number_of_decimals=0),
        lambda doc, _, __: doc.public_metrics["like_count"],
    ),
    (
        ContinuousVariable("Number of Retweets", number_of_decimals=0),
        lambda doc, _, __: doc.public_metrics["retweet_count"],
    ),
    (
        DiscreteVariable("In Reply To"),
        lambda doc, users, _: "@" + users[doc.in_reply_to_user_id].username
        if doc.in_reply_to_user_id and doc.in_reply_to_user_id in users
        else "",
    ),
    (DiscreteVariable("Author Name"), lambda doc, users, __: users[doc.author_id].name),
    (
        StringVariable("Author Description"),
        lambda doc, users, _: users[doc.author_id].description,
    ),
    (
        ContinuousVariable("Author Tweets Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["tweet_count"],
    ),
    (
        ContinuousVariable("Author Following Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["following_count"],
    ),
    (
        ContinuousVariable("Author Followers Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["followers_count"],
    ),
    (
        ContinuousVariable("Author Listed Count", number_of_decimals=0),
        lambda doc, users, _: users[doc.author_id].public_metrics["listed_count"],
    ),
    (
        DiscreteVariable("Author Verified"),
        lambda doc, users, _: str(users[doc.author_id].verified),
    ),
    (ContinuousVariable("Longitude"), partial(coordinates, dim=0)),
    (ContinuousVariable("Latitude"), partial(coordinates, dim=1)),
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

    text_features = [METAS[0][0]]  # Content
    string_attributes = [m for m, _ in METAS if isinstance(m, StringVariable)]

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
        return self._create_corpus()

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
            for i, response in enumerate(paginator):
                users = {u.id: u for u in response.includes.get("users", [])}
                places = {p.id: p for p in response.includes.get("places", [])}
                for j, tweet in enumerate(response.data or [], start=1):
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

    def _create_corpus(self) -> Optional[Corpus]:
        if len(self.tweets) == 0:
            return None

        def to_val(attr, val):
            if isinstance(attr, DiscreteVariable):
                attr.val_from_str_add(val)
            return attr.to_val(val)

        m = [attr for attr, _ in METAS]
        domain = Domain(attributes=[], class_vars=[], metas=m)

        metas = np.array(
            [
                [to_val(attr, t) for (attr, _), t in zip(METAS, ts)]
                for ts in self.tweets.values()
            ],
            dtype=object,
        )
        x = np.empty((len(metas), 0))

        return Corpus.from_numpy(domain, x, metas=metas, text_features=self.text_features)

    def append_history(
        self,
        mode: str,
        query: List[str],
        lang: Optional[str],
        allow_retweets: Optional[bool],
        n_tweets: int,
    ):
        lang = code2lang.get(lang, lang)
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
