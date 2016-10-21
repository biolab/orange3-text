from collections import OrderedDict

import tweepy

from Orange import data
from orangecontrib.text import Corpus

__all__ = ['Credentials', 'TwitterAPI']


def coordinates_geoJSON(json):
    if json:
        return json.get('coordinates', [None, None])
    return [None, None]


class Credentials:
    """ Twitter API credentials. """

    def __init__(self, consumer_key, consumer_secret):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        self.auth = tweepy.OAuthHandler(self.consumer_key, self.consumer_secret)
        self._valid = None

    @property
    def valid(self):
        """bool: Indicates whether it's a valid credentials. """
        if self._valid is None:
            self.check()

        return self._valid

    def check(self):
        try:
            self.auth.get_authorization_url()
            self._valid = True
        except tweepy.TweepError:
            self._valid = False

        return self._valid

    def __getstate__(self):
        odict = self.__dict__.copy()
        odict['_valid'] = None
        odict.pop('auth')
        return odict

    def __setstate__(self, odict):
        self.__dict__.update(odict)
        self.auth = tweepy.OAuthHandler(self.consumer_key,
                                        self.consumer_secret)

    def __eq__(self, other):
        return isinstance(other, Credentials) \
               and self.consumer_key == other.consumer_key \
               and self.consumer_secret == other.consumer_secret


class TwitterAPI:
    """ Fetch tweets from the Tweeter API.

    Notes:
        Every search accumulates downloaded tweets. To remove the stored tweets call `reset` method.
    """
    attributes = []
    class_vars = [
        (data.DiscreteVariable('Author'), lambda doc: '@' + doc.author.screen_name),
    ]

    tv = data.TimeVariable('Date')
    metas = [
        (data.StringVariable('Content'), lambda doc: doc.text),
        (tv, lambda doc: TwitterAPI.tv.parse(doc.created_at.isoformat())),
        (data.DiscreteVariable('Language'), lambda doc: doc.lang),
        (data.DiscreteVariable('Location'), lambda doc: getattr(doc.place, 'country_code', None)),
        (data.ContinuousVariable('Number of Likes'), lambda doc: doc.favorite_count),
        (data.ContinuousVariable('Number of Retweets'), lambda doc: doc.retweet_count),
        (data.DiscreteVariable('In Reply To'),
            lambda doc: '@' + doc.in_reply_to_screen_name if doc.in_reply_to_screen_name else ''),
        (data.DiscreteVariable('Author Name'), lambda doc: doc.author.name),
        (data.StringVariable('Author Description'), lambda doc: doc.author.description),
        (data.ContinuousVariable('Author Statuses Count'), lambda doc: doc.author.statuses_count),
        (data.ContinuousVariable('Author Favourites Count'), lambda doc: doc.author.favourites_count),
        (data.ContinuousVariable('Author Friends Count'), lambda doc: doc.author.friends_count),
        (data.ContinuousVariable('Author Followers Count'), lambda doc: doc.author.followers_count),
        (data.ContinuousVariable('Author Listed Count'), lambda doc: doc.author.listed_count),
        (data.DiscreteVariable('Author Verified'), lambda doc: str(doc.author.verified)),
        (data.ContinuousVariable('Longitude'),
         lambda doc: coordinates_geoJSON(doc.coordinates)[0]),
        (data.ContinuousVariable('Latitude'),
         lambda doc: coordinates_geoJSON(doc.coordinates)[1]),
    ]

    text_features = [metas[1][0]]       # Content
    string_attributes = [m for m, _ in metas if isinstance(m, data.StringVariable)]

    def __init__(self, credentials, on_error=None, on_rate_limit=None,):
        self.key = credentials
        self.api = tweepy.API(credentials.auth)
        self.container = OrderedDict()
        self.search_history = []

        # Callbacks:
        self.on_error = on_error
        self.on_rate_limit = on_rate_limit

    @property
    def tweets(self):
        """ Iterator over the downloaded documents. """
        return self.container.values()

    @staticmethod
    def build_query(word_list=None, authors=None, allow_retweets=True):
        if authors is None:
            authors = []

        if word_list is None:
            word_list = []

        if not word_list and not authors:
            # allows empty queries
            query = "from: "
        else:
            query = " OR ".join(['"{}"'.format(q) for q in word_list] +
                                ['from:{}'.format(user) for user in authors])

        if not allow_retweets:
            query += ' -filter:retweets'

        return query

    def search(self, *, content=None, authors=None,
               max_tweets=None, lang=None, allow_retweets=True,
               collecting=False, on_progress=None, should_break=None):
        """ Performs search for tweets.

        Args:
            max_tweets (int): If present limits the number of downloaded tweets.
            content (list of str): A list of key words to search for.
            authors (list of str): A list of tweets' author.
            lang (str): A language's code (either ISO 639-1 or ISO 639-3 formats).
            allow_retweets(bool): Whether to download retweets.
            collecting (bool): Whether to collect results across multiple
                search calls.
            on_progress (callable): Callback for progress reporting.
            should_break (callback): Callback for breaking the computation
                before the end. If it evaluates to True, downloading is stopped
                and document downloaded till now are returned in a Corpus.

        Returns:
            Corpus
        """
        on_progress = on_progress if on_progress else lambda x, y: (x, y)
        should_break = should_break if should_break else lambda: False

        if not collecting:
            self.reset()

        query = self.build_query(word_list=content, authors=authors,
                                 allow_retweets=allow_retweets)

        try:
            for i, tweet in enumerate(tweepy.Cursor(
                    self.api.search, q=query, lang=lang).items(max_tweets),
                                      start=1):
                if should_break():
                    break
                self.container[tweet.id] = tweet
                on_progress(len(self.container), i)
        except tweepy.TweepError as e:
            if e.response.status_code == 429 and self.on_rate_limit:
                self.on_rate_limit()
            elif self.on_error:
                self.on_error(str(e))

        self.append_history(content, lang, allow_retweets, i)
        return self.create_corpus()

    def create_corpus(self):
        """ Create a corpus with collected tweets. """
        return Corpus.from_documents(self.tweets, 'Twitter', self.attributes,
                                     self.class_vars, self.metas,
                                     title_indices=[-1])

    def reset(self):
        """ Removes all downloaded tweets. """
        self.search_history = []
        self.container = OrderedDict()

    def append_history(self, query, lang, allow_retweets, n_tweets):
        self.search_history.append((
            ('Query', ', '.join(query)),
            ('Language', lang if lang else 'Any'),
            ('Allow retweets', str(allow_retweets)),
            ('Tweets count', n_tweets),
        ))

    def report(self):
        return self.search_history
