""" A module for fetching data from `The Twitter Search API <https://dev.twitter.com/rest/public/search>`_. """
import threading
from collections import OrderedDict

import tweepy

from Orange import data
from orangecontrib.text import Corpus
from orangecontrib.text.language_codes import code2lang

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
    class_vars = []

    metas = [
        (data.DiscreteVariable('Author'), lambda doc: '@'+doc.author.screen_name),
        (data.StringVariable('Content'), lambda doc: doc.text),
        (data.TimeVariable('Date'), lambda doc: doc.created_at.timestamp()),
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

    def __init__(self, credentials, on_start=None, on_progress=None, on_error=None,
                 on_rate_limit=None, on_finish=None):
        self.key = credentials
        self.api = tweepy.API(credentials.auth)
        self.statuses_lock = threading.Lock()
        self.task = None

        # Callbacks:
        self.on_progress = on_progress
        self.on_error = on_error
        self.on_finish = on_finish
        self.on_rate_limit = on_rate_limit
        self.on_start = on_start

        self.container = OrderedDict()
        self.history = []

    @property
    def tweets(self):
        """ Iterator over the downloaded documents. """
        return self.container.values()

    @staticmethod
    def build_query(word_list=None, authors=None, since=None, until=None, allow_retweets=True):
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

        if since:
            query += ' since:' + since.strftime('%Y-%m-%d')
        if until:
            query += ' until:' + until.strftime('%Y-%m-%d')

        if not allow_retweets:
            query += ' -filter:retweets'

        return query

    def search(self, *, word_list=None, authors=None, max_tweets=None, lang=None,
               since=None, until=None, allow_retweets=True):
        """ Performs search for tweets.

        All the parameters optional.

        Args:
            max_tweets (int): If present limits the number of downloaded tweets.
            word_list (list of str): A list of key words to search for.
            authors (list of str): A list of tweets' author.
            lang (str): A language's code (either ISO 639-1 or ISO 639-3 formats).
            since (str): Fetch tweets only from this date.
            until (str): Fetch tweets only to this date.
            allow_retweets(bool): Whether to download retweets.
        """
        query = self.build_query(word_list=word_list, authors=authors,
                                 since=since, until=until, allow_retweets=allow_retweets)

        self.task = SearchTask(self, q=query, lang=lang, max_tweets=max_tweets)
        self.history.append(self.task)
        self.task.start()

    def disconnect(self):
        if self.task:
            self.task.disconnect()

    @property
    def running(self):
        """bool: Indicates whether there is an active task. """
        return self.task is not None and self.task.running

    def join(self, *args):
        if self.task:
            self.task.join(*args)

    def add_status(self, status):
        self.statuses_lock.acquire()
        self.container[status.id] = status
        self.statuses_lock.release()

    def create_corpus(self):
        """ Creates a corpus with collected tweets. """
        self.statuses_lock.acquire()
        corpus = Corpus.from_documents(self.tweets, 'Twitter', self.attributes,
                                       self.class_vars, self.metas, title_indices=[-2])
        self.statuses_lock.release()
        return corpus

    def reset(self):
        """ Removes all downloaded tweets. """
        if self.task:
            self.task.disconnect()
            self.task.join()
        self.history = []
        self.container = OrderedDict()


class SearchTask(threading.Thread):
    def __init__(self, master, q, lang=None, max_tweets=None, **kwargs):
        super().__init__()
        self.master = master
        self.q = q
        self.lang = lang
        self.running = False
        self.max_tweets = max_tweets
        self.kwargs = kwargs

    def disconnect(self):
        self.running = False

    def start(self):
        self.running = True
        self.progress = 0
        if self.master.on_start:
            self.master.on_start()
        super().start()

    def run(self):
        try:
            for status in tweepy.Cursor(self.master.api.search, q=self.q,
                                        lang=self.lang, **self.kwargs).items(self.max_tweets):
                self.master.add_status(status)
                self.progress += 1
                if self.master.on_progress:
                    self.master.on_progress(self.progress)

                if not self.running:
                    break
        except tweepy.TweepError as e:
            if e.response.status_code == 429 and self.master.on_rate_limit:
                self.master.on_rate_limit()
            elif self.master.on_error:
                self.master.on_error(str(e))

        self.finish()

    def finish(self):
        self.running = False
        if self.master.on_finish:
            self.master.on_finish()

    def report(self):
        return (('Query', self.q),
                ('Language', code2lang.get(self.lang, 'Any')),
                ('Tweets count', self.progress))
