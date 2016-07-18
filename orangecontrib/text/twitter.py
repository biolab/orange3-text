import threading
from collections import OrderedDict

import numpy as np
import tweepy

from Orange import data
from orangecontrib.text.corpus import Corpus

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
    """ Fetch tweets from the Tweeter API. """

    metas = [
        (data.StringVariable('text'), lambda st: st.text),
        (data.StringVariable('author_description'), lambda st: st.author.description),
        (data.StringVariable('place'), lambda st: getattr(st.place, 'country_code', None)),
    ]
    text_features = [metas[0][0]]

    attributes = [
        (data.DiscreteVariable('id'), lambda st: st.id_str),
        (data.DiscreteVariable('in_reply_to_user_id'), lambda st: st.in_reply_to_user_id),
        (data.ContinuousVariable('favorite_count'), lambda st: st.favorite_count),
        (data.ContinuousVariable('retweet_count'), lambda st: st.retweet_count),
        (data.TimeVariable('created_at'), lambda st: st.created_at.timestamp()),
        (data.DiscreteVariable('lang'), lambda st: st.lang),
        (data.DiscreteVariable('author_id'), lambda st: st.author.id_str),
        (data.DiscreteVariable('author_name'), lambda st: st.author.name),
        (data.DiscreteVariable('author_screen_name'), lambda st: st.author.screen_name),
        (data.ContinuousVariable('author_statuses_count'), lambda st: st.author.statuses_count),
        (data.ContinuousVariable('author_favourites_count'), lambda st: st.author.favourites_count),
        (data.ContinuousVariable('author_friends_count'), lambda st: st.author.friends_count),
        (data.ContinuousVariable('author_followers_count'), lambda st: st.author.followers_count),
        (data.ContinuousVariable('author_listed_count'), lambda st: st.author.listed_count),
        (data.DiscreteVariable('author_verified'), lambda st: str(st.author.verified)),
        (data.ContinuousVariable('coordinates_longitude'), lambda st: coordinates_geoJSON(st.coordinates)[0]),
        (data.ContinuousVariable('coordinates_latitude'), lambda st: coordinates_geoJSON(st.coordinates)[1]),
    ]

    supported_fields = metas + attributes

    def __init__(self, credentials, on_start=None, on_progress=None, on_error=None, on_finish=None):
        self.busy = False
        self.key = credentials
        self.api = tweepy.API(credentials.auth)
        self.statuses_lock = threading.Lock()
        self.task = None

        # Callbacks:
        self.on_progress = on_progress
        self.on_error = on_error
        self.on_finish = on_finish
        self.on_start = on_start

        self.container = OrderedDict()
        self.history = []

    @property
    def tweets(self):
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
        """ Search for tweets with the given criteria. """
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
        """ Indicates whether there is an active task. """
        return self.task is not None and self.task.running

    def join(self, *args):
        if self.task:
            self.task.join(*args)

    def add_status(self, status):
        status_record = {attr.name: getter(status)
                         for attr, getter in self.supported_fields}
        self.statuses_lock.acquire()
        self.container[status.id] = status_record
        self.statuses_lock.release()

    def create_corpus(self, included_attributes=None):
        """ Creates a corpus with collected tweets. """
        if included_attributes:
            attributes = [(attr, _) for attr, _ in self.attributes
                          if attr.name in included_attributes]

            metas = [(attr, _) for attr, _ in self.metas
                     if attr.name in included_attributes]

            text_features = [attr for attr in self.text_features
                             if attr.name in included_attributes]
        else:
            attributes = self.attributes
            metas = self.metas
            text_features = self.text_features

        domain = data.Domain(attributes=[attr for attr, _ in attributes],
                             metas=[attr for attr, _ in metas])

        self.statuses_lock.acquire()

        for attr in domain.attributes:
            if isinstance(attr, data.DiscreteVariable):
                attr.values = []

        def to_val(attr, val):
            if isinstance(attr, data.DiscreteVariable) and val not in attr.values:
                attr.add_value(val)
            return attr.to_val(val)

        X = np.array([
            [to_val(attr, record[attr.name]) for attr, _ in attributes]
            for record in self.tweets
        ])

        metas = np.array([
            [record[attr.name] for attr, _ in metas]
            for record in self.tweets
        ], dtype=object)
        self.statuses_lock.release()

        return Corpus(X=X, metas=metas, domain=domain, text_features=text_features)

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
            if self.master.on_error:
                self.master.on_error(str(e))

        self.finish()

    def finish(self):
        self.running = False
        if self.master.on_finish:
            self.master.on_finish()

    def report(self):
        return (('Query', self.q),
                ('Language', self.lang or 'any'),
                ('Tweets count', self.progress))
