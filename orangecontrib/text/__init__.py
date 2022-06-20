# Set where NLTK data is downloaded
import os

# temporary solution - remove when Orange 3.33 is released
# it must be imported before nltk_data_dir
from typing import Optional, Dict
from Orange.misc.utils import embedder_utils


def _get_proxies() -> Optional[Dict[str, str]]:
    """
    Return dict with proxy addresses if they exist.
    Returns
    -------
    proxy_dict
        Dictionary with format {proxy type: proxy address} or None if
        they not set.
    """
    def add_scheme(url: Optional[str]) -> Optional[str]:
        if url is not None and "://" not in url:
            # if no scheme default to http - as other libraries do (e.g. requests)
            return f"http://{url}"
        else:
            return url

    http_proxy = add_scheme(os.environ.get("http_proxy"))
    https_proxy = add_scheme(os.environ.get("https_proxy"))
    proxy_dict = {}
    if http_proxy:
        proxy_dict["http://"] = http_proxy
    if https_proxy:
        proxy_dict["https://"] = https_proxy
    return proxy_dict if proxy_dict else None


embedder_utils.get_proxies = _get_proxies
# remove to here


from orangecontrib.text.misc import nltk_data_dir
os.environ['NLTK_DATA'] = nltk_data_dir()

from .corpus import Corpus

from .version import git_revision as __git_revision__
from .version import version as __version__
