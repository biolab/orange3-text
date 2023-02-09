import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import call, patch

from wikipedia import DisambiguationError

from orangecontrib.text.corpus import Corpus
from orangecontrib.text.wikipedia_api import WikipediaAPI, wikipedia


class StoppingMock(mock.Mock):
    def __init__(self, allow_calls=0):
        super().__init__()
        self.allow_calls = allow_calls
        self.call_count = 0

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count > self.allow_calls:
            return True
        else:
            return False


def create_page(title):
    return SimpleNamespace(
        title=title,
        content=title + "text",
        summary=title,
        url=title,
        pageid=int(title[-1]),
        revision_id=1,
    )


ARTICLES = [f"Article {i}" for i in range(1, 11)]
ARTICLE_TEXTS = [create_page(title) for title in ARTICLES]


class WikipediaTests(unittest.TestCase):
    @patch(
        "orangecontrib.text.wikipedia_api.wikipedia.search", return_value=ARTICLES[:2]
    )
    @patch("orangecontrib.text.wikipedia_api.wikipedia.page", side_effect=ARTICLE_TEXTS)
    def test_search(self, _, search_mock):
        on_progress = mock.MagicMock()

        api = WikipediaAPI()

        result = api.search(
            "en", ["Clinton"], articles_per_query=2, on_progress=on_progress
        )
        search_mock.assert_called_with("Clinton", results=2)
        self.assertIsInstance(result, Corpus)
        self.assertEqual(len(result.domain.attributes), 0)
        self.assertEqual(len(result.domain.metas), 7)
        self.assertEqual(len(result), 2)
        self.assertListEqual(
            [["Article 1"], ["Article 2"]], result[:, "Title"].metas.tolist()
        )

        self.assertEqual(on_progress.call_count, 2)
        progress = 0
        for arg in on_progress.call_args_list:
            self.assertGreater(arg[0][0], progress)
            progress = arg[0][0]

    @patch(
        "orangecontrib.text.wikipedia_api.wikipedia.search",
        side_effect=[ARTICLES[:3], [ARTICLES[4]]],
    )
    @patch(
        "orangecontrib.text.wikipedia_api.wikipedia.page",
        side_effect=[DisambiguationError("Article 0", "1")] + ARTICLE_TEXTS,
    )
    def test_search_disambiguation(self, _, search_mock):
        api = WikipediaAPI()
        result = api.search("en", ["Scarf"], articles_per_query=3)
        search_mock.assert_has_calls((call("Scarf", results=3), call("Article 1", 10)))

        self.assertIsInstance(result, Corpus)
        self.assertGreaterEqual(len(result), 3)

    @patch(
        "orangecontrib.text.wikipedia_api.wikipedia.search",
        side_effect=[
            ARTICLES[:2],  # clinton search
            ARTICLES[2:5],  # first scarf
            ARTICLES[2:3],  # firs scarf after DisambiguationError
            ARTICLES[5:8],  # second scarf
            ARTICLES[5:6],  # second scarf after DisambiguationError
        ],
    )
    @patch(
        "orangecontrib.text.wikipedia_api.wikipedia.page",
        side_effect=[DisambiguationError("Article 2", "1")]  # firs scarf
        + ARTICLE_TEXTS[2:5]  # firs scarf - successes
        + [DisambiguationError("Article 2", "1")],  # second scarf
    )
    def test_search_break(self, _, search_mock):
        api = WikipediaAPI()

        # stop immediately
        result = api.search('en', ['Clinton'], articles_per_query=2,
                            should_break=mock.Mock(return_value=True))
        self.assertEqual(len(result), 0)

        # stop inside recursion
        r = api.search("en", ["Scarf"], articles_per_query=3)
        self.assertListEqual(
            [["Article 3"], ["Article 4"], ["Article 5"]], r[:, "Title"].metas.tolist()
        )
        result = api.search(
            "en",
            ["Scarf"],
            articles_per_query=3,
            should_break=StoppingMock(allow_calls=1),
        )
        self.assertEqual(len(result), 0)

    def page(*args, **kwargs):
        raise wikipedia.exceptions.PageError('1')

    @mock.patch('wikipedia.page', page)
    @patch(
        "orangecontrib.text.wikipedia_api.wikipedia.search",
        return_values=ARTICLES[:3],
    )
    def test_page_error(self, _):
        on_error = mock.MagicMock()
        api = WikipediaAPI(on_error=on_error)
        api.search('en', ['Barack Obama'])
        self.assertEqual(on_error.call_count, 0)

    def search(*args, **kwargs):
        raise IOError('Network error')

    @mock.patch('wikipedia.search', search)
    def test_network_errors(self):
        on_error = mock.MagicMock()
        api = WikipediaAPI(on_error=on_error)
        api.search('en', ['Barack Obama'])
        self.assertEqual(on_error.call_count, 1)


if __name__ == "__main__":
    unittest.main()
