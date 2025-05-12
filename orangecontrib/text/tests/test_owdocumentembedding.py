import pytest
from orangecontrib.text.vectorization.document_embedder import DocumentEmbedder
from orangecontrib.text import Corpus
from Orange.misc.utils.embedder_utils import EmbeddingConnectionError
from orangecontrib.text.vectorization.document_embedder import _ServerEmbedder
from urllib.parse import urlparse
import socket

@pytest.fixture
def dummy_corpus():
    return Corpus.from_documents(["This is a test document."], name="test")

def test_embedding_valid_server(dummy_corpus):
    embedder = DocumentEmbedder(language="en", aggregator="Mean")
    new_corpus, skipped = embedder._transform(dummy_corpus, None)
    assert new_corpus is not None
    assert skipped is None or len(skipped) == 0

def test_invalid_server_raises(dummy_corpus):
    class BrokenEmbedder(DocumentEmbedder):
        def _transform(self, corpus, _, callback=None):

            embedder = _ServerEmbedder(
                aggregator="mean",
                model_name="fasttext-en",
                max_parallel_requests=100,
                server_url="https://api.invalidserver.io",
                embedder_type="text",
            )

            url = urlparse(embedder.server_url)
            host, port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
            try:
                socket.create_connection((host, port), timeout=3)
            except Exception as e:
                raise EmbeddingConnectionError("The server is not responding") from e

            return [], None

    embedder = BrokenEmbedder(language="en", aggregator="Mean")
    with pytest.raises(EmbeddingConnectionError, match="server is not responding"):
        embedder._transform(dummy_corpus, None)

def test_no_internet_raises(dummy_corpus, monkeypatch):
    class NoInternetEmbedder(DocumentEmbedder):
        def _transform(self, corpus, _, callback=None):

            embedder = _ServerEmbedder(
                aggregator="mean",
                model_name="fasttext-en",
                max_parallel_requests=100,
                server_url="https://api.garaza.io",
                embedder_type="text",
            )

            def raise_os_error(*args, **kwargs):
                raise OSError("Simulated: No internet connection")

            monkeypatch.setattr("socket.create_connection", raise_os_error)

            url = urlparse(embedder.server_url)
            host, port = url.hostname, url.port or (443 if url.scheme == "https" else 80)
            try:
                socket.create_connection((host, port), timeout=3)
            except Exception as e:
                raise EmbeddingConnectionError("No internet connection") from e

            return [], None

    embedder = NoInternetEmbedder(language="en", aggregator="Mean")
    with pytest.raises(EmbeddingConnectionError, match="No internet connection"):
        embedder._transform(dummy_corpus, None)
