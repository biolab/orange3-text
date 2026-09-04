"""Tests for the HuggingFace Hub download utilities in ``orangecontrib.text.misc``."""
import multiprocessing as mp
import unittest
from unittest.mock import patch


from orangecontrib.text.misc.huggingface_hub_download import (
    download_model_in_subprocess, download_model_with_progress
)

_spawn_ctx = mp.get_context("spawn")
_spawn_ctx_Process = _spawn_ctx.Process

def mock_hf_hub_download_progress(repo_id, filename, **kwargs):
    """Mock that simulates tqdm progress updates."""
    tqdm_class = kwargs.get("tqdm_class")
    # Instantiate the tqdm class and simulate progress updates
    pbar = tqdm_class(desc=filename, total=100, unit="B")
    for progress in [0.25, 0.5, 0.75, 1.0]:
        pbar.update(int(progress * 100) - pbar.n)
    pbar.close()
    return "/fake/path/" + filename

def mock_hf_hub_download_raise_error(*args, **kwargs):
    """Mock that simulates error"""
    raise RuntimeError("network error")


def _bootstrap_download_model_with_progress(
        mock_hf_hub_download, *args
):
    """Bootstrap the test on subprocess side `mock_hf_hub_download`"""
    with patch("huggingface_hub.hf_hub_download") as mock:
        mock.side_effect = mock_hf_hub_download
        download_model_in_subprocess(*args)


class DownloadInSubprocessTest(unittest.TestCase):
    @patch("orangecontrib.text.misc.huggingface_hub_download._spawn_ctx.Process")
    def test(self, mock_ctx):
        def Process(target, args):
            return _spawn_ctx_Process(
                target=_bootstrap_download_model_with_progress,
                args=(mock_hf_hub_download_progress, *args)
            )
        mock_ctx.side_effect = Process

        progress_values = []
        download_model_with_progress("repo", "model.onnx", progress_callback=progress_values.append)
        self.assertEqual(progress_values, [0.25, 0.5, 0.75, 1.0])

        def Process(target, args):
            return _spawn_ctx_Process(
                target=_bootstrap_download_model_with_progress,
                args=(mock_hf_hub_download_raise_error, *args)
            )
        mock_ctx.side_effect = Process
        with self.assertRaisesRegex(Exception, "network error"):
            download_model_with_progress("repo", "model.onnx")


if __name__ == "__main__":
    unittest.main()
