"""Tests for the HuggingFace Hub download utilities in ``orangecontrib.text.misc``."""
import multiprocessing as mp
import unittest
from unittest.mock import patch

from tqdm.auto import tqdm

from orangecontrib.text.misc import download_model_in_subprocess


class DownloadModelWorkerTest(unittest.TestCase):
    """Tests for the ``download_model_in_subprocess`` function."""

    def test_worker_sends_success_message(self):
        """Test that worker sends a success message with the model path."""
        parent_conn, child_conn = mp.Pipe(duplex=False)
        mock_path = "/fake/path/model.onnx"

        with patch(
            "huggingface_hub.hf_hub_download",
            return_value=mock_path,
        ):
            download_model_in_subprocess("test/repo", "model.onnx", child_conn)

        child_conn.close()
        msg_type, msg_data = parent_conn.recv()
        parent_conn.close()
        self.assertEqual(msg_type, "success")
        self.assertEqual(msg_data, mock_path)

    def test_worker_sends_error_message_on_exception(self):
        """Test that worker sends an error message when download fails."""
        parent_conn, child_conn = mp.Pipe(duplex=False)

        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=RuntimeError("network error"),
        ):
            download_model_in_subprocess("test/repo", "model.onnx", child_conn)

        child_conn.close()
        msg_type, msg_data = parent_conn.recv()
        parent_conn.close()
        self.assertEqual(msg_type, "error")
        self.assertIn("network error", msg_data)

    def test_worker_streams_progress_messages(self):
        """Test that worker streams progress messages through the pipe."""
        parent_conn, child_conn = mp.Pipe(duplex=False)

        def mock_hf_hub_download(**kwargs):
            """Mock that simulates tqdm progress updates."""
            tqdm_class = kwargs.get("tqdm_class")
            # Instantiate the tqdm class and simulate progress updates
            pbar = tqdm_class(desc="model.onnx", total=100, unit="B")
            for progress in [0.25, 0.5, 0.75, 1.0]:
                pbar.update(int(progress * 100) - pbar.n)
            pbar.close()
            return "/fake/path/model.onnx"

        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=mock_hf_hub_download,
        ):
            download_model_in_subprocess("test/repo", "model.onnx", child_conn)

        child_conn.close()
        # Drain all messages
        messages = []
        while parent_conn.poll(0):
            try:
                messages.append(parent_conn.recv())
            except EOFError:
                break
        parent_conn.close()

        # Check that progress messages were sent
        progress_msgs = [m for m in messages if m[0] == "progress"]
        self.assertEqual(len(progress_msgs), 4)
        self.assertEqual(progress_msgs[0], ("progress", 0.25))
        self.assertEqual(progress_msgs[1], ("progress", 0.5))
        self.assertEqual(progress_msgs[2], ("progress", 0.75))
        self.assertEqual(progress_msgs[3], ("progress", 1.0))


if __name__ == "__main__":
    unittest.main()
