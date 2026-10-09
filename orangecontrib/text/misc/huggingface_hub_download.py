"""HuggingFace Hub download utilities for subprocess-based model downloads.

Provides a tqdm subclass, a low-level subprocess worker, and a high-level
blocking downloader that streams progress via a callback — all without
requiring the caller to manage ``multiprocessing`` directly.
"""
import logging
import multiprocessing as mp
import time
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional

from tqdm.auto import tqdm

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

# Use "spawn" to avoid fork-related issues in GUI applications.
_spawn_ctx = mp.get_context("spawn")

logger = logging.getLogger(__name__)


class HfHubTqdm(tqdm):
    """A tqdm subclass that reports download progress via a callback.

    Used with huggingface_hub's ``tqdm_class`` parameter to bridge
    HF Hub download progress to Orange's progress bar.
    """

    def __init__(self, *args, callback: Optional[Callable] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._callback = callback

    def update(self, n=1):
        super().update(n)
        if self._callback is not None and self.total is not None:
            self._callback(self.n / self.total)


def download_model_in_subprocess(
    repo_id: str,
    filename: str,
    progress_pipe: "Connection",
) -> None:
    """Subprocess worker that downloads a model and streams progress via pipe.

    Call this function in a ``multiprocessing.Process``.  Messages sent
    through *progress_pipe* (parent-side read end):

    ``('progress', float)``
        Fraction completed in [0, 1].
    ``('success', str)``
        Absolute path to the downloaded model file.
    ``('error', str)``
        Error message string.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``).
    filename : str
        Path to the file within the repository (e.g. ``"onnx/model.onnx"``).
    progress_pipe : multiprocessing.Connection
        Write end of a pipe; messages are sent to the parent process.
    """

    def _send_pipe(msg_type: str, msg_data) -> None:
        """Safely send a message through the pipe."""
        try:
            progress_pipe.send((msg_type, msg_data))
        except Exception:
            pass

    try:
        from huggingface_hub import hf_hub_download

        def pipe_callback(progress: float) -> None:
            _send_pipe("progress", progress)

        onnx_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            tqdm_class=partial(HfHubTqdm, callback=pipe_callback),
        )
        _send_pipe("success", onnx_path)
    except Exception as exc:
        _send_pipe("error", str(exc))
    finally:
        try:
            progress_pipe.close()
        except Exception:
            pass


def download_model_with_progress(
    repo_id: str,
    filename: str,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """Download a model from HuggingFace Hub, streaming progress to *callback*.

    This is the high-level entry point for client code.  It handles all
    multiprocessing plumbing internally and blocks until the download
    completes (or fails).

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``).
    filename : str
        Path to the file within the repository (e.g. ``"onnx/model.onnx"``).
    progress_callback : callable, optional
        A callable that receives a float in [0, 1] representing download
        progress. Called on the main process thread every ~100 ms.

    Returns
    -------
    str
        Absolute path to the downloaded model file.

    Raises
    ------
    Exception
        Re-raises the exception that occurred during download.
    """
    parent_conn, child_conn = _spawn_ctx.Pipe(duplex=False)

    process = _spawn_ctx.Process(
        target=download_model_in_subprocess,
        args=(repo_id, filename, child_conn),
    )
    process.start()
    child_conn.close()

    model_path: Optional[str] = None
    try:
        while process.is_alive():
            while parent_conn.poll(0):
                try:
                    msg_type, msg_data = parent_conn.recv()
                except EOFError:
                    break

                if msg_type == "progress":
                    if progress_callback is not None:
                        progress_callback(msg_data)
                elif msg_type == "error":
                    raise Exception(msg_data)
                elif msg_type == "success":
                    model_path = msg_data
            time.sleep(0.1)

        process.join()
        parent_conn.close()

        # Verify the process exited cleanly and we received a success message.
        if process.exitcode != 0:
            raise RuntimeError("Download subprocess exited with non-zero code")
        if model_path is None:
            raise RuntimeError("Download completed but no success message received")
    except KeyboardInterrupt:
        logger.info("Download cancelled by user")
        process.terminate()
        process.join()
        parent_conn.close()
        raise
    except Exception:
        process.terminate()
        process.join()
        parent_conn.close()
        raise

    return model_path  # type: ignore[return-value]


def is_model_downloaded(repo_id: str, filename: str) -> bool:
    """Check whether a model file is already downloaded in the HuggingFace cache.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id (e.g. ``"sentence-transformers/all-MiniLM-L6-v2"``).
    filename : str
        Path to the file within the repository (e.g. ``"onnx/model.onnx"``).

    Returns
    -------
    bool
        ``True`` if the file exists in the HuggingFace cache, ``False`` otherwise.
    """
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=True,
        )
        return True
    except Exception:
        return False

