"""
Helper module for running ONNX runtime inference in an isolated subprocess.

This module addresses onnxruntime native library conflicts when used in
GUI applications (see
https://www.riverbankcomputing.com/pipermail/pyqt/2025-November/046378.html).

The ``ONNXInferenceSession`` class encapsulates all subprocess plumbing,
providing a clean, ergonomic API for running ONNX inference:

    with ONNXInferenceSession(model_path) as session:
        embeddings = session.run(input_dict)

It can also be used directly with ``with`` or by calling ``close()``.
"""
from __future__ import annotations

import os
import logging
import warnings

import multiprocessing
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)


class _WorkerSession:
    """Session state managed inside the subprocess worker.

    This class lives in the runner module so that ``onnxruntime`` is never
    imported in the main process — it is only loaded inside the child
    process where the ``InferenceSession`` is created.
    """

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model: Optional["ort.InferenceSession"] = None

    def initialize(self) -> None:
        """Lazily initialize the ONNX session."""
        if self.model is not None:
            return
        # https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
        import onnxruntime as ort

        options = ort.SessionOptions()
        # https://github.com/microsoft/onnxruntime/issues/22271
        options.enable_cpu_mem_arena = False

        self.model = ort.InferenceSession(self.model_path, sess_options=options)

    def run(self, inputs: Dict[str, np.ndarray]) -> Sequence[np.ndarray]:
        """Run ONNX inference."""
        self.initialize()
        assert self.model is not None
        return self.model.run(None, inputs)


# Global session instance — only accessed inside the worker subprocess.
_session: Optional[_WorkerSession] = None


def _worker_init(model_path: str) -> None:
    """Initializer for the subprocess worker."""
    global _session
    _session = _WorkerSession(model_path)


def _worker_run(inputs: dict[str, np.ndarray]) -> Sequence[np.ndarray]:
    """Entry point for inference in the subprocess worker."""
    assert _session is not None
    return _session.run(inputs)


class ONNXInferenceSession:
    """Run ONNX inference in an isolated subprocess.

    This class manages a single-process multiprocessing pool (using the
    ``spawn`` start method) to isolate the ONNX ``InferenceSession`` from
    the main process, avoiding native library conflicts.

    Usage as a context manager::

        with ONNXInferenceSession(model_path) as session:
            embeddings = session.run(input_dict)
        # Pool is automatically cleaned up

    Or explicitly::

        session = ONNXInferenceSession(model_path)
        try:
            embeddings = session.run(input_dict)
        finally:
            session.close()

    Parameters
    ----------
    model_path : str
        Path to the ONNX model file.
    """

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._pool: multiprocessing.Pool | None = None
        self._input_names: list[str] = []
        self._embedding_dim: int = 0
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Initialize the subprocess pool if not already done."""
        if self._initialized:
            return

        import onnx

        # Probe model metadata in the main process
        onnx_model = onnx.load(self._model_path)
        self._input_names = [inp.name for inp in onnx_model.graph.input]

        output = onnx_model.graph.output[0]
        shape = output.type.tensor_type.shape
        # Find the first non-zero dimension; typically index 2 (hidden_dim)
        dim_value = 0
        for dim in shape.dim:
            if dim.dim_value is not None and dim.dim_value > 0:
                dim_value = dim.dim_value
        self._embedding_dim = dim_value

        # Create spawn-context pool with a single worker process
        context = multiprocessing.get_context("spawn")
        self._pool = context.Pool(
            processes=1,
            initializer=_worker_init,
            initargs=(self._model_path,),
        )
        self._initialized = True

    @property
    def input_names(self) -> list[str]:
        """List of input names expected by the ONNX model."""
        self._ensure_initialized()
        return self._input_names

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (hidden size) of the model output."""
        self._ensure_initialized()
        if self._embedding_dim == 0:
            warnings.warn(
                "Could not infer embedding dimension", RuntimeWarning,
                stacklevel=2
            )
        return self._embedding_dim

    def run(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        """Run ONNX inference in the subprocess.

        Parameters
        ----------
        inputs : dict
            Dictionary mapping input names to numpy arrays.

        Returns
        -------
        ndarray
            The model output (typically ``last_hidden_state`` of shape
            ``(batch_size, sequence_length, embedding_dim)``).
        """
        self._ensure_initialized()

        # Filter inputs to only those the model accepts
        filtered = {k: v for k, v in inputs.items() if k in self._input_names}
        assert self._pool is not None
        result = self._pool.apply(_worker_run, (filtered,))
        return result

    def close(self) -> None:
        """Close the subprocess pool and clean up resources."""
        if self._pool is not None:
            try:
                self._pool.close()
                self._pool.join()
            except Exception:
                logger.exception("Error closing ONNXInferenceSession pool")
            self._pool = None
            self._initialized = False

    def __enter__(self) -> "ONNXInferenceSession":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit context manager, cleaning up the subprocess pool."""
        self.close()

    def __del__(self) -> None:
        """Ensure cleanup on garbage collection."""
        if self._pool is not None:
            warnings.warn(
                "ONNXInferenceSession was not properly closed. "
                "The subprocess pool may still be running.",
                ResourceWarning,
            )
            try:
                self.close()
            except Exception:
                logger.exception("Error during ONNXInferenceSession cleanup")

