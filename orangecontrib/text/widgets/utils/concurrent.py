import threading
from functools import wraps
from PyQt4.QtCore import pyqtSlot as Slot, QMetaObject, Qt, Q_ARG

from Orange.widgets.widget import OWWidget


class StopExecution(Exception):
    pass


class OWConcurrentWidget(OWWidget):
    """ This widget helps with running task in a separate thread.

    Create a method and wrap it with `asynchronous` decorator in order to execute it in the background.

    You can also use `on_start`, `on_result` and `on_progress` callback to modify widgets' gui.

    """
    running = False
    _thread = None

    @Slot()
    def _on_start(self):
        self.progressBarInit(None)
        self.on_start()

    def on_start(self):
        """ This method can be used to clean the widget from previous result. """

    @Slot(float)
    def _on_progress(self, p):
        self.on_progress(p)

    def on_progress(self, progress):
        """ Overwrite this method in order to report valid progress.

        Notes:
            It's not recommended to trigger processing of the event in `progressBarSet`.
            To avoid it pass `processEvents=None`.
        """
        self.progressBarSet(progress, processEvents=None)

    @Slot(object)
    def _on_result(self, result):
        self.progressBarFinished(None)
        self.on_result(result)

    def on_result(self, result):
        """ The method is designed to show user the result of the task execution
        and send proper values to other widgets"""

    def stop(self):
        """ Use this method to terminate thread execution. """
        self.running = False
        self.join()

    def join(self):
        """ Waits till task is completed. """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def progressBarSet(self, value, processEvents=None):
        """ Changes default processEvents value. """
        super().progressBarSet(value, processEvents=processEvents)


def optional_args(fn):
    """
    Decorator used for decorating other decorators to enable optional arguments.
    I.e. the other decorator can then be called either as `@wrap` or as `@wrap(args)`.

    Adopted from: http://stackoverflow.com/a/20966822/892987
    """
    def wrapped_decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]):
            return fn(args[0])
        else:
            def real_decorator(decorate):
                return fn(decorate, *args, **kwargs)
            return real_decorator

    return wrapped_decorator


@optional_args
def asynchronous(method, allow_partial_results=False):
    """
    This decorator wraps method of a OWConcurrentWidget and runs this method in a separate thread.
    It also calls `on_start`, `on_progress` and `on_result` callbacks of the master widgets.

    Args:
        allow_partial_results (bool): If set, allow for partial results. Consequently, wrapped
            method is responsible for checking `self.running` and returning the partial result.
            Otherwise, the main widget should just call stop() and the wrapper will taker care
            of the rest.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.stop()
        self.running = True

        def on_progress(i):
            if not self.running and not allow_partial_results:
                raise StopExecution
            QMetaObject.invokeMethod(self, "_on_progress", Qt.QueuedConnection,  Q_ARG(float, i))

        def should_break():
            return not self.running

        def func():
            try:
                QMetaObject.invokeMethod(self, "_on_start", Qt.QueuedConnection)
                if allow_partial_results:
                    kwargs['should_break'] = should_break
                res = method(self, *args, on_progress=on_progress, **kwargs)
            except StopExecution:
                res = None

            QMetaObject.invokeMethod(self, "_on_result", Qt.QueuedConnection,
                                     Q_ARG(object, res))
            self.running = False

        self._thread = threading.Thread(target=func, daemon=True)
        self._thread.start()
        return None

    return wrapper
