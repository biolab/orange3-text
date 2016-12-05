"""
Async Module
============

Helper utils for Orange GUI programming.

Provides :func:`asynchronous` decorator for making methods calls in async mode.
Once method is decorated it will have :func:`task.on_start`, :func:`task.on_result` and :func:`task.callback` decorators for callbacks wrapping.

 - `on_start` must take no arguments
 - `on_result` must accept one argument (the result)
 - `callback` can accept any arguments

For instance::

    class Widget(QObject):
        def __init__(self, name):
            super().__init__()
            self.name = name

        @asynchronous
        def task(self):
            for i in range(3):
                time.sleep(0.5)
                self.report_progress(i)
            return 'Done'

        @task.on_start
        def report_start(self):
            print('`{}` started'.format(self.name))

        @task.on_result
        def report_result(self, result):
            print('`{}` result: {}'.format(self.name, result))

        @task.callback
        def report_progress(self, i):
            print('`{}` progress: {}'.format(self.name, i))


Calling an asynchronous method will launch a daemon thread::

    first = Widget(name='First')
    first.task()
    second = Widget(name='Second')
    second.task()

    first.task.join()
    second.task.join()


A possible output::

    `First` started
    `Second` started
    `Second` progress: 0
    `First` progress: 0
    `First` progress: 1
    `Second` progress: 1
    `First` progress: 2
    `First` result: Done
    `Second` progress: 2
    `Second` result: Done


In order to terminate a thread either call :meth:`stop` method or raise :exc:`StopExecution` exception within :meth:`task`::

    first.task.stop()

"""

import threading
from functools import wraps, partial
from AnyQt.QtCore import pyqtSlot as Slot, QMetaObject, Qt, Q_ARG, QObject


class CallbackMethod:
    def __init__(self, master, instance):
        self.instance = instance
        self.master = master

    def __call__(self, *args, **kwargs):
        QMetaObject.invokeMethod(self.master, 'call', Qt.QueuedConnection,
                                 Q_ARG(object, (self.instance, args, kwargs)))


class CallbackFunction(QObject):
    """ PyQt replacement for ordinary function. Will be always called in the main GUI thread (invoked). """

    def __init__(self, func):
        super().__init__()
        self.func = func

    @Slot(object)
    def call(self, scope):
        instance, args, kwargs = scope
        self.func.__get__(instance, type(instance))(*args, **kwargs)

    def __get__(self, instance, owner):
        return CallbackMethod(self, instance)


def callback(func):
    """ Wraps QObject's method and makes its calls always invoked. """
    return wraps(func)(CallbackFunction(func))


class StopExecution(Exception):
    """ An exception to stop execution of thread's inner cycle. """
    pass


class BoundAsyncMethod(QObject):
    def __init__(self, func, instance):
        super().__init__()
        self.im_func = func
        self.im_self = instance

        self.running = False
        self._thread = None

    def __call__(self, *args, **kwargs):
        self.stop()
        self.running = True
        self._thread = threading.Thread(target=self.run, args=args, kwargs=kwargs,
                                        daemon=True)
        self._thread.start()

    def run(self, *args, **kwargs):
        if self.im_func.start_callback:
            QMetaObject.invokeMethod(self.im_self, self.im_func.start_callback,
                                     Qt.QueuedConnection)
        if self.im_self:
            args = (self.im_self,) + args

        try:
            result = self.im_func.method(*args, **kwargs)
        except StopExecution:
            result = None

        if self.im_func.finish_callback:
            QMetaObject.invokeMethod(self.im_self, self.im_func.finish_callback,
                                     Qt.QueuedConnection, Q_ARG(object, result))
        self.running = False

    def stop(self):
        """ Terminates thread execution. """
        self.running = False
        self.join()

    def join(self):
        """ Waits till task is completed. """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()

    def should_break(self):
        return not self.running


class AsyncMethod(QObject):
    def __init__(self, method):
        super().__init__()
        self.method = method
        self.method_name = method.__name__
        self.finish_callback = None
        self.start_callback = None

    def __get__(self, instance, owner):
        """ Bounds methods with instance. """
        bounded = BoundAsyncMethod(self, instance)
        setattr(instance, self.method.__name__, bounded)
        return bounded

    def on_start(self, callback):
        """ On start callback decorator. """
        self.start_callback = callback.__name__
        return Slot()(callback)

    def callback(self, method=None, should_raise=True):
        """ Callback decorator. Add checks for thread state.

        Raises:
             StopExecution: If thread was stopped (`running = False`).
        """
        if method is None:
            return partial(self.callback, should_raise=should_raise)

        async_method = callback(method)

        @wraps(method)
        def wrapper(instance, *args, **kwargs):
            # This check must take place in the background thread.
            if should_raise and not getattr(instance, self.method_name).running:
                raise StopExecution
            # This call must be sent to the main thread.
            return async_method.__get__(instance, method)(*args, **kwargs)

        return wrapper

    def on_result(self, callback):
        """ On result callback decorator. """
        self.finish_callback = callback.__name__
        return Slot(object)(callback)


def asynchronous(task):
    """ Wraps method of a QObject and replaces it with :class:`AsyncMethod` instance
    in order to run this method in a separate thread.
    """
    return wraps(task)(AsyncMethod(task))
