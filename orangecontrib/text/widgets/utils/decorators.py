import functools


def gui_require(attribute, message):
    """
    Args:
        attribute: An attribute to be checked
        message (str): Attribute name of the message
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            is_error = hasattr(self.Error, message)
            binded_to = next(i for i in (self.Error, self.Warning, self.Information)
                             if hasattr(i, message))
            getattr(binded_to, message).clear()
            if not getattr(self, attribute, None):
                getattr(binded_to, message)()
            if not is_error or getattr(self, attribute, None):
                return func(self, *args, **kwargs)
        return wrapper
    return decorator
