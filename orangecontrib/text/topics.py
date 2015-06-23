from Orange.data.table import Table


class Topics(Table):
    """
        Dummy wrapper for Table so signals can distinguish Topics from Data.
    """
    def __new__(cls, *args, **kwargs):
        """Bypass Table.__new__."""
        return object.__new__(Topics)
