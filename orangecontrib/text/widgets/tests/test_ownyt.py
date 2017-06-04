import unittest

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.widgets.ownyt import OWNYT


class TestGuardian(WidgetTest):
    def test_error_credentials(self):
        """
        Handling error due to password credentials.
        GH-253
        """
        with unittest.mock.patch(
            "Orange.widgets.credentials.CredentialManager.__getattr__",
            side_effect=Exception):
            self.create_widget(OWNYT)


if __name__ == "__main__":
    unittest.main()
