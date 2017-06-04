import unittest

from Orange.widgets.tests.base import WidgetTest

from orangecontrib.text.widgets.owguardian import OWGuardian


class TestGuardian(WidgetTest):
    def test_error_credentials(self):
        """
        Handling error due to password credentials.
        GH-253
        """
        with unittest.mock.patch(
            "keyring.get_password",
            side_effect=ValueError):
            self.create_widget(OWGuardian)


if __name__ == "__main__":
    unittest.main()
