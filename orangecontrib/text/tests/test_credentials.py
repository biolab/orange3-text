import unittest

from orangecontrib.text.credentials import CredentialManager


class CredentialManagerTests(unittest.TestCase):
    def test_credential_manager(self):
        cm = CredentialManager('Foo')
        cm.key = 'Bar'
        self.assertEqual(cm.key, 'Bar')
        cm.delete_password()
        self.assertEqual(cm.key, None)
